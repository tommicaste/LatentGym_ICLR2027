import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from skyrl_train.generators.base import BatchMetadata, GeneratorInput, GeneratorOutput, TrainingPhase, TrajectoryID
from skyrl_train.inference_engines.base import ConversationType

CUSTOM_CHAT_TEMPLATES = {
    # chat template for qwen3 that preserves thinking tokens
    "qwen3_with_thinking": (
        "{% for message in messages %}"
        "{% if (message['role'] != 'assistant') %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% elif (message['role'] == 'assistant')%}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% generation %}"
        "{{message['content'] + '<|im_end|>'}}"
        "{% endgeneration %}"
        "{{'\n'}}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "qwen3_without_thinking": (
        "{% for message in messages %}"
        "{% if (message['role'] != 'assistant') %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% elif (message['role'] == 'assistant')%}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% generation %}"
        "{% set full_content = message['content'] %}"
        "{% set mycontent = message['content'] %}"
        "{% set is_last_message = loop.last and messages[-1]['role'] == 'assistant' %}"
        "{% if '</think>' in full_content and not is_last_message %}"
        "{% set mycontent = full_content.split('</think>')[-1].lstrip('\n') %}"
        "{% endif %}"
        "{{mycontent + '<|im_end|>'}}"
        "{% endgeneration %}"
        "{{'\n'}}"
        "{% endif %}"
        "{% endfor %}"
    ),
}


def get_custom_chat_template(chat_template_config: Optional[Union[dict, DictConfig]] = None) -> Optional[str]:
    """
    Get custom chat template based on the new config structure.

    Args:
        chat_template_config: Config dict with 'source' and 'name_or_path' fields.

    Returns:
        Chat template string or None
    """
    if chat_template_config is None:
        return None

    source = chat_template_config.get("source")
    if not source:
        raise ValueError("'source' is required in chat_template_config")

    name_or_path = chat_template_config.get("name_or_path")
    if not name_or_path:
        return None  # if name_or_path is not provided, use the default chat template from the tokenizer

    if source == "name":
        if name_or_path in CUSTOM_CHAT_TEMPLATES:
            return CUSTOM_CHAT_TEMPLATES[name_or_path]
        else:
            raise ValueError(
                f"Template name '{name_or_path}' not found. Available templates: {list(CUSTOM_CHAT_TEMPLATES.keys())}"
            )
    elif source == "file":
        try:
            with open(name_or_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError as e:
            raise ValueError(f"Template file '{name_or_path}' not found") from e
        except OSError as e:
            raise ValueError(f"Error reading template file '{name_or_path}': {e}") from e
    else:
        raise ValueError(f"Invalid source '{source}'. Must be 'name' or 'file'")


def get_generation_prompt_ids(tokenizer) -> List[int]:
    """
    Helper function to get the generation prompt ids for a given tokenizer.
    """
    empty_user = tokenizer.apply_chat_template([{"role": "user", "content": ""}], tokenize=True)
    empty_user_with_generation_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True
    )

    generation_prompt_ids = empty_user_with_generation_prompt[len(empty_user) :]
    return generation_prompt_ids


@torch.no_grad()
def get_metrics_from_generator_output(generator_output: GeneratorOutput, uids: List[str]) -> Tuple[float, float]:
    """
    Get `mean_raw_reward` (or avg_score), `pass_at_n` from generator output.

    The `n` in `pass_at_n` is the number of trajectories we generate for each example. It is
    calculated as `len(generator_output["rewards"]) / len(uids)`, where `len(uids)` is the number of
    unique examples.

    Rewards can be either per-trajectory or per-token, and metrics are computed correspondingly.
    """
    rewards: Union[List[float], List[List[float]]] = generator_output["rewards"]
    if not len(rewards):
        raise ValueError(f"`rewards` must be a non-empty list, got {rewards}")

    # TODO: We should make metrics customizable by the environment.
    # Map from the example's uid to each trajectory's reward on that same example
    uid_to_trajectory_rewards = defaultdict(list)
    if isinstance(rewards[0], list):
        # Token-level rewards: rewards is List[List[float]]
        # For each trajectory, we sum over the token rewards for `mean_raw_reward` computation
        mean_raw_reward = float(np.mean([sum(trajectory_rewards) for trajectory_rewards in rewards]))
        # Assume the last token's reward signifies the trajectory's reward for `pass_at_n` computation
        for i, cur_trajectory_rewards in enumerate(rewards):
            if len(cur_trajectory_rewards) == 0:
                raise ValueError("Token-level rewards must be a non-empty list.")
            uid_to_trajectory_rewards[uids[i]].append(cur_trajectory_rewards[-1])
    else:
        mean_raw_reward = float(np.mean(rewards))
        for i, reward in enumerate(rewards):
            uid_to_trajectory_rewards[uids[i]].append(reward)

    # For each trajectory, if the reward is positive, then it's a "pass". So for a single example, if
    # any of its trajectories' reward is positive, pass@n for that uid is 1.
    pass_at_n = sum(1 for v in uid_to_trajectory_rewards.values() if any(r > 0 for r in v)) / len(
        uid_to_trajectory_rewards
    )

    return mean_raw_reward, pass_at_n


@torch.no_grad()
def get_multi_episode_metrics_from_generator_output(generator_output: GeneratorOutput) -> Tuple[float, float]:
    """
    Most metrics are calculated based on complete trajectories across multiple episodes.
    """
    token_rewards = generator_output["rewards"]

    if not isinstance(token_rewards[0], list):
        raise ValueError("Expected rewards to be List[List[float]] for multi-episode environments.")

    num_episodes = generator_output["metadatas"][0]["total_episodes"]  # Assume all have same num_episodes

    per_episode_rewards: List[List[float]] = [metadata["episode_rewards"] for metadata in generator_output["metadatas"]]
    padded_per_episode_rewards = [rewards + [0.0] * (num_episodes - len(rewards)) for rewards in per_episode_rewards]
    per_episode_rewards = np.array(padded_per_episode_rewards)

    # Calculate how many trajectory did not complete all episodes
    num_trajectories = len(generator_output["metadatas"])
    incomplete_mask = np.array(
        [1 if len(metadata["episode_rewards"]) < num_episodes else 0 for metadata in generator_output["metadatas"]]
    )
    prop_completed_trajectories = (num_trajectories - np.sum(incomplete_mask)) / num_trajectories

    ## Calculate
    # 1. avg number of episodes solved
    num_episodes_solved = np.sum(per_episode_rewards > 0, axis=1)
    mean_num_episodes_solved = float(np.mean(num_episodes_solved))

    # 2. difference from last to first episode in rewards (avg improvement)
    # Note: calculating the difference only on completed trajectories
    per_episode_rewards_masked = per_episode_rewards[incomplete_mask == 0]
    improvement = per_episode_rewards_masked[:, -1] - per_episode_rewards_masked[:, 0]
    mean_improvement = float(np.mean(improvement))
    avg_initial_reward = float(np.mean(per_episode_rewards_masked[:, 0]))
    avg_final_reward = float(np.mean(per_episode_rewards_masked[:, -1]))

    # 3. avg improvement rate (above / # of episodes)
    mean_improvement_rate = mean_improvement / (num_episodes - 1) if num_episodes > 1 else 0.0

    # 4. avg number of turns per episode
    turns_per_episode = [metadata["turns_per_episode"] for metadata in generator_output["metadatas"]]
    padded_turns_per_episode = [tpe + [0] * (num_episodes - len(tpe)) for tpe in turns_per_episode]
    padded_turns_per_episode_masked = np.array(padded_turns_per_episode)[incomplete_mask == 0]
    mean_turns_per_episode = np.mean(padded_turns_per_episode_masked, axis=0).tolist()

    # 5. difference from last to first episode in number of turns
    final_reduction_in_turns = mean_turns_per_episode[0] - mean_turns_per_episode[-1]

    return {
        "avg_num_episodes_solved": mean_num_episodes_solved,
        "avg_reward_improvement": mean_improvement,
        "avg_reward_improvement_rate": mean_improvement_rate,
        "avg_initial_reward": avg_initial_reward,
        "avg_final_reward": avg_final_reward,
        "final_reduction_in_turns": final_reduction_in_turns,
        "prop_completed_trajectories": prop_completed_trajectories,
        **{f"avg_turns_episode_{i}": mean_turns_per_episode[i] for i in range(num_episodes)},
        **{f"avg_reward_episode_{i}": float(np.mean(per_episode_rewards_masked[:, i])) for i in range(num_episodes)},
    }


def concatenate_generator_outputs(generator_outputs: List[GeneratorOutput]) -> GeneratorOutput:
    """
    Used in eval to concatenate the generator outputs of multiple batches.

    `rollout_metrics` are not concatenated because they are already aggregated.
    """
    assert len(generator_outputs) > 0
    has_rollout_logprobs = [output.get("rollout_logprobs") is not None for output in generator_outputs]
    if any(has_rollout_logprobs) and not all(has_rollout_logprobs):
        raise ValueError(
            "generator outputs are expected to all have null rollout_logprobs or all non-null, but received a mix"
        )
    result: GeneratorOutput = {
        "prompt_token_ids": sum([output["prompt_token_ids"] for output in generator_outputs], []),
        "response_ids": sum([output["response_ids"] for output in generator_outputs], []),
        "rewards": sum([output["rewards"] for output in generator_outputs], []),
        "loss_masks": sum([output["loss_masks"] for output in generator_outputs], []),
        "rollout_logprobs": (
            sum([output["rollout_logprobs"] for output in generator_outputs], [])
            if generator_outputs[0]["rollout_logprobs"] is not None
            else None
        ),
    }
    if "stop_reasons" in generator_outputs[0] and generator_outputs[0]["stop_reasons"] is not None:
        result["stop_reasons"] = sum([output["stop_reasons"] for output in generator_outputs], [])

    # propagate additional keys with list values as-is
    additional_keys = [
        key for key in generator_outputs[0] if key not in result and isinstance(generator_outputs[0][key], list)
    ]
    if len(additional_keys):
        logger.info(f"Attempting to concatenate values for additional keys {additional_keys}")
    for key in additional_keys:
        result[key] = sum([generator_output[key] for generator_output in generator_outputs], [])

    return result


def apply_overlong_filtering(
    loss_masks: List[List[int]],
    response_ids: List[List[int]],
    eos_token_id: int,
) -> List[List[int]]:
    """
    Implements DAPO Overlong Filtering: zero-out every token's mask whenever
    the response does not end with the eos token id (i.e. truncated).

    Returns:
        - The loss masks with tokens zeroed out for truncated responses
    """
    assert len(loss_masks) == len(response_ids), "loss_masks and response_ids must have the same length"
    return [
        [0] * len(mask) if not response or response[-1] != eos_token_id else mask
        for mask, response in zip(loss_masks, response_ids)
    ]


def get_rollout_metrics(responses: List[List[int]], rewards: Union[List[float], List[List[float]]]):
    num_tokens_arr = np.array([len(response) for response in responses])
    # Support both response-level and token-level rewards
    flat_rewards = []
    for r in rewards:
        if isinstance(r, list):
            flat_rewards.append(float(np.sum(r)))
        else:
            flat_rewards.append(float(r))
    flat_rewards_arr = np.array(flat_rewards)
    non_zero_rewards_arr = flat_rewards_arr > 0.0
    zero_rewards_arr = flat_rewards_arr == 0.0
    # average tokens for non zero rewards
    avg_tokens_non_zero_rewards = (
        np.mean(num_tokens_arr[non_zero_rewards_arr]) if non_zero_rewards_arr.sum() > 0 else np.zeros(1)
    )
    # average tokens for zero rewards
    avg_tokens_zero_rewards = np.mean(num_tokens_arr[zero_rewards_arr]) if zero_rewards_arr.sum() > 0 else np.zeros(1)

    return {
        "generate/min_num_tokens": np.min(num_tokens_arr).item(),
        "generate/max_num_tokens": np.max(num_tokens_arr).item(),
        "generate/avg_num_tokens": np.mean(num_tokens_arr).item(),
        "generate/std_num_tokens": np.std(num_tokens_arr).item(),
        "generate/avg_tokens_non_zero_rewards": avg_tokens_non_zero_rewards.item(),
        "generate/avg_tokens_zero_rewards": avg_tokens_zero_rewards.item(),
    }


def prepare_generator_input(
    prompts: List[Any],
    n_samples_per_prompt: int,
    sampling_params: Dict[str, Any],
    default_env_class: str,
    training_phase: TrainingPhase,
    global_step: int,
) -> Tuple[GeneratorInput, List[str]]:
    """Prepares the generator input for training and eval

    Args:
        prompts (List[Any]): list of prompts
        n_samples_per_prompt (int): how many samples to create per prompt
        sampling_params (Dict[str, Any]): sampling parameters
        default_env_class (str): env class to use if env class missing from prompts
        training_phase (TrainingPhase): training or eval
        global_step (int): current global step

    Returns:
        Tuple[GeneratorInput, List[str]]: generator input and list of uuids
    """

    all_prompts = [prompt["prompt"] for prompt in prompts for _ in range(n_samples_per_prompt)]

    all_envs = [
        prompt["env_class"] if prompt["env_class"] is not None else default_env_class
        for prompt in prompts
        for _ in range(n_samples_per_prompt)
    ]

    # all the other columns are env_extras
    env_extras = [prompt["env_extras"] for prompt in prompts for _ in range(n_samples_per_prompt)]

    # Create TrajectoryID objects - one UID per row, repetition_id for multiple samples
    trajectory_ids = []
    uids = []
    for _, prompt in enumerate(prompts):
        uid: str = prompt["uid"]

        # Create TrajectoryID for each repetition
        for repetition_id in range(n_samples_per_prompt):
            trajectory_ids.append(TrajectoryID(instance_id=uid, repetition_id=repetition_id))
            uids.append(uid)

    generator_input: GeneratorInput = {
        "prompts": all_prompts,
        "env_classes": all_envs,
        "env_extras": env_extras,
        "sampling_params": sampling_params,
        "trajectory_ids": trajectory_ids,
        "batch_metadata": BatchMetadata(global_step=global_step, training_phase=training_phase),
    }

    return generator_input, uids


def encode_messages_subset(messages: ConversationType, tokenizer):
    """Encodes a subset of messages from a multi-turn conversation using the fixed base approach.

    This function tokenizes messages as if they are part of a larger conversation, ensuring
    no additional default system messages are prepended by the tokenizer's chat template

    The "fixed base approach" works by:
    - Creating a dummy base conversation to establish context
    - Appending the target messages to this base
    - Tokenizing the full conversation and extracting only the tokens for the target messages

    For simple chat templates without complex token splitting behavior, this produces the same
    result as directly tokenizing the messages. For templates like Qwen's ChatML format where
    a default system prompt can be appended, this ensures correct tokenization

    Reference: https://jybsuper.github.io/posts/multiturn_tokenization/#the-breakthrough-fixed-base-approach

    Args:
        messages: List of message dicts with 'role' and 'content' keys. Must contain at least
                 one message. These are assumed to be a subset from a larger conversation.
        tokenizer: HuggingFace tokenizer with chat_template support and eos_token_id defined.

    Returns:
        List[int]: Token IDs for the given messages, with proper multi-turn context handling.
    """
    assert len(messages), "messages list cannot be empty"
    # Follows https://jybsuper.github.io/posts/multiturn_tokenization/#the-breakthrough-fixed-base-approach
    # Ends with an assistant turn so appending a user observation keeps roles alternating
    # (required by Gemma's chat template; harmless for Qwen/Llama).
    base_conversation = [
        {"role": "user", "content": "I am a user."},
        {"role": "assistant", "content": "I am an assistant."},
    ]
    base_conversation_token_ids = tokenizer.apply_chat_template(
        base_conversation,
        add_generation_prompt=False,
        tokenize=True,
    )

    full_conversation = base_conversation + messages
    full_conversation_token_ids = tokenizer.apply_chat_template(
        full_conversation,
        add_generation_prompt=False,
        tokenize=True,
    )
    conversation_token_ids = full_conversation_token_ids[len(base_conversation_token_ids) :]
    return conversation_token_ids
