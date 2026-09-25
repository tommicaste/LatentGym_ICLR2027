"""
APIRunner — Run API models (OpenAI, Anthropic, Google) or VLLMModel
across multi-episode trajectories.

For local vLLM/SGLang models with batched inference, use LocalRunner
which wraps SkyRL's SkyRLGymGenerator.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from latentgym.core.multi_episode_env import MultiEpisodeEnv
from latentgym.eval.types import EpisodeOutcome, TrajectoryResult, OutcomeType
from latentgym.eval.model_interface import ModelInterface, ModelResponse

logger = logging.getLogger(__name__)


class APIRunner:
    """Run a single model across complete multi-episode trajectories.

    Captures all data: per-step actions/feedback, per-episode outcomes,
    full conversation, and all metadata from init/step.
    """

    def __init__(self, model: ModelInterface):
        self.model = model

    async def run_trajectory(
        self,
        env: MultiEpisodeEnv,
        seed: int = 0,
        max_total_turns: int = 500,
        success_threshold: float = 0.01,
    ) -> TrajectoryResult:
        """Run a single trajectory to completion.

        Args:
            env: Initialized MultiEpisodeEnv (already has episode_configs)
            seed: Random seed for this trajectory
            max_total_turns: Safety limit on total turns
            success_threshold: Reward >= this is considered a success

        Returns:
            TrajectoryResult with all episode outcomes, step records, and conversation.
        """
        # Init environment
        conversation, init_metadata = env.init([])

        # Extract env config from init metadata
        env_name = init_metadata.get("env_name", "")
        latent_id = init_metadata.get("latent_id", "")
        reward_type = init_metadata.get("reward_type", "")
        max_turns_per_ep = init_metadata.get("max_turns_per_episode", 0)
        env_params = init_metadata.get("env_params", {})
        episode_configs = init_metadata.get("episode_configs", [])

        episode_outcomes: List[EpisodeOutcome] = []
        reasoning_trace: List[Optional[str]] = []
        current_episode = 0
        episode_turn = 0
        final_metadata: Dict[str, Any] = {}

        done = False
        while not done and (episode_turn + sum(o.turns for o in episode_outcomes)) < max_total_turns:
            # Generate model response (ModelResponse with text + reasoning)
            response = await self.model.generate(conversation)

            episode_turn += 1

            # Record reasoning separately (NOT fed back into conversation)
            reasoning_trace.append(response.reasoning)

            # Only the action text goes into conversation context
            conversation.append({"role": "assistant", "content": response.text})

            # Step environment with action text only
            step_result = env.step(response.text)
            obs = step_result["observations"]
            reward = step_result["reward"]
            done = step_result["done"]
            step_metadata = step_result["metadata"]

            # Add observation to conversation
            if obs:
                conversation.extend(obs)

            # Track episode transitions
            new_episode = step_metadata.get("episode", 0)
            ep_rewards = step_metadata.get("episode_rewards", [])
            ep_turns = step_metadata.get("turns_per_episode", [])

            if new_episode != current_episode or done:
                # Record outcomes for any newly completed episodes
                while len(episode_outcomes) < len(ep_rewards):
                    idx = len(episode_outcomes)
                    ep_reward = ep_rewards[idx]
                    ep_turn_count = ep_turns[idx] if idx < len(ep_turns) else episode_turn

                    # Determine outcome type
                    if ep_reward >= success_threshold:
                        outcome_type = OutcomeType.WIN
                    elif ep_turn_count >= max_turns_per_ep > 0:
                        outcome_type = OutcomeType.TIMEOUT
                    elif ep_reward > 0:
                        outcome_type = OutcomeType.PARTIAL
                    else:
                        outcome_type = OutcomeType.LOSS

                    # Get ground truth for this episode from episode_configs
                    gt = episode_configs[idx] if idx < len(episode_configs) else {}

                    episode_outcomes.append(EpisodeOutcome(
                        episode_idx=idx,
                        reward=ep_reward,
                        turns=ep_turn_count,
                        success=ep_reward >= success_threshold,
                        agent_name=self.model.name,
                        latent_id=step_metadata.get("latent_id", latent_id),
                        max_turns=max_turns_per_ep,
                        outcome_type=outcome_type,
                        ground_truth=gt,
                    ))

                current_episode = new_episode
                episode_turn = 0

            # Save final step metadata
            final_metadata = step_metadata

        env.close()

        # Build agent assignments
        agent_assignments = [self.model.name] * len(episode_outcomes)

        return TrajectoryResult(
            episode_outcomes=episode_outcomes,
            conversation=conversation,
            model_name=self.model.name,
            benchmark_id=init_metadata.get("benchmark_id", ""),
            seed=seed,
            env_name=env_name,
            latent_id=latent_id,
            prompt_id=init_metadata.get("prompt_id", ""),
            feedback_id=init_metadata.get("feedback_id", ""),
            reward_type=reward_type,
            max_turns_per_episode=max_turns_per_ep,
            env_params=env_params,
            agent_assignments=agent_assignments,
            episode_configs=episode_configs,
            reasoning_trace=reasoning_trace,
            init_metadata=init_metadata,
            final_metadata=final_metadata,
        )
