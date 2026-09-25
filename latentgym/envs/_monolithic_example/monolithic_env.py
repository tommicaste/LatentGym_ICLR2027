"""
Monolithic example: Number Guessing in one file.

Demonstrates how to put everything (core env, latents, prompt, feedback)
in a single file. This is an alternative to the split-file approach used
by the other envs. Both approaches work identically.

Usage:
    # Import this file to trigger registration
    import latentgym.envs._monolithic_example.monolithic_env  # noqa

    # Then use via registry
    from latentgym.core.registry import make_env
    from latentgym.core.env_config import FullyDefinedEnv
    from latentgym.core.reward import RewardType

    config = FullyDefinedEnv(
        env_name="numguess_mono",
        latent_id="easy_range",
        prompt_id="default",
        feedback_id="default",
        num_episodes=5,
        reward_type=RewardType.CUMULATIVE,
    )
    env = make_env(config, trajectory_path="path/to/traj.json")

The key difference vs split-file:
    - Everything is in one place (easier to read for a new env)
    - Trade-off: file gets long as you add latents/prompts/feedbacks
    - For production envs with many latents, split-file is preferred
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, Tuple

from latentgym.core.single_episode_env import SingleEpisodeEnv
from latentgym.core.latent import LatentDefinition, LatentComplexity
from latentgym.core.prompt import PromptTemplate
from latentgym.core.feedback import FeedbackHandler
from latentgym.core.registry import register_env, register_latent, register_prompt, register_feedback

ENV_NAME = "numguess_mono"

# =============================================================================
# 1. Core Env — game dynamics
# =============================================================================

class NumberGuessingEnv(SingleEpisodeEnv):
    """Single-episode number guessing: guess a target integer.

    episode_config keys:
        target_number (int): the number to guess
        min_range (int): lower bound shown to player (default 1)
        max_range (int): upper bound shown to player (default 100)
        max_turns (int): max guesses allowed (default 10)
    """

    def __init__(self):
        self._target = 0
        self._min = 1
        self._max = 100
        self._max_turns = 10
        self._turns = 0
        self._done = False

    def reset(self, episode_config: Dict[str, Any]) -> str:
        self._target = episode_config["target_number"]
        self._min = episode_config.get("min_range", 1)
        self._max = episode_config.get("max_range", 100)
        self._max_turns = episode_config.get("max_turns", 10)
        self._turns = 0
        self._done = False
        return (
            f"I'm thinking of a number between {self._min} and {self._max}. "
            f"You have {self._max_turns} guesses. Make a guess!"
        )

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        if self._done:
            return "Game already over.", 0.0, True, {}

        # Extract number from action
        nums = re.findall(r"-?\d+", action)
        if not nums:
            self._turns += 1
            feedback = f"I didn't understand that. Please guess a number between {self._min} and {self._max}."
            done = self._turns >= self._max_turns
            return feedback, 0.0, done, {}

        guess = int(nums[-1])
        self._turns += 1

        if guess == self._target:
            self._done = True
            # Reward: higher for fewer turns
            efficiency = 1.0 - (self._turns - 1) / max(self._max_turns - 1, 1)
            reward = 0.5 + 0.5 * efficiency
            return f"Correct! The number was {self._target}. You got it in {self._turns} guess(es)!", reward, True, {"success": True}

        if self._turns >= self._max_turns:
            self._done = True
            return f"Out of guesses! The number was {self._target}.", 0.0, True, {"success": False}

        direction = "higher" if self._target > guess else "lower"
        remaining = self._max_turns - self._turns
        feedback = f"Wrong! Go {direction}. You have {remaining} guess(es) left."
        return feedback, 0.0, False, {}

    def get_game_rules(self) -> str:
        return (
            f"You are playing Number Guessing. A secret number between "
            f"{self._min} and {self._max} has been chosen. "
            f"Each turn, guess a number and you'll be told if you should go higher or lower. "
            f"Win by guessing the exact number within {self._max_turns} turns. "
            f"To make a guess, write just the number (e.g.: 42)."
        )


# =============================================================================
# 2. Prompt Templates — vary how much meta-info the agent sees
# =============================================================================

class DefaultPrompt(PromptTemplate):
    """Standard prompt: just the game rules, no latent hints."""
    id = "default"

    def initial_system_prompt(self, game_rules: str, env_params: Dict, num_episodes: int) -> str:
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} episode(s). "
            "Learn from each episode to improve your strategy."
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        return (
            f"\n--- Episode {ep_idx + 1} of {num_episodes} ---\n"
            f"Previous episode reward: {prev_episode_reward:.4f}\n"
            "A new number has been chosen. Good luck!"
        )


class HintPrompt(PromptTemplate):
    """Hint prompt: tells agent the range constraint stays the same across episodes."""
    id = "hint"

    def initial_system_prompt(self, game_rules: str, env_params: Dict, num_episodes: int) -> str:
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} episode(s). "
            "HINT: The target number is always drawn from the same fixed range across all episodes. "
            "Use this to your advantage!"
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        return (
            f"\n--- Episode {ep_idx + 1} of {num_episodes} ---\n"
            f"Previous: {prev_episode_reward:.4f}. New number chosen from the same range."
        )


# =============================================================================
# 3. Feedback Handlers — vary how step feedback is formatted
# =============================================================================

class DefaultFeedback(FeedbackHandler):
    """Pass-through: show raw env feedback as-is."""
    id = "default"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict) -> str:
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx: int, episode_reward: float,
                                     episode_info: Dict) -> str:
        result = "Won!" if episode_reward > 0 else "Lost."
        return f"Episode {episode_idx} complete. {result} (reward: {episode_reward:.4f})"


class VerboseFeedback(FeedbackHandler):
    """Verbose: add turn counter and episode context to each message."""
    id = "verbose"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict) -> str:
        return f"[Episode {episode_idx}, Turn {turn}] {raw_feedback}"

    def format_episode_end_feedback(self, episode_idx: int, episode_reward: float,
                                     episode_info: Dict) -> str:
        result = "Won!" if episode_reward > 0 else "Lost."
        return (
            f"Episode {episode_idx} ended. {result}\n"
            f"Reward: {episode_reward:.4f} (higher = fewer guesses needed)"
        )


# =============================================================================
# 4. Latents — constraint generators
# =============================================================================

def _range_generator(lo: int, hi: int):
    """Generator: target always drawn from [lo, hi]."""
    def gen(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
        return {
            "target_number": random.randint(lo, hi),
            "min_range": lo,
            "max_range": hi,
            "max_turns": env_params.get("max_turns", 10),
        }
    return gen


def _fixed_set_generator(values: list):
    """Generator: target always drawn from a fixed set of values."""
    def gen(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
        return {
            "target_number": random.choice(values),
            "min_range": min(values),
            "max_range": max(values),
            "max_turns": env_params.get("max_turns", 10),
        }
    return gen


def _powers_of_two_generator():
    """Generator: target is always a power of 2 within [1, 128]."""
    powers = [1, 2, 4, 8, 16, 32, 64, 128]
    def gen(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
        return {
            "target_number": random.choice(powers),
            "min_range": 1,
            "max_range": 128,
            "max_turns": env_params.get("max_turns", 7),
        }
    return gen


# =============================================================================
# 5. Registration
# =============================================================================

# Register the env class
register_env(ENV_NAME, NumberGuessingEnv)

# Register latents
register_latent(ENV_NAME, LatentDefinition(
    id="easy_range",
    name="Easy Range [1-20]",
    complexity=LatentComplexity.EASY,
    description="Target is always in [1, 20]",
    generator_fn=_range_generator(1, 20),
))

register_latent(ENV_NAME, LatentDefinition(
    id="medium_range",
    name="Medium Range [1-100]",
    complexity=LatentComplexity.MEDIUM,
    description="Target is always in [1, 100]",
    generator_fn=_range_generator(1, 100),
))

register_latent(ENV_NAME, LatentDefinition(
    id="hard_range",
    name="Hard Range [1-1000]",
    complexity=LatentComplexity.HARD,
    description="Target is always in [1, 1000]",
    generator_fn=_range_generator(1, 1000),
))

register_latent(ENV_NAME, LatentDefinition(
    id="fixed_set_small",
    name="Fixed Set of 5",
    complexity=LatentComplexity.EASY,
    description="Target always drawn from a fixed set of 5 numbers",
    generator_fn=_fixed_set_generator([7, 23, 42, 67, 91]),
))

register_latent(ENV_NAME, LatentDefinition(
    id="powers_of_two",
    name="Powers of Two",
    complexity=LatentComplexity.MEDIUM,
    description="Target is always a power of 2 in [1, 128]",
    generator_fn=_powers_of_two_generator(),
))

register_latent(ENV_NAME, LatentDefinition(
    id="upper_half",
    name="Upper Half [51-100]",
    complexity=LatentComplexity.MEDIUM,
    description="Target is always in the upper half [51, 100]",
    generator_fn=_range_generator(51, 100),
))

register_latent(ENV_NAME, LatentDefinition(
    id="lower_half",
    name="Lower Half [1-50]",
    complexity=LatentComplexity.MEDIUM,
    description="Target is always in the lower half [1, 50]",
    generator_fn=_range_generator(1, 50),
))

# Register prompts
register_prompt(ENV_NAME, DefaultPrompt)
register_prompt(ENV_NAME, HintPrompt)

# Register feedbacks
register_feedback(ENV_NAME, DefaultFeedback)
register_feedback(ENV_NAME, VerboseFeedback)
