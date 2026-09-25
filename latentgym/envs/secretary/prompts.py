"""Prompt templates for secretary problem: no_info, some_info, full_info.

All prompts include:
- Game rules ([accept] or [continue], win if you pick the max)
- Multi-episode framing

Differences:
- no_info: No hint about cross-episode patterns
- some_info: Vague hint that positions may have a pattern
- full_info: Grouped hints by pattern type (position, threshold, sequential,
  structural, relational, cross-episode, mathematical)
"""

from typing import Any, Dict

from latentgym.core.prompt import PromptTemplate
from latentgym.core.registry import register_prompt


def _get_full_info_hint(latent_id: str) -> str:
    """Return a grouped hint based on latent type."""

    # Position latents — max at a consistent position/region
    position_latents = {
        "best_is_last", "best_is_first", "first_half_bias", "second_half_bias",
        "best_at_even", "best_at_odd",
    }
    # Also fixed_position_0 through fixed_position_4 and best_in_quarter_0 through 3
    prefix = "IMPORTANT: Across games there is a consistent pattern. "

    if latent_id in position_latents or latent_id.startswith("fixed_position_") or latent_id.startswith("best_in_quarter_"):
        return (
            prefix + "The maximum tends to appear at a consistent position or region "
            "in the sequence. Pay attention to where it appears across games."
        )

    # Threshold latents — value-based acceptance
    if latent_id in ("fixed_high", "threshold_08", "threshold_06"):
        return (
            prefix + "There is a value threshold — the maximum is always above it "
            "and other values are below. Discover the threshold to accept immediately."
        )

    # Sequential/shift latents — position changes predictably across episodes
    if latent_id in ("position_shift", "countdown", "alternating_position",
                     "third_rotation") or latent_id.startswith("position_cycle_"):
        return (
            prefix + "The position of the maximum shifts in a predictable pattern "
            "across episodes. Track where it was in previous episodes to predict the next."
        )

    # Structural latents — values follow a shape within each episode
    if latent_id in ("step_function", "sorted_order", "inverse_order",
                     "valley_pattern", "mountain_pattern", "ascending_spike"):
        return (
            prefix + "The values within each sequence follow a structural pattern. "
            "Once you recognize the shape, you'll know where the maximum tends to be."
        )

    # Relational latents — max position relates to other values in the sequence
    if latent_id in ("relative_jump", "max_after_min", "early_decoy"):
        return (
            prefix + "The maximum's position is related to other values in the sequence. "
            "Look for signals in the values you've already seen to predict when the "
            "maximum is coming."
        )

    # Cross-episode memory latents — position depends on previous episode
    if latent_id in ("mirror_episodes", "random_walk_position",
                     "same_position_streak", "increasing_position"):
        return (
            prefix + "The position of the maximum in each episode depends on where "
            "it was in the previous episode. Track the positions across games."
        )

    # Mathematical latents — position follows a math pattern
    if latent_id in ("fibonacci_positions", "modular_pattern", "prime_positions"):
        return (
            prefix + "The position of the maximum follows a mathematical pattern. "
            "The position indices themselves are the clue."
        )

    # Fallback
    return (
        prefix + "The sequences share a hidden pattern that determines where the "
        "maximum appears. Discover this pattern from early games to improve your strategy."
    )


def _get_full_info_transition_hint(latent_id: str) -> str:
    """Return a grouped transition hint."""
    if latent_id.startswith("fixed_position_") or latent_id.startswith("best_in_quarter_") or \
       latent_id in ("best_is_last", "best_is_first", "first_half_bias",
                     "second_half_bias", "best_at_even", "best_at_odd"):
        return "Recall where the maximum appeared in previous games."

    if latent_id in ("fixed_high", "threshold_08", "threshold_06"):
        return "Recall the value threshold you've observed."

    if latent_id in ("position_shift", "countdown", "alternating_position",
                     "third_rotation") or latent_id.startswith("position_cycle_"):
        return "Track how the maximum's position has shifted across games."

    if latent_id in ("step_function", "sorted_order", "inverse_order",
                     "valley_pattern", "mountain_pattern", "ascending_spike"):
        return "Recall the structural pattern in the values."

    if latent_id in ("relative_jump", "max_after_min", "early_decoy"):
        return "Watch for the signal that predicts the maximum."

    if latent_id in ("mirror_episodes", "random_walk_position",
                     "same_position_streak", "increasing_position"):
        return "Use the previous episode's max position to predict this one."

    if latent_id in ("fibonacci_positions", "modular_pattern", "prime_positions"):
        return "Apply the mathematical pattern to predict the position."

    return "Use what you've learned from previous games."


class SecretaryNoInfoPrompt(PromptTemplate):
    id = "no_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        self._env_params = env_params
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} rounds of this game sequentially."
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        return (
            f"\n\n--- Game {ep_idx + 1} complete! ---\n\n"
            f"You just completed game {ep_idx + 1}. "
            f"Starting next game.\n\n"
            f"--- Game {ep_idx + 2} of {num_episodes} ---"
        )


class SecretarySomeInfoPrompt(PromptTemplate):
    id = "some_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        self._env_params = env_params
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} rounds of this game sequentially. "
            f"The value sequences may have a pattern in where the maximum appears. "
            f"Pay attention to previous rounds to improve your strategy."
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        return (
            f"\n\n--- Game {ep_idx + 1} complete! ---\n\n"
            f"You just completed game {ep_idx + 1}. "
            f"Starting next game. Think about where the maximum tends to appear.\n\n"
            f"--- Game {ep_idx + 2} of {num_episodes} ---"
        )


class SecretaryFullInfoPrompt(PromptTemplate):
    id = "full_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        self._env_params = env_params
        latent_id = env_params.get("latent_id", "")
        hint = _get_full_info_hint(latent_id)
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} rounds of this game sequentially.\n\n"
            f"{hint}"
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        latent_id = getattr(self, "_env_params", {}).get("latent_id", "")
        hint = _get_full_info_transition_hint(latent_id)
        return (
            f"\n\n--- Game {ep_idx + 1} complete! ---\n\n"
            f"You just completed game {ep_idx + 1}. "
            f"Starting next game. {hint}\n\n"
            f"--- Game {ep_idx + 2} of {num_episodes} ---"
        )


register_prompt("secretary", SecretaryNoInfoPrompt)
register_prompt("secretary", SecretarySomeInfoPrompt)
register_prompt("secretary", SecretaryFullInfoPrompt)
