"""Prompt templates for bandits: no_info, some_info, full_info.

All prompts include:
- Game rules (explore with [button], select with [select button], early stopping)
- Multi-episode framing

Differences:
- no_info: No hint about cross-episode patterns
- some_info: Vague hint that button probabilities may have a pattern
- full_info: Grouped hints by latent type (position-fixed, rotation, subset,
  cross-episode memory, mathematical)
"""
from __future__ import annotations

from typing import Any, Dict

from latentgym.core.prompt import PromptTemplate
from latentgym.core.registry import register_prompt


def _get_full_info_hint(latent_id: str) -> str:
    """Return a grouped hint based on latent type."""

    # Position-fixed latents — same button always best
    position_fixed = {
        "fixed_probabilities", "one_hot", "bottom_excluded",
        "even_indices_only", "odd_indices_only",
    }
    if latent_id in position_fixed or latent_id.startswith("loyal_favorite_"):
        return (
            "IMPORTANT: There is a consistent pattern of buttons "
            "across games. Pay attention to which button wins."
        )

    # Rotation/alternating latents — best shifts predictably
    if latent_id in ("clockwise_rotation", "counterclockwise_rotation",
                     "ping_pong", "swap_top_two", "cycle_length_5"):
        return (
            "IMPORTANT: The best color changes in a predictable pattern across "
            "games. Track which button was best in previous games to predict the next."
        )

    # Subset latents — best always from a small group
    if latent_id in ("binary_switch_0_1", "binary_switch_0_last", "top_two_fixed"):
        return (
            "IMPORTANT: The best button is always from a small subset of colors. "
            "Discover which buttons are the contenders to focus your exploration."
        )

    # Cross-episode memory latents — best depends on previous episode
    if latent_id in ("shadow", "cold_hand", "hot_hand", "random_walk", "mirror_mode"):
        return (
            "IMPORTANT: The best button in each game depends on which button was "
            "best in the previous game. Track the history across games."
        )

    # Skip/math latents — position follows math pattern
    if latent_id in ("skip_2", "skip_3", "fibonacci", "prime_indices", "triangular"):
        return (
            "IMPORTANT: The best color follows a mathematical sequence based on "
            "its ordering across games."
        )

    # Same ranking — ordering constant
    if latent_id == "same_ranking":
        return (
            "IMPORTANT: The ranking of buttons never changes across games. "
            "Once you identify the best button, it stays the best."
        )

    # Fallback
    return (
        "IMPORTANT: The reward probabilities across games share a hidden pattern. "
        "Discover this pattern to select the best button earlier in later games."
    )


def _get_full_info_transition_hint(latent_id: str) -> str:
    """Return a grouped transition hint."""
    if latent_id.startswith("loyal_favorite_") or latent_id in (
        "fixed_probabilities", "one_hot", "bottom_excluded",
        "even_indices_only", "odd_indices_only", "same_ranking",
    ):
        return "Recall which button was best in previous games."

    if latent_id in ("clockwise_rotation", "counterclockwise_rotation",
                     "ping_pong", "swap_top_two", "cycle_length_5"):
        return "Track how the best button has shifted across games."

    if latent_id in ("binary_switch_0_1", "binary_switch_0_last", "top_two_fixed"):
        return "Focus on the subset of buttons that have been best."

    if latent_id in ("shadow", "cold_hand", "hot_hand", "random_walk", "mirror_mode"):
        return "Use the previous game's result to predict the best button."

    if latent_id in ("skip_2", "skip_3", "fibonacci", "prime_indices", "triangular"):
        return "Apply the mathematical pattern to predict the next best button."

    return "Use what you've learned from previous games."


class BanditNoInfoPrompt(PromptTemplate):
    id = "no_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any],
                              num_episodes: int) -> str:
        self._env_params = env_params
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} rounds of this game sequentially."
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float,
                                    prev_episode_info: Dict) -> str:
        return (
            f"\n\n--- Game {ep_idx + 1} complete! ---\n\n"
            f"You just completed game {ep_idx + 1}. "
            f"Starting next game.\n\n"
            f"--- Game {ep_idx + 2} of {num_episodes} ---"
        )


class BanditSomeInfoPrompt(PromptTemplate):
    id = "some_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any],
                              num_episodes: int) -> str:
        self._env_params = env_params
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} rounds of this game sequentially. "
            f"The reward probabilities across these rounds may share a hidden "
            f"pattern or constraint. Pay attention to which buttons perform best "
            f"across games."
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float,
                                    prev_episode_info: Dict) -> str:
        return (
            f"\n\n--- Game {ep_idx + 1} complete! ---\n\n"
            f"You just completed game {ep_idx + 1}. "
            f"Starting next game. Think about what you've learned about the "
            f"buttons across games.\n\n"
            f"--- Game {ep_idx + 2} of {num_episodes} ---"
        )


class BanditFullInfoPrompt(PromptTemplate):
    id = "full_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any],
                              num_episodes: int) -> str:
        self._env_params = env_params
        latent_id = env_params.get("latent_id", "")
        hint = _get_full_info_hint(latent_id)
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} rounds of this game sequentially.\n\n"
            f"{hint}"
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float,
                                    prev_episode_info: Dict) -> str:
        latent_id = getattr(self, "_env_params", {}).get("latent_id", "")
        hint = _get_full_info_transition_hint(latent_id)
        return (
            f"\n\n--- Game {ep_idx + 1} complete! ---\n\n"
            f"You just completed game {ep_idx + 1}. "
            f"Starting next game. {hint}\n\n"
            f"--- Game {ep_idx + 2} of {num_episodes} ---"
        )


register_prompt("bandits", BanditNoInfoPrompt)
register_prompt("bandits", BanditSomeInfoPrompt)
register_prompt("bandits", BanditFullInfoPrompt)
