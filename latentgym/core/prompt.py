"""
PromptTemplate — Abstract base class for prompt construction.

Handles two concerns:
1. Initial system prompt: combines game rules with meta-info and episode structure
2. Episode transitions: messages shown between episodes

The PromptTemplate knows nothing about game dynamics — it only provides
structure, hints, and framing. Game rules come from SingleEpisodeEnv.get_game_rules().
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class PromptTemplate(ABC):
    """Abstract base class for prompt templates.

    Each environment defines multiple PromptTemplate variants that differ in
    how much information they reveal about the latent structure:
        - no_info: No hints about patterns
        - some_info: Hints that patterns may exist
        - full_info: Explicit description of the latent constraint

    Attributes:
        id: Unique identifier for this prompt variant (e.g., "no_info", "full_info")
    """
    id: str

    @abstractmethod
    def initial_system_prompt(
        self,
        game_rules: str,
        env_params: Dict[str, Any],
        num_episodes: int,
    ) -> str:
        """Construct the full initial system prompt.

        This is called once at the start of a trajectory. It receives the
        game rules from core_env.get_game_rules() and combines them with
        meta-information about the multi-episode structure.

        Args:
            game_rules: Game rules text from SingleEpisodeEnv.get_game_rules()
            env_params: Environment-level parameters (e.g., button names, word length)
            num_episodes: Total number of episodes in this trajectory

        Returns:
            Complete system prompt string.

        Example (no_info variant):
            f"{game_rules}\\n\\nYou will play {num_episodes} episodes of this game."

        Example (full_info variant):
            f"{game_rules}\\n\\nYou will play {num_episodes} episodes. "
            f"There is a hidden pattern across episodes that you should discover."
        """
        ...

    @abstractmethod
    def episode_transition_message(
        self,
        episode_idx: int,
        num_episodes: int,
        prev_episode_reward: float,
        prev_episode_info: Dict[str, Any],
    ) -> str:
        """Generate the message shown between episodes.

        Called after one episode ends and before the next begins.
        Typically includes episode number, previous reward, and encouragement.

        Args:
            episode_idx: Index of the episode that just completed (0-based)
            num_episodes: Total number of episodes
            prev_episode_reward: Reward from the completed episode
            prev_episode_info: Info dict from the completed episode's final step

        Returns:
            Transition message string.

        Example:
            f"--- Episode {episode_idx + 1} complete! Reward: {prev_episode_reward:.3f} ---\\n"
            f"--- Episode {episode_idx + 2} of {num_episodes} ---"
        """
        ...
