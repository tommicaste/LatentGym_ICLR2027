"""
SingleEpisodeEnv — Abstract base class for single-episode game dynamics.

This ABC defines the interface that all single-episode environments must implement.
It does NOT inherit from skyrl's Env — only MultiEpisodeEnv inherits BaseTextEnv.

The SingleEpisodeEnv handles one episode of a game. The MultiEpisodeEnv manages
the multi-episode loop, episode transitions, prompt composition, and reward aggregation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class SingleEpisodeEnv(ABC):
    """Abstract base class defining dynamics of a single episode.

    Implementers must define:
        - reset(): Initialize a new episode from config, return initial observation
        - step(): Process an action, return (feedback, reward, done, info)
        - get_game_rules(): Return game rules text for prompt construction

    The game rules are used by the layered prompt composition system —
    they provide the env-specific, constant-across-prompt-variants layer
    that gets combined with prompt templates.
    """

    @abstractmethod
    def reset(self, episode_config: Dict[str, Any]) -> str:
        """Reset environment for a new episode.

        Args:
            episode_config: Configuration for this episode, produced by the latent.
                For generator-based latents: output of generator_fn (e.g., button probabilities).
                For filter-based latents: selected item (e.g., target word).
                For static latents: predetermined config (e.g., number range).

        Returns:
            Initial observation text for this episode.
        """
        ...

    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute one action in the current episode.

        Args:
            action: The agent's action (text response).

        Returns:
            Tuple of:
                - feedback_text: Raw feedback from the environment
                - reward: Reward for this step (typically 0 during episode, nonzero at end)
                - episode_done: Whether this episode has ended
                - info: Additional metadata (e.g., parsed action, game state)
        """
        ...

    @abstractmethod
    def get_game_rules(self) -> str:
        """Return game rules text for prompt construction.

        This text describes how the game works — buttons, turns, scoring, etc.
        It is constant across prompt variants (no_info/some_info/full_info)
        and is combined with the PromptTemplate's meta_info and episode_info
        to create the full system prompt.

        Returns:
            Game rules as a string.
        """
        ...

    def close(self) -> None:
        """Clean up any resources held by the environment."""
        pass

    def validate_episode_config(self, config: Dict[str, Any]) -> None:
        """Validate that an episode config is well-formed for this env.

        Override to add env-specific validation.

        Args:
            config: Episode config to validate.

        Raises:
            ValueError: If config is invalid.
        """
        pass

    def get_env_info(self) -> Dict[str, Any]:
        """Return metadata about this environment.

        Override to provide env-specific info (e.g., action space, observation format).

        Returns:
            Dict of metadata.
        """
        return {}
