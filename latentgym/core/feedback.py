"""
FeedbackHandler — Abstract base class for feedback formatting.

Handles two distinct feedback points:
1. Intra-episode: format raw step feedback within an episode (after each turn)
2. End-of-episode: summary feedback when an episode completes

The FeedbackHandler knows nothing about multi-episode structure or prompts.
It only processes raw environment observations into formatted text.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class FeedbackHandler(ABC):
    """Abstract base class for feedback formatting.

    Each environment can define multiple FeedbackHandler variants:
        - standard: Pass through raw feedback as-is
        - with_stats: Append summary statistics
        - minimal: Strip details, return only essential info
        - verbose: Add extra context and explanation

    Attributes:
        id: Unique identifier for this feedback variant (e.g., "standard", "with_stats")
    """
    id: str

    @abstractmethod
    def format_step_feedback(
        self,
        raw_feedback: str,
        episode_idx: int,
        turn: int,
        info: Dict[str, Any],
    ) -> str:
        """Format feedback within an episode (after each turn).

        The raw_feedback comes from SingleEpisodeEnv.step(). This handler
        can filter, augment, or reformat it.

        Args:
            raw_feedback: Raw feedback text from the environment
            episode_idx: Current episode index (0-based)
            turn: Current turn within the episode (1-based)
            info: Info dict from the environment step

        Returns:
            Formatted feedback string.

        Example (standard): return raw_feedback unchanged
        Example (with_stats): f"{raw_feedback}\\n[Stats: {info.get('stats', '')}]"
        Example (minimal): extract only the key result from raw_feedback
        """
        ...

    @abstractmethod
    def format_episode_end_feedback(
        self,
        episode_idx: int,
        episode_reward: float,
        episode_info: Dict[str, Any],
    ) -> str:
        """Format feedback at end of an episode.

        Called when an episode completes, before the episode transition message.
        Provides a summary of the episode outcome.

        Args:
            episode_idx: Index of the completed episode (0-based)
            episode_reward: Total reward earned in this episode
            episode_info: Accumulated info from the episode

        Returns:
            End-of-episode feedback string.

        Example:
            f"Episode {episode_idx + 1} finished. Your score: {episode_reward:.3f}"
        """
        ...
