"""Feedback handlers for word ladder.

standard: passthrough, no solution revealed
information: reveals the optimal path after each episode
"""

from typing import Any, Dict
from latentgym.core.feedback import FeedbackHandler
from latentgym.core.registry import register_feedback


class WordLadderStandardFeedback(FeedbackHandler):
    """Standard passthrough feedback. No solution revealed."""
    id = "standard"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict[str, Any]) -> str:
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx: int,
                                     episode_reward: float,
                                     episode_info: Dict[str, Any]) -> str:
        if episode_reward >= 1.0:
            return f"Episode {episode_idx + 1} finished. You reached the target word! Score: {episode_reward:.3f}"
        else:
            return f"Episode {episode_idx + 1} finished. Score: {episode_reward:.3f}"


class WordLadderInformationFeedback(FeedbackHandler):
    """Reveals the optimal path after each episode.

    Gives the agent maximum learning signal — it can study optimal paths
    to discover the hidden pattern (hub word, word family, etc.).
    """
    id = "information"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict[str, Any]) -> str:
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx: int,
                                     episode_reward: float,
                                     episode_info: Dict[str, Any]) -> str:
        optimal_path = episode_info.get("optimal_path", [])
        optimal_steps = episode_info.get("optimal_steps", "?")
        path_str = " → ".join(optimal_path) if optimal_path else "unknown"

        if episode_reward >= 1.0:
            return (
                f"Episode {episode_idx + 1} finished. You reached the target word! "
                f"Score: {episode_reward:.3f}. "
                f"Optimal path ({optimal_steps} steps): {path_str}"
            )
        else:
            return (
                f"Episode {episode_idx + 1} finished. Score: {episode_reward:.3f}. "
                f"The optimal path was ({optimal_steps} steps): {path_str}"
            )


register_feedback("wordladder", WordLadderStandardFeedback)
register_feedback("wordladder", WordLadderInformationFeedback)
