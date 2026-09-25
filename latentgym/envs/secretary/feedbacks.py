"""Feedback handlers for secretary problem.

standard: passthrough, no solution revealed
information: reveals all draws, max value, and max position after each episode
"""

from typing import Any, Dict, List
from latentgym.core.feedback import FeedbackHandler
from latentgym.core.registry import register_feedback


class SecretaryStandardFeedback(FeedbackHandler):
    """Standard passthrough feedback. No solution revealed."""
    id = "standard"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict[str, Any]) -> str:
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx: int,
                                     episode_reward: float,
                                     episode_info: Dict[str, Any]) -> str:
        if episode_reward >= 1.0:
            return f"Episode {episode_idx + 1} finished. You picked the maximum value! Score: 1.000"
        else:
            return f"Episode {episode_idx + 1} finished. That wasn't the maximum. Score: 0.000"


class SecretaryInformationFeedback(FeedbackHandler):
    """Reveals draws, max value, and max position after each episode.

    Gives the agent maximum learning signal — it can study where the
    maximum was to discover the hidden pattern across episodes.
    """
    id = "information"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict[str, Any]) -> str:
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx: int,
                                     episode_reward: float,
                                     episode_info: Dict[str, Any]) -> str:
        draws = episode_info.get("draws", [])
        max_val = episode_info.get("max_value", "?")
        max_pos = episode_info.get("max_position", "?")
        num_draws = episode_info.get("num_draws", len(draws))

        # Format draws as a compact list
        draws_str = ", ".join(f"{d:.3f}" for d in draws) if draws else "?"

        if episode_reward >= 1.0:
            return (
                f"Episode {episode_idx + 1} finished. You picked the maximum value! Score: 1.000. "
                f"The maximum was {max_val:.3f} at position {max_pos} (of {num_draws}). "
                f"All values: [{draws_str}]"
            )
        else:
            return (
                f"Episode {episode_idx + 1} finished. That wasn't the maximum. Score: 0.000. "
                f"The maximum was {max_val:.3f} at position {max_pos} (of {num_draws}). "
                f"All values: [{draws_str}]"
            )


register_feedback("secretary", SecretaryStandardFeedback)
register_feedback("secretary", SecretaryInformationFeedback)
