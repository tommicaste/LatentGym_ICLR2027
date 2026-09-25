"""Feedback handlers for number guessing."""

from typing import Any, Dict

from latentgym.core.feedback import FeedbackHandler
from latentgym.core.registry import register_feedback


class NumberGuessingStandardFeedback(FeedbackHandler):
    """Standard passthrough feedback."""
    id = "standard"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict[str, Any]) -> str:
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx: int,
                                     episode_reward: float,
                                     episode_info: Dict[str, Any]) -> str:
        solved = episode_info.get("solved", False)
        if solved:
            return f"Episode {episode_idx + 1} finished. Score: {episode_reward:.3f}"
        else:
            return f"Episode {episode_idx + 1} finished. You didn't find the number. Score: 0.000"


class NumberGuessingInformationFeedback(FeedbackHandler):
    """Feedback that reveals the target number at the end of each episode."""
    id = "information"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict[str, Any]) -> str:
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx: int,
                                     episode_reward: float,
                                     episode_info: Dict[str, Any]) -> str:
        solved = episode_info.get("solved", False)
        target = episode_info.get("target_number", "?")
        if solved:
            return (f"Episode {episode_idx + 1} finished. Score: {episode_reward:.3f}. "
                    f"The number was {target}.")
        else:
            return (f"Episode {episode_idx + 1} finished. You didn't find the number. Score: 0.000. "
                    f"The number was {target}.")


register_feedback("number_guessing", NumberGuessingStandardFeedback)
register_feedback("number_guessing", NumberGuessingInformationFeedback)
