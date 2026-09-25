"""Feedback handlers for hangman."""

from typing import Any, Dict
from latentgym.core.feedback import FeedbackHandler
from latentgym.core.registry import register_feedback


class HangmanStandardFeedback(FeedbackHandler):
    id = "standard"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict[str, Any]) -> str:
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx: int,
                                     episode_reward: float,
                                     episode_info: Dict[str, Any]) -> str:
        if episode_reward >= 1.0:
            return f"Episode {episode_idx + 1} finished. You guessed the word! Score: {episode_reward:.3f}"
        else:
            return f"Episode {episode_idx + 1} finished. Score: {episode_reward:.3f}"


class HangmanInformationFeedback(FeedbackHandler):
    """Information feedback: reward, final board state, letters guessed, and target word revealed."""
    id = "information"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict[str, Any]) -> str:
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx: int,
                                     episode_reward: float,
                                     episode_info: Dict[str, Any]) -> str:
        target = episode_info.get("target_word", "?")
        board = episode_info.get("current_board", "")
        guessed = episode_info.get("guessed_letters", [])
        won = episode_reward >= 1.0
        outcome = "You guessed the word!" if won else "You didn't guess the word."
        guessed_str = ",".join(guessed) if guessed else "none"
        return (
            f"Episode {episode_idx + 1} finished. {outcome} Score: {episode_reward:.3f}. "
            f"Final board: [{board}]. Letters guessed: {guessed_str}. "
            f"The target word was '{target}'."
        )


register_feedback("hangman", HangmanStandardFeedback)
register_feedback("hangman", HangmanInformationFeedback)
