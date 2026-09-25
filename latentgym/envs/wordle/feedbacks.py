"""
Feedback handlers for the Wordle environment.

format_step_feedback: within-episode (pass through TextArena feedback)
format_episode_end_feedback: last episode only (matches old format)
"""
from __future__ import annotations

from typing import Any, Dict

from latentgym.core.feedback import FeedbackHandler
from latentgym.core.registry import register_feedback


class WordleStandardFeedback(FeedbackHandler):
    """Standard feedback: reward and win/loss, no answer revealed."""
    id = "standard"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict[str, Any]) -> str:
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx: int, episode_reward: float,
                                     episode_info: Dict[str, Any]) -> str:
        guess_history = episode_info.get("guess_history", [])
        won = guess_history and all(f == "G" for f in guess_history[-1][1])
        if won:
            return f"Episode {episode_idx + 1} finished. You guessed the word! Score: {episode_reward:.3f}"
        else:
            return f"Episode {episode_idx + 1} finished. You didn't guess the word. Score: {episode_reward:.3f}"


class WordleInformationFeedback(FeedbackHandler):
    """Information feedback: reward, last guess with letter feedback, and target word revealed."""
    id = "information"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict[str, Any]) -> str:
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx: int, episode_reward: float,
                                     episode_info: Dict[str, Any]) -> str:
        target = episode_info.get("target_word", "?")
        guess_history = episode_info.get("guess_history", [])
        if guess_history:
            last_guess, last_feedback = guess_history[-1]
            return (
                f"Episode {episode_idx + 1} finished. Score: {episode_reward:.3f}. "
                f"Your last guess [{last_guess}]: {''.join(last_feedback)}. "
                f"The target word was '{target}'."
            )
        return (
            f"Episode {episode_idx + 1} finished. Score: {episode_reward:.3f}. "
            f"No valid guesses made. The target word was '{target}'."
        )


register_feedback("wordle", WordleStandardFeedback)
register_feedback("wordle", WordleInformationFeedback)
