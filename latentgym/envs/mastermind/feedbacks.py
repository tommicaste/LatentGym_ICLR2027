"""Feedback handlers for mastermind."""

from typing import Any, Dict
from latentgym.core.feedback import FeedbackHandler
from latentgym.core.registry import register_feedback


class MastermindStandardFeedback(FeedbackHandler):
    id = "standard"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict[str, Any]) -> str:
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx: int,
                                     episode_reward: float,
                                     episode_info: Dict[str, Any]) -> str:
        won = episode_info.get("won", False)
        if won:
            return f"Episode {episode_idx + 1} finished. You cracked the code! Score: {episode_reward:.3f}"
        else:
            return f"Episode {episode_idx + 1} finished. You didn't crack the code. Score: {episode_reward:.3f}"


class MastermindInformationFeedback(FeedbackHandler):
    id = "information"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict[str, Any]) -> str:
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx: int,
                                     episode_reward: float,
                                     episode_info: Dict[str, Any]) -> str:
        won = episode_info.get("won", False)
        secret = episode_info.get("secret_code", "?")
        last_guess = episode_info.get("last_guess", {})
        guess_str = " ".join(str(d) for d in last_guess.get("guess", []))
        black = last_guess.get("black", 0)
        white = last_guess.get("white", 0)

        if won:
            return (
                f"Episode {episode_idx + 1} finished. You cracked the code! Score: {episode_reward:.3f}. "
                f"Your last guess [{guess_str}]: {black} black, {white} white. "
                f"The secret code was {secret}."
            )
        else:
            return (
                f"Episode {episode_idx + 1} finished. You didn't crack the code. Score: {episode_reward:.3f}. "
                f"Your last guess [{guess_str}]: {black} black, {white} white. "
                f"The secret code was {secret}."
            )


register_feedback("mastermind", MastermindStandardFeedback)
register_feedback("mastermind", MastermindInformationFeedback)
