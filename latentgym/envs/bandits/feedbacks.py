"""Feedback handlers for bandits.

standard: passthrough, no ground truth revealed
information: reveals ground truth probabilities and best button after each episode
"""
from __future__ import annotations

from typing import Any, Dict

from latentgym.core.feedback import FeedbackHandler
from latentgym.core.registry import register_feedback


class BanditStandardFeedback(FeedbackHandler):
    """Standard passthrough feedback. No ground truth revealed."""
    id = "standard"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict[str, Any]) -> str:
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx: int,
                                     episode_reward: float,
                                     episode_info: Dict[str, Any]) -> str:
        selected = episode_info.get("selected_button", "?")
        is_correct = episode_info.get("is_correct", None)
        if is_correct:
            return (
                f"Episode {episode_idx + 1} finished. You selected '{selected}' — "
                f"correct! Score: {episode_reward:.3f}"
            )
        elif is_correct is False:
            return (
                f"Episode {episode_idx + 1} finished. You selected '{selected}' — "
                f"wrong. Score: {episode_reward:.3f}"
            )
        else:
            return f"Episode {episode_idx + 1} finished. Score: {episode_reward:.3f}"


class BanditInformationFeedback(FeedbackHandler):
    """Reveals ground truth probabilities after each episode.

    Gives the agent maximum learning signal — it can study which button
    was actually best to discover the hidden pattern across episodes.
    """
    id = "information"

    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict[str, Any]) -> str:
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx: int,
                                     episode_reward: float,
                                     episode_info: Dict[str, Any]) -> str:
        selected = episode_info.get("selected_button", "?")
        best = episode_info.get("best_button", "?")
        is_correct = episode_info.get("is_correct", None)
        gt = episode_info.get("ground_truth", {})

        # Format probabilities
        probs_str = ", ".join(
            f"{btn}={prob:.2f}" for btn, prob in sorted(gt.items(), key=lambda x: -x[1])
        ) if gt else "?"

        if is_correct:
            return (
                f"Episode {episode_idx + 1} finished. You selected '{selected}' — "
                f"correct! Score: {episode_reward:.3f}. "
                f"Probabilities: {probs_str}"
            )
        elif is_correct is False:
            return (
                f"Episode {episode_idx + 1} finished. You selected '{selected}' — "
                f"wrong. The best was '{best}'. Score: {episode_reward:.3f}. "
                f"Probabilities: {probs_str}"
            )
        else:
            return (
                f"Episode {episode_idx + 1} finished. Score: {episode_reward:.3f}. "
                f"Probabilities: {probs_str}"
            )


register_feedback("bandits", BanditStandardFeedback)
register_feedback("bandits", BanditInformationFeedback)
