"""Feedback handlers for the bandit generator example."""
from __future__ import annotations
from typing import Any, Dict
from latentgym.core.feedback import FeedbackHandler
from latentgym.core.registry import register_feedback

ENV_NAME = "bandit_generator_example"


class StandardFeedback(FeedbackHandler):
    id = "standard"
    def format_step_feedback(self, raw_feedback: str, episode_idx: int,
                              turn: int, info: Dict[str, Any]) -> str:
        return raw_feedback
    def format_episode_end_feedback(self, episode_idx: int, episode_reward: float,
                                     episode_info: Dict[str, Any]) -> str:
        return f"--- Final episode complete! Reward: {episode_reward:.3f} ---"


register_feedback(ENV_NAME, StandardFeedback)
