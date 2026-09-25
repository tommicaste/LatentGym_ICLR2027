"""Prompt templates for the bandit generator example."""
from __future__ import annotations
from typing import Any, Dict
from latentgym.core.prompt import PromptTemplate
from latentgym.core.registry import register_prompt

ENV_NAME = "bandit_generator_example"


class NoInfoPrompt(PromptTemplate):
    id = "no_info"
    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        return f"{game_rules}\n\nYou will play {num_episodes} rounds of this game sequentially."
    def episode_transition_message(self, episode_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict[str, Any]) -> str:
        return (f"--- Episode {episode_idx + 1} complete! Reward: {prev_episode_reward:.3f} ---\n\n"
                f"--- Episode {episode_idx + 2} of {num_episodes} ---")


class FullInfoPrompt(PromptTemplate):
    id = "full_info"
    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        return (f"{game_rules}\n\nYou will play {num_episodes} rounds sequentially. "
                f"The reward probabilities share a hidden pattern. Discover it to improve.")
    def episode_transition_message(self, episode_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict[str, Any]) -> str:
        return (f"--- Episode {episode_idx + 1} complete! Reward: {prev_episode_reward:.3f} ---\n\n"
                f"--- Episode {episode_idx + 2} of {num_episodes} ---")


register_prompt(ENV_NAME, NoInfoPrompt)
register_prompt(ENV_NAME, FullInfoPrompt)
