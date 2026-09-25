"""Prompt templates for mastermind."""

from typing import Any, Dict
from latentgym.core.prompt import PromptTemplate
from latentgym.core.registry import register_prompt


class MastermindNoInfoPrompt(PromptTemplate):
    id = "no_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        return f"{game_rules}\n\nYou will play {num_episodes} rounds of Mastermind sequentially."

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        return (
            f"\n\n--- Game {ep_idx + 1} complete! ---\n\n"
            f"You just completed game {ep_idx + 1}. Starting next Mastermind game.\n\n"
            f"--- Game {ep_idx + 2} of {num_episodes} ---"
        )


class MastermindSomeInfoPrompt(PromptTemplate):
    id = "some_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} rounds of Mastermind sequentially. "
            f"The secret codes may share a common pattern. Pay attention to the "
            f"codes you discover across games."
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        return (
            f"\n\n--- Game {ep_idx + 1} complete! ---\n\n"
            f"You just completed game {ep_idx + 1}. Starting next Mastermind game. "
            f"Think about what pattern the codes share.\n\n"
            f"--- Game {ep_idx + 2} of {num_episodes} ---"
        )


class MastermindFullInfoPrompt(PromptTemplate):
    id = "full_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} rounds of Mastermind sequentially. "
            f"All secret codes share a hidden structural pattern (e.g., ascending order, "
            f"palindrome, specific digit constraint). Discover this pattern to crack "
            f"codes faster in later rounds."
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        return (
            f"\n\n--- Game {ep_idx + 1} complete! ---\n\n"
            f"You just completed game {ep_idx + 1}. Starting next Mastermind game. "
            f"Use the hidden pattern you've discovered to narrow your search space.\n\n"
            f"--- Game {ep_idx + 2} of {num_episodes} ---"
        )


register_prompt("mastermind", MastermindNoInfoPrompt)
register_prompt("mastermind", MastermindSomeInfoPrompt)
register_prompt("mastermind", MastermindFullInfoPrompt)
