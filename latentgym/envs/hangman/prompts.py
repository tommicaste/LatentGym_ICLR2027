"""Prompt templates for hangman."""

from typing import Any, Dict
from latentgym.core.prompt import PromptTemplate
from latentgym.core.registry import register_prompt


class HangmanNoInfoPrompt(PromptTemplate):
    id = "no_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} rounds of Hangman sequentially."
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        return (
            f"\n\n--- Episode {ep_idx + 1} complete! Reward: {prev_episode_reward:.3f} ---\n\n"
            f"--- Episode {ep_idx + 2} of {num_episodes} ---\n"
            f"A new word has been chosen."
        )


class HangmanSomeInfoPrompt(PromptTemplate):
    id = "some_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} rounds of Hangman sequentially. "
            f"The words may share common properties. Pay attention to patterns "
            f"across games to improve your guessing strategy."
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        return (
            f"\n\n--- Episode {ep_idx + 1} complete! Reward: {prev_episode_reward:.3f} ---\n\n"
            f"--- Episode {ep_idx + 2} of {num_episodes} ---\n"
            f"A new word has been chosen. Consider what the previous words had in common."
        )


class HangmanFullInfoPrompt(PromptTemplate):
    id = "full_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} rounds of Hangman sequentially. "
            f"All target words share a hidden property (e.g., same length, same "
            f"starting letter, same vowel pattern). Discover this property from "
            f"early rounds to guess more efficiently in later rounds."
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        return (
            f"\n\n--- Episode {ep_idx + 1} complete! Reward: {prev_episode_reward:.3f} ---\n\n"
            f"--- Episode {ep_idx + 2} of {num_episodes} ---\n"
            f"A new word has been chosen. Use what you've learned about the hidden "
            f"word property to narrow down your guesses."
        )


register_prompt("hangman", HangmanNoInfoPrompt)
register_prompt("hangman", HangmanSomeInfoPrompt)
register_prompt("hangman", HangmanFullInfoPrompt)
