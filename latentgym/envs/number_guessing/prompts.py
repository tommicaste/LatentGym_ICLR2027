"""Prompt templates for number guessing: no_info, some_info, full_info.

All prompts include:
- Range info (min_range, max_range) from env_params
- Action format instruction: [number]
- CRITICAL instruction to output only the number
- num_episodes framing

Differences:
- no_info: No hint about cross-episode patterns
- some_info: Vague hint that numbers might follow a pattern
- full_info: Setting-dependent hint (set size, range type, etc.)
"""

from typing import Any, Dict

from latentgym.core.prompt import PromptTemplate
from latentgym.core.registry import register_prompt


def _format_block(min_range: int, max_range: int) -> str:
    """Common format instruction block shared by all prompts."""
    return (
        f"For each guess, I will tell you if the number is greater than, less than, "
        f"or equal to your guess. Your goal is to guess the number correctly in as few "
        f"turns as possible.\n\n"
        f"Your guess must be wrapped in square "
        f"brackets. Format your guess as: [number] (e.g., [{(min_range + max_range) // 2}])"
    )


def _get_full_info_hint(latent_id: str) -> str:
    """Return a setting-dependent hint for the full_info initial prompt."""
    if latent_id.startswith("set_of_"):
        n = latent_id.split("_")[-1]
        return (
            f"IMPORTANT: All the numbers in this series of games are drawn from a set of "
            f"{n} specific numbers. Use the information from previous games to identify "
            f"this set and improve your strategy."
        )
    elif latent_id == "range_100":
        return (
            "IMPORTANT: All the numbers in this series of games fall within a contiguous "
            "range of 100 numbers. Use information from previous games to narrow down "
            "this range and improve your strategy."
        )
    elif latent_id == "range_1000":
        return (
            "IMPORTANT: All the numbers in this series of games fall within a contiguous "
            "range of 1000 numbers. Use information from previous games to narrow down "
            "this range and improve your strategy."
        )
    elif latent_id == "dynamic_range":
        return (
            "IMPORTANT: All the numbers in this series of games fall within a contiguous "
            "sub-range of 1000 numbers somewhere within the full range you see. Use "
            "previous games to narrow down where this sub-range lies."
        )
    elif latent_id == "dynamic_full_range":
        return (
            "IMPORTANT: All the numbers can appear anywhere within the range shown. "
            "The range itself is the constraint — it stays the same across all games. "
            "Use previous games to learn the boundaries."
        )
    elif latent_id == "two_ranges":
        return (
            "IMPORTANT: The numbers in this series of games come from two separate "
            "non-overlapping ranges, each of size 500. Use the information from previous games to identify "
            "these ranges and improve your strategy."
        )
    else:
        return (
            "IMPORTANT: The target numbers across all games share a hidden pattern — they "
            "are constrained to a specific set or range. Discover this pattern from early "
            "games to guess faster in later games."
        )


def _get_full_info_transition_hint(latent_id: str) -> str:
    """Return a setting-dependent hint for the full_info transition message."""
    if latent_id.startswith("set_of_"):
        n = latent_id.split("_")[-1]
        return (
            f"Recall that the numbers are from a set of {n} numbers and you should "
            f"use the past history to improve your strategy."
        )
    elif latent_id in ("range_100"):
        return (
            "Recall that the numbers fall within a contiguous range of 100 numbers. Use the past "
            "history to narrow it down."
        )
    elif latent_id in ("range_1000"):
        return (
            "Recall that the numbers fall within a contiguous range of 1000 numbers. Use the past "
            "history to narrow it down."
        )
    elif latent_id == "dynamic_range":
        return (
            "Recall that the numbers fall within a 1000-number sub-range of the full range. "
            "Use the past history to narrow it down."
        )
    elif latent_id == "dynamic_full_range":
        return (
            "Recall that the numbers can appear anywhere within the shown range. "
            "Use the past history to learn the boundaries."
        )
    elif latent_id == "two_ranges":
        return (
            "Recall that the numbers come from two non-overlapping ranges, each of size 500. Use the past "
            "history to identify them."
        )
    else:
        return (
            "Recall that the numbers share a hidden pattern. Use the past history "
            "to improve your strategy."
        )


class NumberGuessingNoInfoPrompt(PromptTemplate):
    id = "no_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        self._env_params = env_params
        min_r = env_params.get("min_range", 1)
        max_r = env_params.get("max_range", 1000)
        return (
            f"You are playing {num_episodes} number guessing games sequentially. "
            f"In each game, you need to guess an integer (whole number, not a decimal) "
            f"between {min_r} and {max_r} (inclusive).\n\n"
            f"{_format_block(min_r, max_r)}"
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        return (
            f"\n\n--- Game {ep_idx + 1} complete! ---\n\n"
            f"You just completed game {ep_idx + 1}. "
            f"Starting next number guessing game, now ask first question of your next game.\n\n"
            f"--- Game {ep_idx + 2} of {num_episodes} ---"
        )


class NumberGuessingSomeInfoPrompt(PromptTemplate):
    id = "some_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        self._env_params = env_params
        min_r = env_params.get("min_range", 1)
        max_r = env_params.get("max_range", 1000)
        return (
            f"You are playing a series of {num_episodes} number guessing games. "
            f"In each game, you need to guess an integer (whole number, not a decimal) "
            f"between {min_r} and {max_r} (inclusive).\n\n"
            f"The numbers in these games might follow a pattern. Pay attention to the "
            f"numbers you encounter across different games, as this might help you "
            f"improve your strategy.\n\n"
            f"{_format_block(min_r, max_r)}"
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        return (
            f"\n\n--- Game {ep_idx + 1} complete! ---\n\n"
            f"You just completed game {ep_idx + 1}. "
            f"Starting next number guessing game, now ask first question of your next game. "
            f"You might want to use past history to improve your strategy.\n\n"
            f"--- Game {ep_idx + 2} of {num_episodes} ---"
        )


class NumberGuessingFullInfoPrompt(PromptTemplate):
    id = "full_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        self._env_params = env_params
        min_r = env_params.get("min_range", 1)
        max_r = env_params.get("max_range", 1000)
        latent_id = env_params.get("latent_id", "")
        hint = _get_full_info_hint(latent_id)
        return (
            f"You are playing a series of {num_episodes} number guessing games. "
            f"In each game, you need to guess an integer (whole number, not a decimal) "
            f"between {min_r} and {max_r} (inclusive).\n\n"
            f"{hint}\n\n"
            f"{_format_block(min_r, max_r)}"
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        latent_id = getattr(self, "_env_params", {}).get("latent_id", "")
        hint = _get_full_info_transition_hint(latent_id)
        return (
            f"\n\n--- Game {ep_idx + 1} complete! ---\n\n"
            f"You just completed game {ep_idx + 1}. "
            f"Starting next number guessing game, now ask first question of your next game. "
            f"{hint}\n\n"
            f"--- Game {ep_idx + 2} of {num_episodes} ---"
        )


register_prompt("number_guessing", NumberGuessingNoInfoPrompt)
register_prompt("number_guessing", NumberGuessingSomeInfoPrompt)
register_prompt("number_guessing", NumberGuessingFullInfoPrompt)
