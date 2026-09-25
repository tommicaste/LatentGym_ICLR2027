"""Prompt templates for word ladder: no_info, some_info, full_info.

All prompts include:
- Game rules (transform start → target, one letter at a time, valid words)
- Action format instruction: [word]
- Multi-episode framing

Differences:
- no_info: No hint about cross-episode patterns
- some_info: Vague hint that solutions may share properties
- full_info: Latent-specific hint (hub word, restricted vocab, word family, etc.)
"""

from typing import Any, Dict

from latentgym.core.prompt import PromptTemplate
from latentgym.core.registry import register_prompt


def _base_prompt(num_episodes: int) -> str:
    """Common base for all word ladder prompts."""
    return (
        f"You are playing a series of {num_episodes} Word Ladder games. "
        f"In each game, you are given a start word and a target word. "
        f"Transform the start word into the target word by changing exactly one "
        f"letter at a time. Each intermediate word must be a valid English word.\n\n"
        f"You may reason about your strategy, but your answer must be wrapped in "
        f"square brackets. Format your guess as: [word] (e.g., [cold])"
    )


def _get_full_info_hint(latent_id: str) -> str:
    """Return a latent-specific hint for the full_info prompt."""
    prefix = "IMPORTANT: There is a pattern across games. "

    # Hub word latents
    if latent_id.startswith("hub_word_"):
        return (
            prefix + "All the word pairs in this series can be solved by going "
            "through a specific intermediate 'hub' word. Discover this hub word and "
            "use it to split every puzzle into two easier sub-problems."
        )
    # Restricted vocabulary latents
    elif latent_id.startswith("restricted_vocab_"):
        return (
            prefix + "All the solutions in this series use words from a small fixed "
            "vocabulary of about 40 words. Learn which words are 'in play' across games "
            "to navigate faster."
        )
    # Positional order latents
    elif latent_id == "order_left_to_right":
        return (
            prefix + "The optimal solutions tend to change letter positions from left "
            "to right. Try changing the leftmost differing letter first."
        )
    elif latent_id == "order_right_to_left":
        return (
            prefix + "The optimal solutions tend to change letter positions from right "
            "to left. Try changing the rightmost differing letter first."
        )
    elif latent_id == "order_outside_in":
        return (
            prefix + "The optimal solutions tend to change outer letter positions first, "
            "then inner ones."
        )
    # Substitution pattern latents
    elif latent_id == "subs_vowel_swaps":
        return (
            prefix + "The optimal paths primarily involve changing vowels (a, e, i, o, u). "
            "Consonants tend to stay fixed across steps."
        )
    elif latent_id == "subs_consonant_swaps":
        return (
            prefix + "The optimal paths primarily involve changing consonants. "
            "Vowels tend to stay fixed across steps."
        )
    elif latent_id == "subs_alternating":
        return (
            prefix + "The optimal paths alternate between changing vowels and consonants "
            "at each step."
        )
    elif latent_id == "subs_phonetic_group":
        return (
            prefix + "Consonant changes in the optimal paths stay within phonetic groups "
            "(b/p/d/t, f/v/s/z, m/n/l/r). Use these groups to find valid intermediate words."
        )
    # Word family chain latents
    elif latent_id.startswith("family_contains_"):
        bigram = latent_id.replace("family_contains_", "")
        return (
            prefix + f"The intermediate words in the optimal paths all contain the "
            f"letter pattern '{bigram}'. Stay within this word family for shorter solutions."
        )
    elif latent_id.startswith("family_ends_"):
        letter = latent_id.replace("family_ends_", "")
        return (
            prefix + f"The intermediate words in the optimal paths all end with "
            f"'{letter}'. Stay within words ending in '{letter}' for shorter solutions."
        )
    elif latent_id.startswith("family_pattern_"):
        pattern = latent_id.replace("family_pattern_", "").upper()
        return (
            prefix + f"The intermediate words in the optimal paths all follow a "
            f"{pattern} pattern (C=consonant, V=vowel). Stay within this pattern."
        )
    # Fallback
    else:
        return (
            prefix + "The word pairs and their solutions share a hidden structural "
            "pattern. Discover this pattern from early games to solve faster in later games."
        )


def _get_full_info_transition_hint(latent_id: str) -> str:
    """Return a latent-specific hint for the full_info transition message."""
    if latent_id.startswith("hub_word_"):
        return "Recall that all solutions go through a common hub word."
    elif latent_id.startswith("restricted_vocab_"):
        return "Recall that solutions use a small fixed vocabulary."
    elif latent_id.startswith("order_"):
        return "Recall the positional order pattern in the solutions."
    elif latent_id.startswith("subs_"):
        return "Recall the letter substitution pattern in the solutions."
    elif latent_id.startswith("family_"):
        return "Recall the word family pattern in the intermediate words."
    else:
        return "Use what you've learned from previous games."


class WordLadderNoInfoPrompt(PromptTemplate):
    id = "no_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        self._env_params = env_params
        return _base_prompt(num_episodes)

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        return (
            f"\n\n--- Game {ep_idx + 1} complete! ---\n\n"
            f"You just completed game {ep_idx + 1}. "
            f"Starting next Word Ladder game.\n\n"
            f"--- Game {ep_idx + 2} of {num_episodes} ---"
        )


class WordLadderSomeInfoPrompt(PromptTemplate):
    id = "some_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        self._env_params = env_params
        return (
            f"{_base_prompt(num_episodes)}\n\n"
            f"The word pairs and their solutions across games may share common "
            f"properties or patterns. Pay attention to the words that appear in "
            f"your solutions — this might help you solve future games faster."
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        return (
            f"\n\n--- Game {ep_idx + 1} complete! ---\n\n"
            f"You just completed game {ep_idx + 1}. "
            f"Starting next Word Ladder game. Think about what your previous "
            f"solutions had in common.\n\n"
            f"--- Game {ep_idx + 2} of {num_episodes} ---"
        )


class WordLadderFullInfoPrompt(PromptTemplate):
    id = "full_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any], num_episodes: int) -> str:
        self._env_params = env_params
        latent_id = env_params.get("latent_id", "")
        hint = _get_full_info_hint(latent_id)
        return (
            f"{_base_prompt(num_episodes)}\n\n"
            f"{hint}"
        )

    def episode_transition_message(self, ep_idx: int, num_episodes: int,
                                    prev_episode_reward: float, prev_episode_info: Dict) -> str:
        latent_id = getattr(self, "_env_params", {}).get("latent_id", "")
        hint = _get_full_info_transition_hint(latent_id)
        return (
            f"\n\n--- Game {ep_idx + 1} complete! ---\n\n"
            f"You just completed game {ep_idx + 1}. "
            f"Starting next Word Ladder game. {hint}\n\n"
            f"--- Game {ep_idx + 2} of {num_episodes} ---"
        )


register_prompt("wordladder", WordLadderNoInfoPrompt)
register_prompt("wordladder", WordLadderSomeInfoPrompt)
register_prompt("wordladder", WordLadderFullInfoPrompt)
