"""
Prompt templates for the Wordle environment.

Transition messages match the old textarena_with_latents format,
including the "secret word may have changed" note.
"""
from __future__ import annotations

from typing import Any, Dict

from latentgym.core.prompt import PromptTemplate
from latentgym.core.registry import register_prompt, get_latent


class WordleNoInfoPrompt(PromptTemplate):
    id = "no_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any],
                               num_episodes: int) -> str:
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} rounds of this game sequentially."
        )

    def episode_transition_message(self, episode_idx: int, num_episodes: int,
                                    prev_episode_reward: float,
                                    prev_episode_info: Dict[str, Any]) -> str:
        return (
            f"\n\n--- Game {episode_idx + 1} complete! ---\n\n"
            f"You just completed game {episode_idx + 1}. Starting next Wordle game.\n\n"
            f"--- Game {episode_idx + 2} of {num_episodes} ---"
        )


class WordleSomeInfoPrompt(PromptTemplate):
    id = "some_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any],
                               num_episodes: int) -> str:
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} rounds sequentially. "
            f"The secret words across rounds may share a hidden pattern. "
            f"Pay attention to common features of the target words."
        )

    def episode_transition_message(self, episode_idx: int, num_episodes: int,
                                    prev_episode_reward: float,
                                    prev_episode_info: Dict[str, Any]) -> str:
        return (
            f"\n\n--- Game {episode_idx + 1} complete! ---\n\n"
            f"You just completed game {episode_idx + 1}. Starting next Wordle game. "
            f"Think about patterns you've noticed in the target words so far.\n\n"
            f"--- Game {episode_idx + 2} of {num_episodes} ---"
        )


class WordleFullInfoPrompt(PromptTemplate):
    id = "full_info"

    def initial_system_prompt(self, game_rules: str, env_params: Dict[str, Any],
                               num_episodes: int) -> str:
        self._env_params = env_params
        latent_id = env_params.get("latent_id", "")
        if latent_id:
            hint = get_latent("wordle", latent_id).description
        else:
            hint = "The secret words share a hidden constraint"
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} rounds sequentially. "
            f"IMPORTANT: {hint}. "
            f"Use this knowledge to guess more efficiently in later rounds."
        )

    def episode_transition_message(self, episode_idx: int, num_episodes: int,
                                    prev_episode_reward: float,
                                    prev_episode_info: Dict[str, Any]) -> str:
        latent_id = getattr(self, "_env_params", {}).get("latent_id", "")
        if latent_id:
            reminder = f"Recall: {get_latent('wordle', latent_id).description}."
        else:
            reminder = "Use what you've learned about the hidden constraint."
        return (
            f"\n\n--- Game {episode_idx + 1} complete! ---\n\n"
            f"You just completed game {episode_idx + 1}. Starting next Wordle game. "
            f"{reminder}\n\n"
            f"--- Game {episode_idx + 2} of {num_episodes} ---"
        )


register_prompt("wordle", WordleNoInfoPrompt)
register_prompt("wordle", WordleSomeInfoPrompt)
register_prompt("wordle", WordleFullInfoPrompt)
