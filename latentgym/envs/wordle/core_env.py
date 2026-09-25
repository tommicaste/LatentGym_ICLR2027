"""
Wordle SingleEpisodeEnv — thin wrapper around TextArena's WordleEnv.

Delegates all game logic to TextArena. Only adapts the interface
to SingleEpisodeEnv (reset/step/get_game_rules).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from textarena.envs.Wordle.env import WordleEnv

from latentgym.core.single_episode_env import SingleEpisodeEnv
from latentgym.core.registry import register_env

DEFAULT_WORD_LENGTH = 5
DEFAULT_MAX_ATTEMPTS = 10


class WordleSingleEpisodeEnv(SingleEpisodeEnv):
    """Thin wrapper around TextArena's WordleEnv.

    Episode config must contain:
        - target_word: str — the secret word for this episode
        - max_attempts: int (optional, defaults to 10)
        - word_length: int (optional, defaults to 5)
    """

    def __init__(self):
        self.word_length = 5
        self.max_attempts = 10
        self.hardcore = False
        self.turn_penalty = float(os.environ.get("WORDLE_TURN_PENALTY", "0.0"))
        self._env: WordleEnv | None = None
        self._env_params_key: tuple = ()

    def reset(self, episode_config: Dict[str, Any]) -> str:
        target_word = episode_config["target_word"]
        max_attempts = episode_config.get("max_turns_per_episode",
                       episode_config.get("max_attempts", self.max_attempts))
        word_length = episode_config.get("word_length", self.word_length)
        self.turn_penalty = episode_config.get("turn_penalty", self.turn_penalty)

        # Reuse existing env if constructor params haven't changed
        params_key = (word_length, max_attempts, self.hardcore, self.turn_penalty)
        if self._env is None or params_key != self._env_params_key:
            self._env = WordleEnv(
                word_length=word_length,
                num_guesses=max_attempts,
                hardcore=self.hardcore,
                turn_penalty=self.turn_penalty,
            )
            self._env_params_key = params_key
        else:
            self._env.num_guesses = max_attempts

        self._env.reset(num_players=1)
        self._env.state.game_state["secret_word"] = target_word.lower()

        return "Enter your guess to begin."

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        done, info = self._env.step(action)

        feedback = self._last_observation()
        reward = 0.0

        if done:
            # TextArena computes reward: 1.0 for win, percentage_completion for loss
            reward = self._get_reward()

        guess_history = self._env.state.game_state.get("guess_history", [])
        if done and guess_history and all(f == "G" for f in guess_history[-1][1]):
            word, _ = guess_history[-1]
            view = self._env.state.game_state.get("player_view", "")
            remaining = max(0, self.max_attempts - len(guess_history))
            feedback = f"You submitted [{word}].\nFeedback:\n{view}\nYou have {remaining} guesses left."

        target_word = self._env.state.game_state.get("secret_word", "?")
        return feedback, reward, done, {"turn": self._env.state.turn, "action": action, "guess_history": guess_history, "target_word": target_word}

    def get_game_rules(self) -> str:
        return (
            f"You are playing Wordle. A secret {self.word_length}-letter word has been chosen. "
            f"You have {self.max_attempts} attempts to guess it.\n\n"
            f"For each guess, wrap your word in square brackets (e.g., '[apple]').\n"
            f"Feedback for each letter:\n"
            f"  - G (green): correct letter in the correct position\n"
            f"  - Y (yellow): letter exists in the word but in the wrong position\n"
            f"  - X (wrong): letter is not in the word\n\n"
            f"IMPORTANT: Output your guess as [word]. "
            f"Square brackets must be used ONLY for the final guess — never elsewhere in your response."
        )

    def get_env_info(self) -> Dict[str, Any]:
        return {"word_length": self.word_length, "max_attempts": self.max_attempts}

    def _last_observation(self) -> str:
        """Get the most recent observation from TextArena."""
        obs = self._env.state.observations
        if obs and 0 in obs:
            for entry in reversed(obs[0]):
                if len(entry) >= 2 and entry[0] != 0:
                    return entry[1]
        return "Enter your guess to begin."

    def _get_reward(self) -> float:
        """Get reward from TextArena's state."""
        if hasattr(self._env.state, "outcome") and self._env.state.outcome:
            return self._env.state.outcome.get("reward", 0.0)
        rewards = getattr(self._env.state, "rewards", {})
        if rewards:
            return list(rewards.values())[0]
        return 0.0


register_env(
    name="wordle",
    env_class=WordleSingleEpisodeEnv,
    default_num_episodes=5,
    description="Wordle: guess a 5-letter word with G/Y/X feedback.",
    word_length=DEFAULT_WORD_LENGTH,
    max_turns_per_episode=DEFAULT_MAX_ATTEMPTS,
)
