"""Mastermind single-episode environment wrapping TextArena's MastermindEnv."""

import os
from typing import Any, Dict, List, Tuple

from latentgym.core.single_episode_env import SingleEpisodeEnv

try:
    from textarena.envs.Mastermind.env import MastermindEnv
except ImportError:
    MastermindEnv = None


class MastermindSingleEpisodeEnv(SingleEpisodeEnv):
    """Guess a secret code. Feedback: black pegs (right position) + white pegs (right digit, wrong position)."""

    def __init__(self):
        self._env = None
        self._code_length = 4
        self._num_numbers = 6
        self.turn_penalty = float(os.environ.get("MASTERMIND_TURN_PENALTY", "0.0"))
        self._env_params_key: tuple = ()

    def reset(self, episode_config: Dict[str, Any]) -> str:
        secret_code = episode_config["secret_code"]
        self._code_length = episode_config.get("code_length", 4)
        self._num_numbers = episode_config.get("num_numbers", 6)
        duplicates = episode_config.get("duplicates_allowed", False)
        self._max_turns = episode_config.get("max_turns_per_episode",
                    episode_config.get("max_turns", 10))
        max_turns = self._max_turns
        self.turn_penalty = episode_config.get("turn_penalty", self.turn_penalty)

        params_key = (self._code_length, self._num_numbers, max_turns, duplicates, self.turn_penalty)
        if self._env is None or params_key != self._env_params_key:
            self._env = MastermindEnv(
                code_length=self._code_length,
                num_numbers=self._num_numbers,
                max_turns=max_turns,
                duplicate_numbers=duplicates,
                turn_penalty=self.turn_penalty,
            )
            self._env_params_key = params_key

        self._env.reset(num_players=1)

        # Override secret code
        self._env.state.game_state["secret_code"] = list(secret_code)

        return self._extract_initial_obs()

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        episode_done, info = self._env.step(action)
        feedback = self._extract_feedback()
        if not episode_done and self._env.state.turn >= self._max_turns:
            episode_done = True
            reward = self._env._get_percentage_completion()
        else:
            reward = self._get_reward() if episode_done else 0.0

        history = self._env.state.game_state.get("history", [])
        secret_code = self._env.state.game_state.get("secret_code", [])
        last_guess = history[-1] if history else {}
        won = bool(last_guess and last_guess.get("black", 0) == self._code_length)

        if episode_done and won:
            g = " ".join(str(d) for d in last_guess["guess"])
            b, w = last_guess["black"], last_guess["white"]
            feedback = f"Submitted [{g}]. Feedback: {b} black peg(s), {w} white peg(s)."

        return feedback, reward, episode_done, {
            "won": won,
            "secret_code": secret_code,
            "last_guess": last_guess,
        }

    def _extract_initial_obs(self) -> str:
        return f"You have {self._max_turns} turns to guess the code. Enter your guess."

    def _extract_feedback(self) -> str:
        obs = self._env.state.observations
        if obs and 0 in obs:
            messages = [t[1] for t in obs[0] if len(t) >= 2 and t[0] != 0]
            if messages:
                return messages[-1]
        return "Continue guessing."

    def _get_reward(self) -> float:
        if hasattr(self._env.state, 'outcome') and self._env.state.outcome:
            return float(self._env.state.outcome.get('reward', 0.0))
        rewards = getattr(self._env.state, 'rewards', {})
        if rewards:
            return float(list(rewards.values())[0])
        return 0.0

    def get_game_rules(self) -> str:
        return (
            f"You are playing Mastermind. A secret code of {self._code_length} digits "
            f"has been chosen, each digit between 1 and {self._num_numbers}. "
            f"Each turn, guess the code using format [1 2 3 4]. "
            f"You'll receive feedback: black pegs (correct digit in correct position) "
            f"and white pegs (correct digit in wrong position). "
            f"Win by finding the exact code.\n\n"
            f"IMPORTANT: Only use square brackets [X X X X] for your actual guess submission. "
            f"Do not use square brackets when discussing or referencing previous guesses."
        )

    def close(self) -> None:
        pass
