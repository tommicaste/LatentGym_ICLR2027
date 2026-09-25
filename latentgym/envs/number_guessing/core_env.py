"""Number Guessing single-episode environment. Self-contained, no TextArena dependency."""

import re
from typing import Any, Dict, Tuple

from latentgym.core.single_episode_env import SingleEpisodeEnv


class NumberGuessingSingleEpisodeEnv(SingleEpisodeEnv):
    """Agent guesses a target number within a range. Feedback: greater/less/correct."""

    REWARD_DECAY = 0.020  # Per-turn reward decay

    def __init__(self):
        self._target: int = 0
        self._min_range: int = 1
        self._max_range: int = 1000
        self._turns: int = 0
        self._max_turns: int = 30
        self._solved: bool = False

    def reset(self, episode_config: Dict[str, Any]) -> str:
        self._target = episode_config["target_number"]
        self._min_range = episode_config.get("min_range", 1)
        self._max_range = episode_config.get("max_range", 1000)
        self._max_turns = episode_config.get("max_turns_per_episode",
                          episode_config.get("max_turns", 30))
        self._turns = 0
        self._solved = False
        return (
            f"I'm thinking of a number between {self._min_range} and {self._max_range}. "
            f"You have {self._max_turns} guesses. What's your guess?"
        )

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        self._turns += 1

        # Parse integer from action.
        # Try bracketed format first: [42], take the LAST match (final answer).
        # Fall back to last bare number if no brackets.
        guess = None
        bracketed = re.findall(r'\[(\d+)\]', action)
        if bracketed:
            guess = int(bracketed[-1])
        else:
            bare = re.findall(r'\d+', action)
            if bare:
                guess = int(bare[-1])

        if guess is None:
            feedback = (
                f"Invalid format. Please guess a number in brackets, e.g. [{self._min_range}]. "
                f"Your guess must be between {self._min_range} and {self._max_range}."
            )
            done = self._turns >= self._max_turns
            return feedback, 0.0, done, {"turn": self._turns, "target_number": self._target}

        # Validate range
        if guess < self._min_range or guess > self._max_range:
            feedback = f"Your guess must be between {self._min_range} and {self._max_range}. Try again."
            done = self._turns >= self._max_turns
            return feedback, 0.0 if done else 0.0, done, {"turn": self._turns, "target_number": self._target}

        # Check guess
        if guess == self._target:
            self._solved = True
            reward = max(0.0, 1.0 - self._turns * self.REWARD_DECAY)
            feedback = f"Correct! You guessed the number {self._target} in {self._turns} turns."
            return feedback, reward, True, {"turn": self._turns, "solved": True, "target_number": self._target}
        elif guess < self._target:
            feedback = f"The number is greater than {guess}."
        else:
            feedback = f"The number is less than {guess}."

        done = self._turns >= self._max_turns
        if done:
            feedback += f"\nYou've run out of guesses."

        return feedback, 0.0, done, {"turn": self._turns, "solved": False, "target_number": self._target}

    def get_game_rules(self) -> str:
        # Kept minimal — the prompt templates handle the full framing,
        # including format instructions, range info, and latent hints.
        return ""

    def close(self) -> None:
        pass
