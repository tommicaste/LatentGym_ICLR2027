"""Hangman single-episode environment wrapping TextArena's HangmanEnv."""

from typing import Any, Dict, Tuple

from latentgym.core.single_episode_env import SingleEpisodeEnv

try:
    from textarena.envs.Hangman.env import HangmanEnv
except ImportError:
    HangmanEnv = None


class HangmanSingleEpisodeEnv(SingleEpisodeEnv):
    """Guess a word letter by letter. Wraps TextArena HangmanEnv."""

    def __init__(self):
        self._env = None
        self._max_attempts = 6
        self._env_params_key: tuple = ()

    def reset(self, episode_config: Dict[str, Any]) -> str:
        target_word = episode_config["target_word"]
        self._max_attempts = episode_config.get("max_turns_per_episode",
                             episode_config.get("max_attempts", 6))

        # Reuse env if constructor params haven't changed
        params_key = (self._max_attempts,)
        if self._env is None or params_key != self._env_params_key:
            self._env = HangmanEnv(
                word_list=[target_word],
                max_attempts=self._max_attempts,
            )
            self._env_params_key = params_key

        self._env.reset(num_players=1)

        # Override target word in game state
        self._env.state.game_state["target_word"] = target_word.lower()
        self._env.state.game_state["target_letters"] = list(target_word.upper())
        self._env.state.game_state["current_board"] = ["_"] * len(target_word)
        self._env.state.game_state["guessed_letters"] = set()
        self._env.state.game_state["tries_left"] = self._max_attempts

        # Re-observe after overriding game state so the initial board
        # reflects the actual target word (not TextArena's random word)
        self._env._observe_current_state()

        return self._extract_initial_obs()

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        episode_done, info = self._env.step(action)
        if not episode_done and self._env.state.turn >= self._max_attempts:
            episode_done = True
        feedback = self._extract_feedback()
        reward = self._get_reward() if episode_done else 0.0
        gs = self._env.state.game_state
        return feedback, reward, episode_done, {
            "reward": reward,
            "target_word": gs.get("target_word", "?"),
            "current_board": "".join(gs.get("current_board", [])),
            "guessed_letters": sorted(gs.get("guessed_letters", set())),
            "tries_left": gs.get("tries_left", 0),
        }

    def _extract_initial_obs(self) -> str:
        obs = self._env.state.observations
        if obs and 0 in obs:
            for t in reversed(obs[0]):
                if len(t) >= 2 and t[0] != 0:
                    return t[1]
        board = " ".join(self._env.state.game_state["current_board"])
        return f"Guess the word: {board}\nYou have {self._max_attempts} attempts."

    def _extract_feedback(self) -> str:
        obs = self._env.state.observations
        if obs and 0 in obs:
            messages = [t[1] for t in obs[0] if len(t) >= 2 and t[0] != 0]
            if messages:
                return messages[-1]
        return "Continue guessing."

    def _get_reward(self) -> float:
        # Check outcome
        if hasattr(self._env.state, 'outcome') and self._env.state.outcome:
            r = self._env.state.outcome.get('reward', 0.0)
            return float(r)
        rewards = getattr(self._env.state, 'rewards', {})
        if rewards:
            return float(list(rewards.values())[0])
        # Fallback: check if word is complete
        board = self._env.state.game_state.get("current_board", [])
        if "_" not in board:
            return 1.0
        return sum(1 for c in board if c != "_") / max(len(board), 1)

    def get_game_rules(self) -> str:
        return (
            "You are playing Hangman. A secret word has been chosen. "
            "Each turn, guess a letter using [L] format (e.g., [A]) or guess the "
            "full word using [WORD] format (e.g., [APPLE]). "
            "Correct letters are revealed on the board. Wrong guesses cost an attempt. "
            "Win by revealing all letters or guessing the word before running out of attempts.\n\n"
            "IMPORTANT: Only use square brackets [...] to submit your actual guess. "
            "Do not use square brackets anywhere else in your response."
        )

    def close(self) -> None:
        pass
