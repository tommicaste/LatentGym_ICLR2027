"""Word Ladder single-episode environment wrapping TextArena's WordLadderEnv."""

import re
from typing import Any, Dict, Tuple

from latentgym.core.single_episode_env import SingleEpisodeEnv

try:
    import textarena as ta
    from textarena.envs.WordLadder.env import WordLadderEnv
except ImportError:
    ta = None
    WordLadderEnv = None


class WordLadderSingleEpisodeEnv(SingleEpisodeEnv):
    """Transform start_word into target_word by changing one letter at a time."""

    def __init__(self):
        self._env = None
        self._max_turns = 20
        self._env_params_key: tuple = ()

    def reset(self, episode_config: Dict[str, Any]) -> str:
        start_word = episode_config["start_word"].lower()
        target_word = episode_config["target_word"].lower()
        self._max_turns = episode_config.get("max_turns_per_episode",
                          episode_config.get("max_turns", 20))
        self._optimal_path = episode_config.get("optimal_path", [start_word, target_word])
        self._optimal_steps = episode_config.get("optimal_steps", len(self._optimal_path) - 1)
        self._turns = 0

        # Create WordLadderEnv instance if needed (loads word dictionaries once)
        params_key = (self._max_turns,)
        if self._env is None or params_key != self._env_params_key:
            self._env = WordLadderEnv(
                min_distance=1,
                max_distance=20,
                max_turns=self._max_turns,
            )
            self._env_params_key = params_key

        # --- Old approach: called self._env.reset() which runs expensive
        # --- BFS pair sampling (_sample_start_target), then we overrode the
        # --- words anyway. Replaced with direct state setup (Option A).
        #
        # self._env.reset(num_players=1)
        #
        # # Override generated words with trajectory-specified pair
        # self._env.start_word = start_word
        # self._env.target_word = target_word
        # self._env.current_word = start_word
        # self._env.history = [start_word]
        #
        # # Update game state
        # self._env.state.game_state["start_word"] = start_word
        # self._env.state.game_state["target_word"] = target_word
        #
        # # Clear stale observations and show initial state
        # self._env.state.observations = {0: []}

        # --- Option A: set up state directly, skipping _sample_start_target BFS
        self._env.start_word = start_word
        self._env.target_word = target_word
        self._env.current_word = start_word
        self._env.history = [start_word]

        self._env.state = ta.SinglePlayerState(
            num_players=1, max_turns=self._max_turns
        )
        game_state = {
            "start_word": start_word,
            "target_word": target_word,
            "rendered_text": self._env._render_text(),
        }
        self._env.state.reset(
            game_state=game_state,
            player_prompt_function=self._env._generate_player_prompt,
        )

        initial_msg = (
            f"Transform '{start_word}' into '{target_word}' "
            f"by changing one letter at a time. Each intermediate word must be "
            f"a valid English word. Reply with only your next word in brackets, "
            f"e.g. [cord]. Do not list the full chain. You have {self._max_turns} turns."
        )
        self._env.state.observations[0].append((-1, initial_msg, ta.ObservationType.GAME_MESSAGE))

        return initial_msg

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        self._turns += 1
        # If model lists multiple [word] entries, pick the first one
        # that's different from the current word (models tend to re-list current word)
        current = getattr(self._env, 'current_word', '').lower()
        words = re.findall(r'\[([a-zA-Z]+)\]', action)
        for w in words:
            if w.lower() != current:
                action = f'[{w}]'
                break
        episode_done, info = self._env.step(action)
        feedback = self._extract_feedback()
        reward = self._get_reward() if episode_done else 0.0
        return feedback, reward, episode_done, {
            "reward": reward,
            "start_word": self._env.start_word,
            "target_word": self._env.target_word,
            "optimal_path": self._optimal_path,
            "optimal_steps": self._optimal_steps,
        }

    def _extract_feedback(self) -> str:
        obs = self._env.state.observations
        if obs and 0 in obs:
            messages = [t[1] for t in obs[0] if len(t) >= 2 and t[0] != 0]
            if messages:
                return messages[-1]
        return "Continue."

    REWARD_DECAY = 0.03  # Per excess turn penalty

    def _get_reward(self) -> float:
        """Reward based on excess turns beyond optimal.

        Solved: 1.0 - (turns - optimal_steps) * DECAY, clamped to 0.0
        Failed: partial credit = matching_letters / word_length, no decay bonus

        Examples (DECAY=0.03, optimal=5):
          Solved in 5 turns (optimal): 1.0
          Solved in 8 turns:  1.0 - 3*0.03 = 0.91
          Solved in 15 turns: 1.0 - 10*0.03 = 0.70
          Failed, 3/4 match:  0.75
        """
        solved = (self._env.current_word == self._env.target_word)
        if solved:
            optimal = self._optimal_steps or self._turns
            excess = max(0, self._turns - optimal)
            return max(0.0, 1.0 - excess * self.REWARD_DECAY)
        else:
            matches = sum(a == b for a, b in zip(self._env.current_word, self._env.target_word))
            return matches / max(len(self._env.target_word), 1)

    def get_game_rules(self) -> str:
        return (
            "You are playing Word Ladder. You start with a word and must transform it "
            "into a target word by changing exactly one letter at a time. Each intermediate "
            "word must be a valid English word. Reply with only your next word in brackets, "
            "e.g. [cord]. Do not list the full chain. Win by reaching the target word within the turn limit.\n\n"
            "IMPORTANT: Only use square brackets [...] to submit your actual next word. "
            "Do not use square brackets anywhere else in your response."
        )

    def close(self) -> None:
        pass
