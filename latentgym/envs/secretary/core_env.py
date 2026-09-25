"""Secretary problem single-episode environment wrapping TextArena's SecretaryEnv."""

import re
from typing import Any, Dict, Tuple

from latentgym.core.single_episode_env import SingleEpisodeEnv

try:
    from textarena.envs.Secretary.env import SecretaryEnv
except ImportError:
    SecretaryEnv = None


class SecretarySingleEpisodeEnv(SingleEpisodeEnv):
    """Optimal stopping: see values sequentially, accept or continue. Win if you pick the max."""

    def __init__(self):
        self._env = None
        self._num_draws = 10
        self._env_params_key: tuple = ()

    def reset(self, episode_config: Dict[str, Any]) -> str:
        draws = episode_config["draws"]
        self._num_draws = episode_config.get("max_turns_per_episode",
                          episode_config.get("num_draws", len(draws)))

        params_key = (self._num_draws,)
        if self._env is None or params_key != self._env_params_key:
            self._env = SecretaryEnv(N=self._num_draws)
            self._env_params_key = params_key

        self._env.reset(num_players=1)

        # Override random draws with trajectory-specified draws
        self._env.state.game_state["draws"] = list(draws)
        # Reset index to 0, clear observations, show first value
        self._env.state.game_state["current_idx"] = 0
        self._env.state.game_state["accepted_idx"] = None
        self._accepted_idx = None
        # Clear stale observations from the random draws
        self._env.state.observations = {0: []}
        self._env._show_next_value()

        return self._extract_initial_obs()

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        # Track current index before step (TextArena doesn't set accepted_idx)
        current_idx = self._env.state.game_state.get("current_idx", 0)

        # Normalize action:
        # 1. Extract [accept] or [continue] from anywhere in the sentence
        # 2. If neither found (invalid), treat as [continue] so the game progresses
        m = re.search(r'\[(accept|continue)\]', action, re.IGNORECASE)
        was_invalid = m is None
        normalized_action = f'[{m.group(1).lower()}]' if m else '[continue]'

        episode_done, info = self._env.step(normalized_action)
        feedback = self._extract_feedback()

        # Visible feedback: tell the player their invalid action was treated as [continue]
        if was_invalid:
            feedback = (
                "Your response was invalid format (expected [accept] or [continue]) "
                "— treated as [continue], skipped to next value.\n" + feedback
            )

        # Determine accepted index ourselves
        if episode_done:
            if "accept" in normalized_action:
                # Player accepted at current_idx (0-based, before increment)
                self._accepted_idx = max(0, current_idx - 1)
            else:
                # Auto-accepted last value (continued past all)
                draws = self._env.state.game_state.get("draws", [])
                self._accepted_idx = len(draws) - 1 if draws else 0

        reward = self._get_reward() if episode_done else 0.0
        draws = self._env.state.game_state.get("draws", [])
        max_val = max(draws) if draws else 0
        max_pos = draws.index(max_val) if draws else -1
        accepted_idx = self._accepted_idx if episode_done else None
        accepted_val = draws[accepted_idx] if accepted_idx is not None and draws else None
        return feedback, reward, episode_done, {
            "reward": reward,
            "draws": draws,
            "max_value": max_val,
            "max_position": max_pos,
            "accepted_value": accepted_val,
            "accepted_position": accepted_idx,
            "num_draws": len(draws),
        }

    def _extract_initial_obs(self) -> str:
        obs = self._env.state.observations
        if obs and 0 in obs:
            for t in reversed(obs[0]):
                if len(t) >= 2 and t[0] != 0:
                    return t[1]
        draws = self._env.state.game_state["draws"]
        return f"The current value is {draws[0]:.4f}. [accept] or [continue]?"

    def _extract_feedback(self) -> str:
        obs = self._env.state.observations
        if obs and 0 in obs:
            messages = [t[1] for t in obs[0] if len(t) >= 2 and t[0] != 0]
            if messages:
                return messages[-1]
        return "Continue."

    def _get_reward(self) -> float:
        """1.0 if picked the max, otherwise 0.5 * accepted/max.

        - Picked the max: 1.0 (clear success signal)
        - Close miss (0.90/0.95): 0.5 * 0.947 = 0.474
        - Bad pick (0.30/0.95): 0.5 * 0.316 = 0.158
        - Random strategy (~0.5 avg value, max ~0.95): ~0.26 avg reward
        - Perfect strategy: 1.0

        The 0.5 scaling ensures a gap between finding the max (1.0) and
        near-misses (~0.47), making the binary success clearly distinguishable
        while still giving partial credit gradient.
        """
        draws = self._env.state.game_state.get("draws", [])
        accepted_idx = getattr(self, '_accepted_idx', None)

        if not draws:
            return 0.0

        max_val = max(draws)
        if max_val <= 0:
            return 0.0

        # Get the accepted value
        if accepted_idx is not None and 0 <= accepted_idx < len(draws):
            accepted_val = draws[accepted_idx]
        else:
            return 0.0

        if accepted_val >= max_val:
            return 1.0
        return 0.5 * (accepted_val / max_val)

    def get_game_rules(self) -> str:
        return (
            f"You are playing the Secretary Problem (optimal stopping). "
            f"You will see {self._num_draws} values one at a time. "
            f"For each value, decide: [accept] to claim it, or [continue] to skip and see the next. "
            f"Once you accept, the game ends. If you reach the last value, you must accept it. "
            f"You win (reward 1.0) if the value you accept is the maximum of all values. "
            f"Otherwise you get partial credit based on how close your accepted value is to the maximum.\n\n"
            f"IMPORTANT: Output your action as [accept] or [continue]. "
            f"Square brackets must be used ONLY for the final action — never elsewhere in your response."
        )

    def close(self) -> None:
        pass
