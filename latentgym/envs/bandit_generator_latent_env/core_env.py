"""
Bandit SingleEpisodeEnv — thin wrapper around TextArena's BanditEnv.

Same game logic as envs/bandits/core_env.py. Registered under a separate
name ("bandit_generator_example") to demonstrate the generator_fn path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from textarena.envs.Bandit.env import BanditEnv

from latentgym.core.single_episode_env import SingleEpisodeEnv
from latentgym.core.registry import register_env

DEFAULT_BUTTONS = ["red", "blue", "green", "yellow", "purple"]
DEFAULT_NUM_TURNS = 20


class BanditSingleEpisodeEnv(SingleEpisodeEnv):
    """Single episode of multi-armed bandits. Wraps TextArena's BanditEnv.

    Episode config must contain:
        ground_truth: Dict[str, float] — button name → probability
    """

    def __init__(self, buttons: List[str] = None, num_turns: int = DEFAULT_NUM_TURNS):
        self.buttons = buttons or DEFAULT_BUTTONS
        self.num_turns = num_turns
        self._env: BanditEnv | None = None

    def reset(self, episode_config: Dict[str, Any]) -> str:
        ground_truth = episode_config["ground_truth"]
        # TextArena off-by-one fix: pass num_turns - 1
        self._env = BanditEnv(buttons=self.buttons, num_turns=self.num_turns - 1)
        self._env.reset(num_players=1)
        self._env.state.game_state["ground_truth"] = ground_truth.copy()
        self._env.state.game_state["history"] = {b: [] for b in self.buttons}
        return self._last_observation()

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        done, info = self._env.step(action)
        feedback = self._last_observation()
        reward = 0.0
        if done:
            reward = self._get_reward()
        return feedback, reward, done, {"action": action}

    def get_game_rules(self) -> str:
        buttons_str = ", ".join(self.buttons)
        return (
            f"You are playing a Multi-Armed Bandit game with {len(self.buttons)} buttons "
            f"({buttons_str}). Each button has a hidden probability of giving a reward "
            f"(0 or 1). You have {self.num_turns} turns to explore the buttons and "
            f"learn their probabilities. On the final turn, select the button you "
            f"believe has the highest probability.\n\n"
            f"Format your action as: [button_name] (e.g., [red])"
        )

    def _last_observation(self) -> str:
        obs = self._env.state.observations
        if obs and 0 in obs:
            for entry in reversed(obs[0]):
                if len(entry) >= 2 and entry[0] != 0:
                    return entry[1]
        return "Game started. Choose a button to press."

    def _get_reward(self) -> float:
        """TextArena returns regret. Convert: 1.0 if optimal, 0.0 otherwise."""
        if hasattr(self._env.state, "outcome") and self._env.state.outcome:
            regret = self._env.state.outcome.get("reward", 0.0)
            return 1.0 if regret == 0.0 else 0.0
        rewards = getattr(self._env.state, "rewards", {})
        if rewards:
            return 1.0 if list(rewards.values())[0] == 0.0 else 0.0
        return 0.0


register_env(
    name="bandit_generator_example",
    env_class=BanditSingleEpisodeEnv,
    default_num_episodes=10,
    description="Bandit example using generator_fn path (no trajectory files needed).",
    buttons=DEFAULT_BUTTONS,
    max_turns_per_episode=DEFAULT_NUM_TURNS,
)
