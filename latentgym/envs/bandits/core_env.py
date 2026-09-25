"""
Bandit SingleEpisodeEnv — wraps TextArena's BanditEnv with early stopping.

The agent explores buttons to learn their hidden reward probabilities, then
locks in a final answer with [select button_name]. Earlier correct selections
get higher reward — this incentivizes learning the latent pattern across
episodes to skip exploration.

Actions:
    [red]           — explore: press button, observe 0/1 reward
    [select red]    — lock in final answer, episode ends immediately

Reward:
    Correct (selected the highest-prob button): 1.0 - turns_used * DECAY
    Wrong: 0.0
"""
from __future__ import annotations

import re
import random
from typing import Any, Dict, List, Tuple

from textarena.envs.Bandit.env import BanditEnv

from latentgym.core.single_episode_env import SingleEpisodeEnv
from latentgym.core.registry import register_env

DEFAULT_BUTTONS = ["red", "blue", "green", "yellow", "purple"]
DEFAULT_NUM_TURNS = 30


class BanditSingleEpisodeEnv(SingleEpisodeEnv):
    """Multi-armed bandits with early stopping.

    The agent can explore with [button_name] or lock in with [select button_name].
    Earlier correct selections get higher reward.

    Episode config must contain:
        - ground_truth: Dict[str, float] mapping button names to probabilities
    """

    REWARD_DECAY = 0.015  # Per-turn penalty

    def __init__(self):
        self.buttons = DEFAULT_BUTTONS
        self.num_turns = DEFAULT_NUM_TURNS
        self.include_summary = False
        self._bandit_env: BanditEnv | None = None
        self._turn = 0
        self._selected = None  # The button the agent locked in
        self._ground_truth = {}
        self._env_params_key: tuple = ()

    def reset(self, episode_config: Dict[str, Any]) -> str:
        """Reset for a new episode with the given ground truth probabilities."""
        self._ground_truth = episode_config.get("ground_truth", {})
        num_turns = episode_config.get("max_turns_per_episode",
                    episode_config.get("num_turns", self.num_turns))
        buttons = episode_config.get("buttons", self.buttons)

        params_key = (tuple(buttons), num_turns, self.include_summary)
        if self._bandit_env is None or params_key != self._env_params_key:
            self._bandit_env = BanditEnv(
                buttons=buttons,
                num_turns=num_turns - 1,
                include_summary=self.include_summary,
            )
            self._env_params_key = params_key

        self._bandit_env.reset(num_players=1)

        # Override ground truth probabilities
        self._bandit_env.state.game_state["ground_truth"] = self._ground_truth.copy()
        self._bandit_env.state.game_state["history"] = {b: [] for b in buttons}

        # Clear TextArena's default observations (its prompt conflicts with ours)
        self._bandit_env.state.observations = {0: []}

        self._turn = 0
        self._selected = None
        self.buttons = buttons
        self.num_turns = num_turns

        buttons_str = ", ".join(buttons)
        return (
            f"Game started. You have {len(buttons)} buttons: {buttons_str}. "
            f"You have up to {num_turns} turns. Explore with [button_name] "
            f"or lock in your answer with [select button_name]."
        )

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute one turn.

        If action matches [select X], lock in X as final answer and end.
        Otherwise, pass to TextArena for normal exploration.
        """
        self._turn += 1

        # Check for early stopping: [select button_name], Select: [button], etc.
        select_match = re.search(r'\[select\s+(\w+)\]', action, re.IGNORECASE)
        if not select_match:
            # Also match "Select: [red]" or "select [red]" patterns
            select_match = re.search(r'select[:\s]+\[(\w+)\]', action, re.IGNORECASE)
        if select_match:
            selected_button = select_match.group(1).lower()
            return self._handle_selection(selected_button)

        # Normal exploration turn — pass to TextArena
        episode_done, info = self._bandit_env.step(action)
        if not episode_done and self._bandit_env.state.turn >= self.num_turns:
            episode_done = True
        feedback = self._extract_feedback()

        if episode_done:
            # TextArena forced end (last turn) — treat last action as selection
            # Extract button name from action
            button_match = re.search(r'\[(\w+)\]', action)
            if button_match:
                selected_button = button_match.group(1).lower()
                return self._handle_selection(selected_button)
            # Fallback: no valid button found
            return feedback, 0.0, True, self._make_info(action, None)

        return feedback, 0.0, False, self._make_info(action, None)

    def _handle_selection(self, selected_button: str) -> Tuple[str, float, bool, Dict]:
        """Handle final selection — compute reward and end episode."""
        self._selected = selected_button

        # Find the best button
        best_button = max(self._ground_truth, key=self._ground_truth.get)
        is_correct = (selected_button == best_button)

        # Reward: correct = 1.0 - turns * DECAY, wrong = 0.0
        if is_correct:
            reward = max(0.0, 1.0 - self._turn * self.REWARD_DECAY)
        else:
            reward = 0.0

        feedback = (
            f"You selected '{selected_button}' as your final answer on turn {self._turn}. "
        )
        if is_correct:
            feedback += "Correct!"
        else:
            feedback += f"Wrong! The highest probability button was not '{selected_button}'."

        return feedback, reward, True, self._make_info(f"select {selected_button}", selected_button)

    def _make_info(self, action: str, selected: str = None) -> Dict[str, Any]:
        """Build info dict for step return."""
        best_button = max(self._ground_truth, key=self._ground_truth.get) if self._ground_truth else None
        return {
            "turn": self._turn,
            "action": action,
            "selected_button": selected,
            "best_button": best_button,
            "ground_truth": self._ground_truth,
            "is_correct": selected == best_button if selected else None,
        }

    def get_game_rules(self) -> str:
        """Return game rules for prompt construction."""
        buttons_str = ", ".join(self.buttons)
        return (
            f"You are playing a Multi-Armed Bandit game with {len(self.buttons)} buttons "
            f"({buttons_str}). Each button has a hidden probability of giving a reward "
            f"(0 or 1). You have up to {self.num_turns} turns.\n\n"
            f"On each turn you can either:\n"
            f"  - Explore: [button_name] (e.g., [red]) — press a button and observe the reward\n"
            f"  - Select: [select button_name] (e.g., [select red]) — lock in your final answer\n\n"
            f"Once you select, the game ends immediately. Your goal is to identify the "
            f"button with the highest probability. Selecting correctly earlier gives a "
            f"higher reward. If you don't select by the last turn, your last explore "
            f"action counts as your selection.\n\n"
            f"IMPORTANT: Only use square brackets [...] to submit your action for this turn "
            f"(either [button_name] to explore or [select button_name] to finalize). "
            f"Do not use square brackets anywhere else in your response."
        )

    def get_env_info(self) -> Dict[str, Any]:
        return {
            "buttons": self.buttons,
            "num_turns": self.num_turns,
        }

    def _get_initial_observation(self) -> str:
        """Extract initial observation from BanditEnv."""
        if not self._bandit_env:
            return "Game started. Choose a button to press or select your final answer."
        obs_dict = self._bandit_env.state.observations
        if obs_dict and 0 in obs_dict:
            for t in reversed(obs_dict[0]):
                if len(t) >= 2 and t[0] != 0:
                    return t[1]
        return "Game started. Choose a button to press or select your final answer."

    def _extract_feedback(self) -> str:
        """Extract the latest feedback from BanditEnv's observations."""
        obs_dict = self._bandit_env.state.observations
        if obs_dict and 0 in obs_dict:
            messages = obs_dict[0]
            if messages:
                last = messages[-1]
                if len(last) >= 2 and last[0] != 0:
                    return last[1]
        return "Continue playing."

    def close(self) -> None:
        pass


# Register the environment
register_env(
    name="bandits",
    env_class=BanditSingleEpisodeEnv,
    default_num_episodes=10,
    description="Multi-armed bandit with 5 buttons. Explore and identify the best button.",
    buttons=DEFAULT_BUTTONS,
    max_turns_per_episode=DEFAULT_NUM_TURNS,
)
