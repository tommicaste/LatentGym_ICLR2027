"""
RewardAggregator — Reward computation for multi-episode trajectories.

Defines how per-episode rewards are aggregated into trajectory-level rewards.
Four reward types:
    - CUMULATIVE: Sum of all episode rewards
    - TERMINAL: Last episode reward only
    - IMPROVEMENT: Last - first episode reward
    - PER_EPISODE: Each episode's reward returned at episode boundary

Note: reward_type is primarily for training (shapes RL reward signal).
For eval, all metrics are always computed regardless of reward_type.
"""
from __future__ import annotations

from enum import Enum
from typing import List


class RewardType(Enum):
    """How per-episode rewards are aggregated for training."""
    CUMULATIVE = "cumulative"      # Sum of all episode rewards
    TERMINAL = "terminal"          # Last episode reward only
    IMPROVEMENT = "improvement"    # Last - first episode reward
    PER_EPISODE = "per_episode"    # Each episode's reward at episode boundary


class RewardAggregator:
    """Computes trajectory-level rewards from per-episode rewards.

    Used by MultiEpisodeEnv to determine what reward to return at each step.

    Args:
        reward_type: How to aggregate episode rewards.
    """

    def __init__(self, reward_type: RewardType):
        self.reward_type = reward_type

    def compute_step_reward(
        self,
        episode_rewards: List[float],
        episode_just_ended: bool,
        trajectory_done: bool,
    ) -> float:
        """Compute the reward to return at a single step.

        For CUMULATIVE/TERMINAL/IMPROVEMENT: returns 0 until trajectory ends,
        then returns the final aggregated reward.
        For PER_EPISODE: returns the episode reward when an episode ends, else 0.

        Args:
            episode_rewards: List of rewards for all completed episodes so far
            episode_just_ended: Whether an episode just completed on this step
            trajectory_done: Whether the entire trajectory is done

        Returns:
            Reward value for this step.
        """
        if self.reward_type == RewardType.PER_EPISODE:
            if episode_just_ended and episode_rewards:
                return episode_rewards[-1]
            return 0.0
        else:
            # CUMULATIVE, TERMINAL, IMPROVEMENT: reward only at trajectory end
            if trajectory_done:
                return self.compute_final_reward(episode_rewards)
            return 0.0

    def compute_final_reward(self, episode_rewards: List[float]) -> float:
        """Compute the trajectory-level reward from all episode rewards.

        Args:
            episode_rewards: Complete list of per-episode rewards.

        Returns:
            Aggregated reward.
        """
        if not episode_rewards:
            return 0.0

        if self.reward_type == RewardType.CUMULATIVE:
            return sum(episode_rewards)
        elif self.reward_type == RewardType.TERMINAL:
            return episode_rewards[-1]
        elif self.reward_type == RewardType.IMPROVEMENT:
            if len(episode_rewards) >= 2:
                return episode_rewards[-1] - episode_rewards[0]
            return episode_rewards[-1] if episode_rewards else 0.0
        elif self.reward_type == RewardType.PER_EPISODE:
            # For per_episode, final reward is the sum (used when computing
            # a single trajectory-level number for reporting)
            return sum(episode_rewards)
        else:
            return sum(episode_rewards)
