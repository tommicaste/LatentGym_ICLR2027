"""
Metrics for double-agent evaluation.

Records raw per-episode data and computes:
- Pre/post switch performance
- Transfer effects
- Adaptation speed
- Per-agent breakdown
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from latentgym.eval.types import TrajectoryResult


def compute_double_agent_metrics(
    trajectories: List[TrajectoryResult],
    switch_episode: int,
) -> Dict[str, Any]:
    """Compute metrics for double-agent trajectories.

    Args:
        trajectories: Results from double-agent runs.
        switch_episode: Episode index where agent switched.

    Returns:
        Dict of metric_name → value, including raw per-episode data.
    """
    if not trajectories:
        return {}

    num_episodes = trajectories[0].num_episodes if trajectories else 0

    # ── Raw per-episode data (averaged across trajectories) ──
    per_episode_avg_rewards = []
    per_episode_std_rewards = []
    per_episode_avg_turns = []
    for ep_idx in range(num_episodes):
        ep_rewards = [t.episode_rewards[ep_idx] for t in trajectories if ep_idx < len(t.episode_rewards)]
        ep_turns = [t.episode_turns[ep_idx] for t in trajectories if ep_idx < len(t.episode_turns)]
        if ep_rewards:
            per_episode_avg_rewards.append(float(np.mean(ep_rewards)))
            per_episode_std_rewards.append(float(np.std(ep_rewards)))
        if ep_turns:
            per_episode_avg_turns.append(float(np.mean(ep_turns)))

    # ── Pre/post switch aggregates ──
    pre_switch_rewards = []
    post_switch_rewards = []
    transfer_effects = []

    for t in trajectories:
        pre = [r for i, r in enumerate(t.episode_rewards) if i < switch_episode]
        post = [r for i, r in enumerate(t.episode_rewards) if i >= switch_episode]

        if pre:
            pre_switch_rewards.append(float(np.mean(pre)))
        if post:
            post_switch_rewards.append(float(np.mean(post)))
        if pre and post:
            transfer_effects.append(float(np.mean(post)) - float(np.mean(pre)))

    # ── Adaptation speed: reward change in first few episodes after switch ──
    adaptation_rewards = []
    for t in trajectories:
        post = [r for i, r in enumerate(t.episode_rewards) if i >= switch_episode]
        if len(post) >= 2:
            adaptation_rewards.append(post[1] - post[0])

    # ── Per-agent breakdown ──
    per_agent = {}
    for t in trajectories:
        for o in t.episode_outcomes:
            name = o.agent_name
            if name not in per_agent:
                per_agent[name] = {"rewards": [], "turns": [], "successes": []}
            per_agent[name]["rewards"].append(o.reward)
            per_agent[name]["turns"].append(o.turns)
            per_agent[name]["successes"].append(o.success)

    per_agent_metrics = {}
    for name, data in per_agent.items():
        per_agent_metrics[name] = {
            "avg_reward": float(np.mean(data["rewards"])),
            "std_reward": float(np.std(data["rewards"])),
            "avg_turns": float(np.mean(data["turns"])),
            "success_rate": float(np.mean(data["successes"])),
            "n_episodes": len(data["rewards"]),
        }

    return {
        # Switch info
        "switch_episode": switch_episode,
        "n_trajectories": len(trajectories),
        "n_episodes": num_episodes,

        # Pre/post switch
        "avg_pre_switch_reward": float(np.mean(pre_switch_rewards)) if pre_switch_rewards else 0.0,
        "std_pre_switch_reward": float(np.std(pre_switch_rewards)) if pre_switch_rewards else 0.0,
        "avg_post_switch_reward": float(np.mean(post_switch_rewards)) if post_switch_rewards else 0.0,
        "std_post_switch_reward": float(np.std(post_switch_rewards)) if post_switch_rewards else 0.0,

        # Transfer
        "avg_transfer_effect": float(np.mean(transfer_effects)) if transfer_effects else 0.0,
        "std_transfer_effect": float(np.std(transfer_effects)) if transfer_effects else 0.0,

        # Adaptation
        "avg_adaptation_speed": float(np.mean(adaptation_rewards)) if adaptation_rewards else 0.0,

        # Per-episode raw data
        "per_episode_avg_rewards": per_episode_avg_rewards,
        "per_episode_std_rewards": per_episode_std_rewards,
        "per_episode_avg_turns": per_episode_avg_turns,

        # Per-agent breakdown
        "per_agent": per_agent_metrics,
    }
