"""
Metrics computation for single-agent evaluation.

Matches the metrics from skyrl_train/eval/metrics.py:
- compute_single_agent_metrics: per-config aggregated metrics
- compute_comparison_metrics: finetuned vs base model comparison
- compute_detailed_metrics: exploration/exploitation phase analysis
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from latentgym.eval.types import TrajectoryResult


def compute_single_agent_metrics(trajectories: List[TrajectoryResult]) -> Dict[str, Any]:
    """Compute aggregated metrics for a set of trajectories from one config.

    Matches compute_per_config_metrics from old eval code.

    Args:
        trajectories: Results from running one model on one env config.

    Returns:
        Dict of metric_name → value.
    """
    if not trajectories:
        return {}

    # Trajectory-level metrics
    cumulative_rewards = [t.cumulative_reward for t in trajectories]
    terminal_rewards = [t.terminal_reward for t in trajectories]
    initial_rewards = [t.initial_reward for t in trajectories]
    improvements = [t.improvement for t in trajectories]
    mean_rewards = [t.mean_reward for t in trajectories]
    success_rates = [t.success_rate for t in trajectories]
    total_turns = [t.total_turns for t in trajectories]
    mean_turns_list = [t.mean_turns for t in trajectories]

    # Per-episode breakdown
    num_episodes = trajectories[0].num_episodes if trajectories else 0
    per_episode_avg_rewards = []
    per_episode_std_rewards = []
    per_episode_avg_turns = []

    for ep_idx in range(num_episodes):
        ep_rewards = [
            t.episode_rewards[ep_idx]
            for t in trajectories
            if ep_idx < len(t.episode_rewards)
        ]
        ep_turns = [
            t.episode_turns[ep_idx]
            for t in trajectories
            if ep_idx < len(t.episode_turns)
        ]
        if ep_rewards:
            per_episode_avg_rewards.append(float(np.mean(ep_rewards)))
            per_episode_std_rewards.append(float(np.std(ep_rewards)))
        if ep_turns:
            per_episode_avg_turns.append(float(np.mean(ep_turns)))

    return {
        # Trajectory-level
        "avg_trajectory_reward": float(np.mean(cumulative_rewards)),
        "std_trajectory_reward": float(np.std(cumulative_rewards)),
        "min_trajectory_reward": float(np.min(cumulative_rewards)),
        "max_trajectory_reward": float(np.max(cumulative_rewards)),

        # Episode-level aggregates
        "avg_initial_reward": float(np.mean(initial_rewards)),
        "avg_final_reward": float(np.mean(terminal_rewards)),
        "avg_improvement": float(np.mean(improvements)),
        "std_improvement": float(np.std(improvements)),
        "avg_cumulative_reward": float(np.mean(cumulative_rewards)),
        "avg_mean_reward": float(np.mean(mean_rewards)),
        "avg_success_rate": float(np.mean(success_rates)),

        # Turn metrics
        "avg_total_turns": float(np.mean(total_turns)),
        "std_total_turns": float(np.std(total_turns)),
        "avg_mean_turns_per_episode": float(np.mean(mean_turns_list)),

        # Per-episode breakdown
        "per_episode_avg_rewards": per_episode_avg_rewards,
        "per_episode_std_rewards": per_episode_std_rewards,
        "per_episode_avg_turns": per_episode_avg_turns,

        # Learning metrics
        "learning_slope": float(np.mean(improvements)) / max(num_episodes - 1, 1),

        # Counts
        "n_trajectories": len(trajectories),
        "n_episodes": num_episodes,
    }


def compute_comparison_metrics(
    results: Dict[str, List[TrajectoryResult]],
    finetuned_key: str = "P_f",
    base_key: str = "Q_b",
    f_then_b_key: str = "P_f->Q_b",
    b_then_f_key: str = "Q_b->P_f",
) -> Dict[str, Any]:
    """Compute comparison metrics between model configurations.

    Compares finetuned vs base, and optionally transfer schedules.
    Matches compute_comparison_metrics from old eval code.

    Args:
        results: Dict mapping config name → trajectory results
            Expected keys: "P_f" (finetuned), "Q_b" (base),
            optionally "P_f->Q_b" and "Q_b->P_f" (transfer schedules)

    Returns:
        Dict with per_config metrics and cross-config comparisons.
    """
    # Compute per-config metrics
    per_config = {
        name: compute_single_agent_metrics(trajectories)
        for name, trajectories in results.items()
    }

    p_f = per_config.get(finetuned_key, {})
    q_b = per_config.get(base_key, {})
    p_f_q_b = per_config.get(f_then_b_key, {})
    q_b_p_f = per_config.get(b_then_f_key, {})

    # Cross-config comparisons
    overall_improvement = p_f.get("avg_final_reward", 0) - q_b.get("avg_final_reward", 0)
    initial_prior_improvement = p_f.get("avg_initial_reward", 0) - q_b.get("avg_initial_reward", 0)
    icl_difference = p_f.get("avg_improvement", 0) - q_b.get("avg_improvement", 0)

    # Transfer effects
    transfer_f_to_b = None
    transfer_b_to_f = None
    if f_then_b_key in per_config and base_key in per_config:
        transfer_f_to_b = p_f_q_b.get("avg_trajectory_reward", 0) - q_b.get("avg_trajectory_reward", 0)
    if b_then_f_key in per_config and finetuned_key in per_config:
        transfer_b_to_f = q_b_p_f.get("avg_trajectory_reward", 0) - p_f.get("avg_trajectory_reward", 0)

    # Turn efficiency
    turn_efficiency_difference = None
    if p_f and q_b:
        turn_efficiency_difference = (
            q_b.get("avg_mean_turns_per_episode", 0) - p_f.get("avg_mean_turns_per_episode", 0)
        )

    return {
        "per_config": per_config,
        "overall_improvement": overall_improvement,
        "initial_prior_improvement": initial_prior_improvement,
        "icl_difference": icl_difference,
        "transfer_f_to_b": transfer_f_to_b,
        "transfer_b_to_f": transfer_b_to_f,
        "turn_efficiency_difference": turn_efficiency_difference,
    }


def compute_detailed_metrics(
    results: Dict[str, List[TrajectoryResult]],
    k: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute detailed per-episode metrics with exploration/exploitation phases.

    Matches compute_detailed_metrics from old eval code.

    Args:
        results: Dict mapping config name → trajectory results
        k: Exploration/exploitation boundary (episode index).
           If None, defaults to num_episodes // 2.

    Returns:
        Dict: config_name → detailed metrics
    """
    metrics = {}

    for name, trajectories in results.items():
        if not trajectories:
            continue

        all_ep_rewards = [t.episode_rewards for t in trajectories]
        all_ep_turns = [t.episode_turns for t in trajectories]
        n_episodes = len(all_ep_rewards[0]) if all_ep_rewards else 0

        boundary = k if k is not None else n_episodes // 2

        # Per-episode averages
        per_episode_avg = [
            float(np.mean([traj[i] for traj in all_ep_rewards if i < len(traj)]))
            for i in range(n_episodes)
        ]
        per_episode_std = [
            float(np.std([traj[i] for traj in all_ep_rewards if i < len(traj)]))
            for i in range(n_episodes)
        ]
        per_episode_turns_avg = [
            float(np.mean([traj[i] for traj in all_ep_turns if i < len(traj)]))
            for i in range(n_episodes)
        ]

        # Exploration vs exploitation phases
        early_rewards = per_episode_avg[:boundary] if boundary > 0 else []
        late_rewards = per_episode_avg[boundary:] if boundary < n_episodes else []
        early_turns = per_episode_turns_avg[:boundary] if boundary > 0 else []
        late_turns = per_episode_turns_avg[boundary:] if boundary < n_episodes else []

        metrics[name] = {
            # Per-episode data
            "per_episode_avg_reward": per_episode_avg,
            "per_episode_std_reward": per_episode_std,
            "per_episode_avg_turns": per_episode_turns_avg,

            # Basic metrics
            "initial_reward": per_episode_avg[0] if per_episode_avg else 0,
            "final_reward": per_episode_avg[-1] if per_episode_avg else 0,
            "improvement": (per_episode_avg[-1] - per_episode_avg[0]) if len(per_episode_avg) >= 2 else 0,

            # Exploration phase
            "exploration_mean_reward": float(np.mean(early_rewards)) if early_rewards else 0,
            "exploration_std_reward": float(np.std(early_rewards)) if early_rewards else 0,
            "exploration_mean_turns": float(np.mean(early_turns)) if early_turns else 0,

            # Exploitation phase
            "exploitation_mean_reward": float(np.mean(late_rewards)) if late_rewards else 0,
            "exploitation_std_reward": float(np.std(late_rewards)) if late_rewards else 0,
            "exploitation_mean_turns": float(np.mean(late_turns)) if late_turns else 0,

            # Learning curve
            "learning_slope": (per_episode_avg[-1] - per_episode_avg[0]) / (n_episodes - 1) if n_episodes > 1 else 0,

            # Counts
            "n_episodes": n_episodes,
            "exploration_exploitation_boundary": boundary,
        }

    return metrics


def compute_per_agent_metrics(
    results: Dict[str, List[TrajectoryResult]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute metrics broken down by which agent played each episode.

    Matches compute_per_agent_metrics from old eval code.

    Args:
        results: Dict mapping config name → trajectory results

    Returns:
        Nested dict: config_name → agent_name → metrics
    """
    per_agent = {}

    for config_name, trajectories in results.items():
        if not trajectories:
            continue

        per_agent[config_name] = {}

        # Get all unique agent names
        agent_names = set()
        for t in trajectories:
            for o in t.episode_outcomes:
                agent_names.add(o.agent_name)

        for agent_name in agent_names:
            agent_rewards = []
            agent_turns = []
            agent_successes = []

            for t in trajectories:
                for o in t.episode_outcomes:
                    if o.agent_name == agent_name:
                        agent_rewards.append(o.reward)
                        agent_turns.append(o.turns)
                        agent_successes.append(o.success)

            if agent_rewards:
                per_agent[config_name][agent_name] = {
                    "avg_reward": float(np.mean(agent_rewards)),
                    "std_reward": float(np.std(agent_rewards)),
                    "avg_turns": float(np.mean(agent_turns)),
                    "success_rate": float(np.mean(agent_successes)),
                    "n_episodes": len(agent_rewards),
                }

    return per_agent


def format_metrics_summary(
    per_config: Dict[str, Dict[str, Any]],
    comparison: Optional[Dict[str, Any]] = None,
    detailed: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Format metrics as a human-readable summary string.

    Args:
        per_config: Per-configuration metrics from compute_single_agent_metrics
        comparison: Optional comparison metrics
        detailed: Optional detailed phase metrics

    Returns:
        Formatted string summary.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION METRICS SUMMARY")
    lines.append("=" * 60)

    # Per-config
    for name, m in per_config.items():
        lines.append(f"\n{name}:")
        lines.append(f"  Avg Trajectory Reward: {m.get('avg_trajectory_reward', 0):.4f} "
                     f"(+/- {m.get('std_trajectory_reward', 0):.4f})")
        lines.append(f"  Avg Initial Reward: {m.get('avg_initial_reward', 0):.4f}")
        lines.append(f"  Avg Final Reward: {m.get('avg_final_reward', 0):.4f}")
        lines.append(f"  Avg Improvement (ICL): {m.get('avg_improvement', 0):+.4f}")
        lines.append(f"  Learning Slope: {m.get('learning_slope', 0):+.4f}")
        lines.append(f"  Avg Success Rate: {m.get('avg_success_rate', 0):.2%}")
        lines.append(f"  Avg Turns/Episode: {m.get('avg_mean_turns_per_episode', 0):.2f}")

    # Comparison
    if comparison:
        lines.append("\n--- Comparison Metrics ---")
        lines.append(f"Overall Improvement (finetuned vs base): {comparison.get('overall_improvement', 0):+.4f}")
        lines.append(f"Initial Prior Improvement: {comparison.get('initial_prior_improvement', 0):+.4f}")
        lines.append(f"ICL Difference: {comparison.get('icl_difference', 0):+.4f}")
        if comparison.get("transfer_f_to_b") is not None:
            lines.append(f"Transfer Effect (F→B): {comparison['transfer_f_to_b']:+.4f}")
        if comparison.get("transfer_b_to_f") is not None:
            lines.append(f"Transfer Effect (B→F): {comparison['transfer_b_to_f']:+.4f}")
        if comparison.get("turn_efficiency_difference") is not None:
            lines.append(f"Turn Efficiency Difference: {comparison['turn_efficiency_difference']:+.4f}")

    # Detailed
    if detailed:
        lines.append("\n--- Phase Metrics ---")
        for name, d in detailed.items():
            lines.append(f"\n{name}:")
            boundary = d.get("exploration_exploitation_boundary", 0)
            lines.append(f"  Exploration (first {boundary} eps):")
            lines.append(f"    Mean Reward: {d.get('exploration_mean_reward', 0):.4f} "
                        f"(+/- {d.get('exploration_std_reward', 0):.4f})")
            lines.append(f"  Exploitation (remaining eps):")
            lines.append(f"    Mean Reward: {d.get('exploitation_mean_reward', 0):.4f} "
                        f"(+/- {d.get('exploitation_std_reward', 0):.4f})")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
