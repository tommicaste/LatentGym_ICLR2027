"""
Leaderboard generation and comparison utilities.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def compute_leaderboard(
    per_model: Dict[str, Dict[str, Any]],
    primary_metric: str = "avg_mean_reward",
    env_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compute ranked leaderboard from per_model metrics.

    Args:
        per_model: DataStore metrics["per_model"] — model → benchmark_id → metrics
        primary_metric: Metric used for ranking
        env_filter: Only consider benchmark_ids for this env name

    Returns:
        List of dicts sorted by primary_metric descending, with rank added.
    """
    scores = {}
    counts = {}

    for model_name, bid_metrics in per_model.items():
        for bid, m in bid_metrics.items():
            if env_filter and not bid.startswith(env_filter + "/"):
                continue
            val = m.get(primary_metric)
            if val is not None:
                scores[model_name] = scores.get(model_name, 0.0) + float(val)
                counts[model_name] = counts.get(model_name, 0) + 1

    # Average across configs
    avg_scores = {
        model: scores[model] / counts[model]
        for model in scores
        if counts.get(model, 0) > 0
    }

    ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {
            "rank": i + 1,
            "model": name,
            "avg_reward": round(score, 4),
            "n_configs": counts[name],
        }
        for i, (name, score) in enumerate(ranked)
    ]


def compute_model_comparison(
    per_model: Dict[str, Dict[str, Any]],
    model_a: str,
    model_b: str,
) -> Dict[str, Any]:
    """Compare two models head-to-head across all shared benchmark_ids.

    Returns:
        Dict with:
            - wins_a: number of configs where A > B
            - wins_b: number of configs where B > A
            - ties: configs where diff < 0.001
            - per_config_delta: benchmark_id → A_score - B_score
            - avg_delta: mean(A) - mean(B)
    """
    a_metrics = per_model.get(model_a, {})
    b_metrics = per_model.get(model_b, {})
    shared_bids = set(a_metrics.keys()) & set(b_metrics.keys())

    wins_a = 0
    wins_b = 0
    ties = 0
    per_config_delta = {}
    deltas = []

    for bid in sorted(shared_bids):
        a_val = a_metrics[bid].get("avg_mean_reward", 0.0)
        b_val = b_metrics[bid].get("avg_mean_reward", 0.0)
        delta = a_val - b_val
        per_config_delta[bid] = round(delta, 4)
        deltas.append(delta)

        if abs(delta) < 0.001:
            ties += 1
        elif delta > 0:
            wins_a += 1
        else:
            wins_b += 1

    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

    return {
        "model_a": model_a,
        "model_b": model_b,
        "n_configs": len(shared_bids),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "per_config_delta": per_config_delta,
        "avg_delta": round(avg_delta, 4),
    }


def format_leaderboard(leaderboard: List[Dict[str, Any]]) -> str:
    """Format leaderboard as a plain-text table."""
    if not leaderboard:
        return "No results."
    lines = [
        f"{'Rank':>4}  {'Model':<30}  {'Avg Reward':>12}  {'# Configs':>10}",
        "-" * 62,
    ]
    for entry in leaderboard:
        lines.append(
            f"{entry['rank']:>4}  {entry['model']:<30}  "
            f"{entry['avg_reward']:>12.4f}  {entry.get('n_configs', '?'):>10}"
        )
    return "\n".join(lines)
