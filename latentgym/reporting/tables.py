"""
Table generation from DataStore metrics.

Produces markdown, LaTeX, and CSV tables for benchmark results.
"""
from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Optional


def make_results_table(
    per_model: Dict[str, Dict[str, Any]],
    env_filter: Optional[str] = None,
    metric: str = "avg_mean_reward",
    fmt: str = "markdown",
) -> str:
    """Build a model × benchmark_id results table.

    Args:
        per_model: DataStore metrics["per_model"]
        env_filter: If set, only include benchmark_ids starting with this env name
        metric: Which metric to display
        fmt: "markdown", "latex", or "csv"

    Returns:
        Formatted table string
    """
    # Collect all benchmark_ids (columns)
    all_bids: set = set()
    for bid_metrics in per_model.values():
        all_bids.update(bid_metrics.keys())
    all_bids = sorted(all_bids)

    if env_filter:
        all_bids = [b for b in all_bids if b.startswith(env_filter + "/")]

    if not all_bids:
        return f"No results for env_filter={env_filter!r}"

    models = sorted(per_model.keys())

    # Shorten benchmark_ids for display
    short_bids = [b.split("/")[1] if "/" in b else b for b in all_bids]

    rows = []
    for model in models:
        row = [model]
        for bid in all_bids:
            val = per_model.get(model, {}).get(bid, {}).get(metric, None)
            row.append(f"{val:.4f}" if val is not None else "—")
        rows.append(row)

    headers = ["Model"] + short_bids

    if fmt == "markdown":
        return _to_markdown(headers, rows)
    elif fmt == "latex":
        return _to_latex(headers, rows, caption=f"Results: {metric}")
    elif fmt == "csv":
        return _to_csv(headers, rows)
    else:
        raise ValueError(f"Unknown fmt: {fmt}")


def make_leaderboard_table(
    leaderboard: List[Dict[str, Any]],
    fmt: str = "markdown",
) -> str:
    """Build leaderboard table.

    Args:
        leaderboard: DataStore metrics["leaderboard"]
        fmt: "markdown", "latex", or "csv"

    Returns:
        Formatted table string
    """
    headers = ["Rank", "Model", "Avg Reward"]
    rows = [
        [str(entry["rank"]), entry["model"], f"{entry['avg_reward']:.4f}"]
        for entry in leaderboard
    ]

    if fmt == "markdown":
        return _to_markdown(headers, rows)
    elif fmt == "latex":
        return _to_latex(headers, rows, caption="Benchmark Leaderboard")
    elif fmt == "csv":
        return _to_csv(headers, rows)
    else:
        raise ValueError(f"Unknown fmt: {fmt}")


def make_per_env_table(
    per_env: Dict[str, Any],
    metric: str = "avg_mean_reward",
    fmt: str = "markdown",
) -> str:
    """Build env × model table showing one aggregated metric.

    Args:
        per_env: DataStore metrics["per_env"]
        metric: Which metric to aggregate per env
        fmt: "markdown", "latex", or "csv"
    """
    env_names = sorted(per_env.keys())
    model_names: set = set()
    for model_data in per_env.values():
        model_names.update(model_data.keys())
    model_names = sorted(model_names)

    headers = ["Environment"] + model_names
    rows = []
    for env_name in env_names:
        row = [env_name]
        for model_name in model_names:
            val = per_env.get(env_name, {}).get(model_name, {}).get(metric, None)
            row.append(f"{val:.4f}" if val is not None else "—")
        rows.append(row)

    if fmt == "markdown":
        return _to_markdown(headers, rows)
    elif fmt == "latex":
        return _to_latex(headers, rows, caption=f"Per-Environment Results: {metric}")
    elif fmt == "csv":
        return _to_csv(headers, rows)
    else:
        raise ValueError(f"Unknown fmt: {fmt}")


def make_complexity_table(
    per_complexity: Dict[str, Any],
    metric: str = "avg_mean_reward",
    fmt: str = "markdown",
) -> str:
    """Build latent complexity × model table."""
    complexities = sorted(per_complexity.keys())
    model_names: set = set()
    for model_data in per_complexity.values():
        model_names.update(model_data.keys())
    model_names = sorted(model_names)

    headers = ["Complexity"] + model_names
    rows = []
    for complexity in complexities:
        row = [complexity]
        for model_name in model_names:
            val = per_complexity.get(complexity, {}).get(model_name, {}).get(metric, None)
            row.append(f"{val:.4f}" if val is not None else "—")
        rows.append(row)

    if fmt == "markdown":
        return _to_markdown(headers, rows)
    elif fmt == "latex":
        return _to_latex(headers, rows, caption=f"Results by Latent Complexity: {metric}")
    elif fmt == "csv":
        return _to_csv(headers, rows)
    else:
        raise ValueError(f"Unknown fmt: {fmt}")


# =============================================================================
# Double-agent tables
# =============================================================================

def make_double_agent_summary_table(
    per_schedule: Dict[str, Any],
    fmt: str = "markdown",
) -> str:
    """Build double-agent summary table with pre/post switch and transfer metrics.

    Args:
        per_schedule: DataStore metrics["double_agent"]["per_schedule"]
        fmt: "markdown", "latex", or "csv"
    """
    headers = [
        "Schedule", "Switch Ep", "Pre-Switch Reward", "Post-Switch Reward",
        "Transfer Effect", "Adaptation Speed", "# Traj",
    ]
    rows = []
    for key, m in sorted(per_schedule.items()):
        rows.append([
            key,
            str(m.get("switch_episode", 0)),
            f"{m.get('avg_pre_switch_reward', 0):.4f}",
            f"{m.get('avg_post_switch_reward', 0):.4f}",
            f"{m.get('avg_transfer_effect', 0):+.4f}",
            f"{m.get('avg_adaptation_speed', 0):+.4f}",
            str(m.get("n_trajectories", 0)),
        ])

    if not rows:
        return "No double-agent results."

    if fmt == "markdown":
        return _to_markdown(headers, rows)
    elif fmt == "latex":
        return _to_latex(headers, rows, caption="Double-Agent Summary")
    elif fmt == "csv":
        return _to_csv(headers, rows)
    else:
        raise ValueError(f"Unknown fmt: {fmt}")


def make_per_agent_table(
    per_agent: Dict[str, Any],
    fmt: str = "markdown",
) -> str:
    """Build per-agent breakdown table.

    Args:
        per_agent: DataStore metrics["double_agent"]["per_agent"]
        fmt: "markdown", "latex", or "csv"
    """
    headers = ["Schedule", "Agent", "Avg Reward", "Std Reward", "Success Rate", "Avg Turns", "# Episodes"]
    rows = []
    for key, agents in sorted(per_agent.items()):
        for agent_name, m in sorted(agents.items()):
            rows.append([
                key,
                agent_name,
                f"{m.get('avg_reward', 0):.4f}",
                f"{m.get('std_reward', 0):.4f}",
                f"{m.get('success_rate', 0):.4f}",
                f"{m.get('avg_turns', 0):.2f}",
                str(m.get("n_episodes", 0)),
            ])

    if not rows:
        return "No per-agent results."

    if fmt == "markdown":
        return _to_markdown(headers, rows)
    elif fmt == "latex":
        return _to_latex(headers, rows, caption="Per-Agent Breakdown")
    elif fmt == "csv":
        return _to_csv(headers, rows)
    else:
        raise ValueError(f"Unknown fmt: {fmt}")


def make_comparison_table(
    comparisons: Dict[str, Any],
    fmt: str = "markdown",
) -> str:
    """Build comparison table (finetuned vs base) per benchmark_id.

    Args:
        comparisons: DataStore metrics["double_agent"]["comparisons"]
        fmt: "markdown", "latex", or "csv"
    """
    per_bid = comparisons.get("per_benchmark_id", {})
    if not per_bid:
        return "No comparison results."

    headers = [
        "Benchmark ID", "Overall Improvement", "Initial Prior Impr.",
        "ICL Difference", "Transfer F→B", "Transfer B→F",
    ]
    rows = []
    for bid, m in sorted(per_bid.items()):
        rows.append([
            bid,
            f"{m.get('overall_improvement', 0):+.4f}",
            f"{m.get('initial_prior_improvement', 0):+.4f}",
            f"{m.get('icl_difference', 0):+.4f}",
            f"{m.get('transfer_f_to_b', 0):+.4f}" if m.get("transfer_f_to_b") is not None else "—",
            f"{m.get('transfer_b_to_f', 0):+.4f}" if m.get("transfer_b_to_f") is not None else "—",
        ])

    if fmt == "markdown":
        return _to_markdown(headers, rows)
    elif fmt == "latex":
        return _to_latex(headers, rows, caption="Model Comparison: Finetuned vs Base")
    elif fmt == "csv":
        return _to_csv(headers, rows)
    else:
        raise ValueError(f"Unknown fmt: {fmt}")


def make_exploration_exploitation_table(
    detailed: Dict[str, Any],
    fmt: str = "markdown",
) -> str:
    """Build exploration vs exploitation phase comparison table.

    Args:
        detailed: DataStore metrics["detailed"]["exploration_exploitation"]
        fmt: "markdown", "latex", or "csv"
    """
    if not detailed:
        return "No detailed metrics."

    headers = [
        "Config", "Exploration Reward", "Exploitation Reward",
        "Improvement", "Learning Slope", "Boundary Ep",
    ]
    rows = []
    for key, m in sorted(detailed.items()):
        rows.append([
            key,
            f"{m.get('exploration_mean_reward', 0):.4f}",
            f"{m.get('exploitation_mean_reward', 0):.4f}",
            f"{m.get('improvement', 0):+.4f}",
            f"{m.get('learning_slope', 0):+.4f}",
            str(m.get("exploration_exploitation_boundary", 0)),
        ])

    if fmt == "markdown":
        return _to_markdown(headers, rows)
    elif fmt == "latex":
        return _to_latex(headers, rows, caption="Exploration vs Exploitation Phases")
    elif fmt == "csv":
        return _to_csv(headers, rows)
    else:
        raise ValueError(f"Unknown fmt: {fmt}")


# =============================================================================
# Formatting helpers
# =============================================================================

def _to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    # Column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    lines = [
        fmt_row(headers),
        "| " + " | ".join("-" * w for w in widths) + " |",
    ] + [fmt_row(row) for row in rows]
    return "\n".join(lines)


def _to_latex(headers: List[str], rows: List[List[str]], caption: str = "") -> str:
    n_cols = len(headers)
    col_spec = "l" + "r" * (n_cols - 1)
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
    ]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _to_csv(headers: List[str], rows: List[List[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerows(rows)
    return buf.getvalue()
