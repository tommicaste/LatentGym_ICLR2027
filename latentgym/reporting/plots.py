"""
Plot generation from DataStore metrics.

Produces matplotlib figures for learning curves, bar charts, and heatmaps.
All functions return matplotlib Figure objects (caller saves/shows).

Requires: matplotlib, numpy
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def plot_learning_curves(
    per_model: Dict[str, Dict[str, Any]],
    benchmark_id: str,
    metric: str = "per_episode_avg_rewards",
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Plot per-episode reward curves for all models on one benchmark_id.

    Args:
        per_model: DataStore metrics["per_model"]
        benchmark_id: Which env config to plot
        metric: "per_episode_avg_rewards" or "per_episode_avg_turns"
        title: Plot title (defaults to benchmark_id)

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title or f"Learning Curve: {benchmark_id}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward" if "reward" in metric else "Avg Turns")

    for model_name, bid_metrics in per_model.items():
        m = bid_metrics.get(benchmark_id, {})
        y = m.get(metric, [])
        if not y:
            continue
        x = list(range(len(y)))
        std = m.get("per_episode_std_rewards", [0] * len(y))
        ax.plot(x, y, marker="o", markersize=3, label=model_name)
        ax.fill_between(x,
                        [yi - si for yi, si in zip(y, std)],
                        [yi + si for yi, si in zip(y, std)],
                        alpha=0.15)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_env_bar_chart(
    per_env: Dict[str, Any],
    metric: str = "avg_mean_reward",
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Bar chart: models side-by-side, grouped by environment.

    Args:
        per_env: DataStore metrics["per_env"]
        metric: Which metric to plot
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import numpy as np

    env_names = sorted(per_env.keys())
    model_names: list = sorted({
        m for env_data in per_env.values() for m in env_data.keys()
    })

    x = np.arange(len(env_names))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(env_names) * 1.5), 5))
    ax.set_title(title or f"Results by Environment: {metric}")
    ax.set_xlabel("Environment")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, rotation=30, ha="right")

    for i, model_name in enumerate(model_names):
        vals = [
            per_env.get(env, {}).get(model_name, {}).get(metric, 0.0)
            for env in env_names
        ]
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=model_name)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_heatmap(
    per_model: Dict[str, Dict[str, Any]],
    metric: str = "avg_mean_reward",
    env_filter: Optional[str] = None,
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Heatmap: model (rows) × benchmark_id (cols), colored by metric.

    Args:
        per_model: DataStore metrics["per_model"]
        metric: Which metric to use as cell value
        env_filter: Only include benchmark_ids with this env prefix
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import numpy as np

    models = sorted(per_model.keys())
    all_bids: set = set()
    for bid_metrics in per_model.values():
        all_bids.update(bid_metrics.keys())
    bids = sorted(all_bids)
    if env_filter:
        bids = [b for b in bids if b.startswith(env_filter + "/")]

    if not bids or not models:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig

    # Short labels for benchmark_ids
    short_bids = [b.split("/")[1] if "/" in b else b for b in bids]

    data = np.zeros((len(models), len(bids)))
    for i, model in enumerate(models):
        for j, bid in enumerate(bids):
            val = per_model.get(model, {}).get(bid, {}).get(metric, 0.0)
            data[i, j] = val if val is not None else 0.0

    fig, ax = plt.subplots(figsize=(max(8, len(bids) * 0.6), max(4, len(models) * 0.8)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label=metric)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xticks(range(len(bids)))
    ax.set_xticklabels(short_bids, rotation=45, ha="right")
    ax.set_title(title or f"Heatmap: {metric}")

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(bids)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")

    fig.tight_layout()
    return fig


def plot_improvement_distribution(
    per_model: Dict[str, Dict[str, Any]],
    benchmark_id: str,
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Box plot of improvement (last - first episode reward) distributions.

    Args:
        per_model: DataStore metrics["per_model"]
        benchmark_id: Which env config to compare
        title: Plot title
    """
    import matplotlib.pyplot as plt

    models = []
    improvements = []
    for model_name, bid_metrics in per_model.items():
        m = bid_metrics.get(benchmark_id, {})
        val = m.get("avg_improvement", None)
        if val is not None:
            models.append(model_name)
            improvements.append(val)

    if not models:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(models, improvements, color=["green" if v >= 0 else "red" for v in improvements])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title or f"In-Context Learning Improvement: {benchmark_id}")
    ax.set_ylabel("Avg Improvement (last - first episode)")
    ax.set_xlabel("Model")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return fig


# =============================================================================
# Double-agent plots
# =============================================================================

def plot_double_agent_learning_curve(
    per_schedule: Dict[str, Any],
    schedule_key: str,
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Plot per-episode reward curve with switch point marked.

    Args:
        per_schedule: DataStore metrics["double_agent"]["per_schedule"]
        schedule_key: Which schedule to plot (e.g., "P_f→Q_b/bandits/loyal_favorite_0")
        title: Plot title

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    m = per_schedule.get(schedule_key, {})
    y = m.get("per_episode_avg_rewards", [])
    std = m.get("per_episode_std_rewards", [0] * len(y))
    switch_ep = m.get("switch_episode", 0)

    if not y:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig

    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(len(y)))

    # Color pre/post switch differently
    pre_x = [i for i in x if i < switch_ep]
    pre_y = [y[i] for i in pre_x]
    post_x = [i for i in x if i >= switch_ep]
    post_y = [y[i] for i in post_x]

    ax.plot(pre_x, pre_y, "o-", markersize=4, color="#1976D2", label="Agent 1 (pre-switch)")
    ax.plot(post_x, post_y, "s-", markersize=4, color="#D32F2F", label="Agent 2 (post-switch)")

    # Std band
    ax.fill_between(x,
                    [yi - si for yi, si in zip(y, std)],
                    [yi + si for yi, si in zip(y, std)],
                    alpha=0.1, color="gray")

    # Switch line
    ax.axvline(switch_ep - 0.5, color="black", linestyle="--", linewidth=1.5,
               label=f"Switch at ep {switch_ep}")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward")
    ax.set_title(title or f"Double-Agent Learning Curve: {schedule_key}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pre_post_switch_comparison(
    per_schedule: Dict[str, Any],
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Bar chart comparing pre-switch vs post-switch reward across all schedules.

    Args:
        per_schedule: DataStore metrics["double_agent"]["per_schedule"]
        title: Plot title

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    keys = sorted(per_schedule.keys())
    if not keys:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig

    # Short labels
    short_keys = [k.split("/")[-1] if "/" in k else k for k in keys]
    pre_rewards = [per_schedule[k].get("avg_pre_switch_reward", 0) for k in keys]
    post_rewards = [per_schedule[k].get("avg_post_switch_reward", 0) for k in keys]
    pre_std = [per_schedule[k].get("std_pre_switch_reward", 0) for k in keys]
    post_std = [per_schedule[k].get("std_post_switch_reward", 0) for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 5))
    ax.bar(x - width / 2, pre_rewards, width, yerr=pre_std, label="Pre-Switch",
           color="#1976D2", alpha=0.8, capsize=3)
    ax.bar(x + width / 2, post_rewards, width, yerr=post_std, label="Post-Switch",
           color="#D32F2F", alpha=0.8, capsize=3)

    ax.set_xlabel("Schedule")
    ax.set_ylabel("Avg Reward")
    ax.set_title(title or "Pre-Switch vs Post-Switch Reward")
    ax.set_xticks(x)
    ax.set_xticklabels(short_keys, rotation=30, ha="right")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_per_agent_comparison(
    per_agent: Dict[str, Any],
    metric: str = "avg_reward",
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Bar chart comparing agents across all schedules.

    Args:
        per_agent: DataStore metrics["double_agent"]["per_agent"]
        metric: "avg_reward", "success_rate", or "avg_turns"
        title: Plot title

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not per_agent:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig

    # Collect all agent names
    all_agents = set()
    for agents in per_agent.values():
        all_agents.update(agents.keys())
    all_agents = sorted(all_agents)

    schedule_keys = sorted(per_agent.keys())
    short_keys = [k.split("/")[-1] if "/" in k else k for k in schedule_keys]

    x = np.arange(len(schedule_keys))
    width = 0.8 / max(len(all_agents), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(schedule_keys) * 1.2), 5))
    for i, agent_name in enumerate(all_agents):
        vals = [
            per_agent.get(k, {}).get(agent_name, {}).get(metric, 0)
            for k in schedule_keys
        ]
        offset = (i - len(all_agents) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=agent_name)

    ylabel = metric.replace("_", " ").title()
    ax.set_xlabel("Schedule")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Per-Agent {ylabel}")
    ax.set_xticks(x)
    ax.set_xticklabels(short_keys, rotation=30, ha="right")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_transfer_effects(
    per_schedule: Dict[str, Any],
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Bar chart of transfer effects (post - pre reward) per schedule.

    Positive = post-switch agent benefits from pre-switch context.
    Negative = post-switch agent does worse than pre-switch.

    Args:
        per_schedule: DataStore metrics["double_agent"]["per_schedule"]
        title: Plot title

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    keys = sorted(per_schedule.keys())
    if not keys:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig

    short_keys = [k.split("/")[-1] if "/" in k else k for k in keys]
    effects = [per_schedule[k].get("avg_transfer_effect", 0) for k in keys]
    stds = [per_schedule[k].get("std_transfer_effect", 0) for k in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.0), 4))
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in effects]
    ax.bar(short_keys, effects, yerr=stds, color=colors, alpha=0.8, capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Schedule")
    ax.set_ylabel("Transfer Effect (post − pre reward)")
    ax.set_title(title or "Transfer Effects")
    plt.xticks(rotation=30, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def save_all_plots(
    data: Dict[str, Any],
    output_dir: str,
    fmt: str = "png",
):
    """Generate and save all standard plots from a DataStore output.

    Args:
        data: Output of DataStore.load(data_dir)
        output_dir: Directory to save plots
        fmt: Image format ("png", "pdf", "svg")
    """
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    per_model = data["metrics"].get("per_model", {})
    per_env = data["metrics"].get("per_env", {})
    benchmark_ids = {
        bid
        for bid_metrics in per_model.values()
        for bid in bid_metrics.keys()
    }

    plots_saved = []

    # Per-env bar chart
    if per_env:
        fig = plot_env_bar_chart(per_env)
        path = f"{output_dir}/bar_env_avg_reward.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plots_saved.append(path)
        import matplotlib.pyplot as plt
        plt.close(fig)

    # Learning curves per benchmark_id — both rewards and turns
    for bid in sorted(benchmark_ids):
        safe_bid = bid.replace("/", "__")

        # Reward curve
        fig = plot_learning_curves(per_model, bid, metric="per_episode_avg_rewards")
        path = f"{output_dir}/learning_curve_{safe_bid}.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plots_saved.append(path)
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Turns curve
        fig = plot_learning_curves(per_model, bid, metric="per_episode_avg_turns",
                                   title=f"Turns per Episode: {bid}")
        path = f"{output_dir}/turns_curve_{safe_bid}.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plots_saved.append(path)
        plt.close(fig)

    # Heatmap per env
    env_names = sorted({bid.split("/")[0] for bid in benchmark_ids})
    for env_name in env_names:
        fig = plot_heatmap(per_model, env_filter=env_name)
        path = f"{output_dir}/heatmap_{env_name}.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plots_saved.append(path)
        import matplotlib.pyplot as plt
        plt.close(fig)

    # ── Double-agent plots (only if data present) ──

    per_schedule = data["metrics"].get("double_agent", {}).get("per_schedule", {})
    per_agent = data["metrics"].get("double_agent", {}).get("per_agent", {})

    if per_schedule:
        # Pre/post switch comparison bar chart
        fig = plot_pre_post_switch_comparison(per_schedule)
        path = f"{output_dir}/bar_pre_post_switch.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plots_saved.append(path)
        plt.close(fig)

        # Transfer effects bar chart
        fig = plot_transfer_effects(per_schedule)
        path = f"{output_dir}/bar_transfer_effects.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plots_saved.append(path)
        plt.close(fig)

        # Per-schedule learning curves with switch point
        for key in sorted(per_schedule.keys()):
            safe_key = key.replace("/", "__")
            fig = plot_double_agent_learning_curve(per_schedule, key)
            path = f"{output_dir}/da_learning_curve_{safe_key}.{fmt}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plots_saved.append(path)
            plt.close(fig)

    if per_agent:
        # Per-agent reward comparison
        fig = plot_per_agent_comparison(per_agent, metric="avg_reward")
        path = f"{output_dir}/bar_per_agent_reward.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plots_saved.append(path)
        plt.close(fig)

        # Per-agent success rate
        fig = plot_per_agent_comparison(per_agent, metric="success_rate")
        path = f"{output_dir}/bar_per_agent_success.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plots_saved.append(path)
        plt.close(fig)

    return plots_saved
