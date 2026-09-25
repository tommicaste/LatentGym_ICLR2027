"""
Optional W&B integration for benchmark evaluation.

Logs metrics, learning curves, comparison charts, and trajectory artifacts.
Fully optional — eval pipeline works without wandb installed.

Usage:
    # With wandb:
    tracker = create_tracker(WandbConfig(project="benchmark", run_name="gpt4o-bandits"))
    tracker.init(run_config={...})
    tracker.log_trajectory("gpt4o", "bandits/loyal_favorite_0/...", result)
    tracker.log_single_agent_summary({"gpt4o": metrics})
    tracker.log_double_agent_summary(double_metrics, switch_episode=5)
    tracker.save_trajectories_artifact(results)
    tracker.finish()

    # Without wandb (no-op):
    tracker = create_tracker(None)
    tracker.log_trajectory(...)  # does nothing
"""
from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .types import EpisodeOutcome, TrajectoryResult
from .results import BenchmarkResults


@dataclass
class WandbConfig:
    """Configuration for W&B tracking."""
    project: str = "benchmark-eval"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    save_trajectories: bool = True
    log_per_episode: bool = True
    log_interval: int = 1
    offline: bool = False


class WandbTracker:
    """W&B tracker for benchmark evaluation.

    Logs:
    - Per-episode metrics (rewards, turns, agent, outcome type)
    - Per-trajectory metrics (cumulative, terminal, improvement, etc.)
    - Single-agent summary (per-config aggregates, learning curves)
    - Double-agent summary (pre/post switch, transfer effects)
    - Comparison charts (bar charts, learning curves overlay)
    - Summary tables
    - Full trajectory artifacts
    """

    def __init__(self, config: WandbConfig):
        if not WANDB_AVAILABLE:
            raise ImportError("wandb not installed. Install with: pip install wandb")
        self.config = config
        self.run = None
        self._traj_count = 0
        self._step = 0

    def init(self, run_config: Optional[Dict[str, Any]] = None) -> Any:
        """Initialize W&B run."""
        mode = "offline" if self.config.offline else "online"
        self.run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=self.config.run_name,
            tags=self.config.tags,
            group=self.config.group,
            mode=mode,
            config=run_config or {},
        )
        wandb.define_metric("episode/*", step_metric="global_step")
        wandb.define_metric("trajectory/*", step_metric="traj_idx")
        logger.info(f"W&B run initialized: {self.run.url}")
        return self.run

    # ── Per-episode logging ──

    def log_episode(
        self,
        model_name: str,
        benchmark_id: str,
        trajectory_idx: int,
        outcome: EpisodeOutcome,
    ):
        """Log a single episode outcome."""
        if not self.config.log_per_episode:
            return

        self._step += 1
        prefix = f"episode/{model_name}/{benchmark_id}"
        wandb.log({
            "global_step": self._step,
            f"{prefix}/reward": outcome.reward,
            f"{prefix}/turns": outcome.turns,
            f"{prefix}/success": int(outcome.success),
            f"{prefix}/agent": outcome.agent_name,
            f"{prefix}/outcome_type": outcome.outcome_type.value,
            f"{prefix}/turn_efficiency": outcome.turn_efficiency,
            f"{prefix}/episode_idx": outcome.episode_idx,
            f"{prefix}/trajectory_idx": trajectory_idx,
        })

    # ── Per-trajectory logging ──

    def log_trajectory(
        self,
        model_name: str,
        benchmark_id: str,
        trajectory_idx: int,
        result: TrajectoryResult,
    ):
        """Log a complete trajectory result."""
        self._traj_count += 1
        if self._traj_count % self.config.log_interval != 0:
            return

        prefix = f"trajectory/{model_name}/{benchmark_id}"
        metrics = {
            "traj_idx": trajectory_idx,
            f"{prefix}/cumulative_reward": result.cumulative_reward,
            f"{prefix}/terminal_reward": result.terminal_reward,
            f"{prefix}/initial_reward": result.initial_reward,
            f"{prefix}/improvement": result.improvement,
            f"{prefix}/mean_reward": result.mean_reward,
            f"{prefix}/success_rate": result.success_rate,
            f"{prefix}/total_turns": result.total_turns,
            f"{prefix}/mean_turns": result.mean_turns,
            f"{prefix}/num_episodes": result.num_episodes,
        }

        # Per-episode breakdown
        for i, (reward, turns) in enumerate(zip(result.episode_rewards, result.episode_turns)):
            metrics[f"{prefix}/ep_{i}_reward"] = reward
            metrics[f"{prefix}/ep_{i}_turns"] = turns

        wandb.log(metrics)

    # ── Single-agent summary ──

    def log_single_agent_summary(
        self,
        per_config_metrics: Dict[str, Dict[str, Any]],
    ):
        """Log aggregated single-agent metrics for all configs.

        Args:
            per_config_metrics: config_name → metrics dict
                (from compute_single_agent_metrics)
        """
        # Scalar metrics
        for config_name, m in per_config_metrics.items():
            for metric_name, value in m.items():
                if isinstance(value, (int, float)):
                    wandb.log({f"summary/{config_name}/{metric_name}": value})

        # Bar chart: avg reward per config
        data = [
            [name, m.get("avg_trajectory_reward", 0), m.get("std_trajectory_reward", 0)]
            for name, m in per_config_metrics.items()
        ]
        table = wandb.Table(data=data, columns=["config", "avg_reward", "std_reward"])
        wandb.log({
            "charts/config_comparison": wandb.plot.bar(
                table, "config", "avg_reward",
                title="Average Trajectory Reward by Config"
            )
        })

        # Learning curves overlay
        self._log_learning_curves(per_config_metrics)

        # Summary table
        self._log_summary_table(per_config_metrics)

    def _log_learning_curves(self, per_config_metrics: Dict[str, Dict[str, Any]]):
        """Overlay learning curves for all configs."""
        config_names = []
        xs = []
        ys = []

        for name, m in per_config_metrics.items():
            per_ep = m.get("per_episode_avg_rewards", [])
            if per_ep:
                config_names.append(name)
                xs.append(list(range(len(per_ep))))
                ys.append(per_ep)

        if xs and ys:
            wandb.log({
                "charts/learning_curves": wandb.plot.line_series(
                    xs=xs, ys=ys, keys=config_names,
                    title="Learning Curves (Avg Episode Reward)",
                    xname="Episode",
                )
            })

    def _log_summary_table(self, per_config_metrics: Dict[str, Dict[str, Any]]):
        """Log a summary table."""
        columns = [
            "config", "n_trajectories",
            "avg_reward", "std_reward",
            "avg_initial", "avg_final", "avg_improvement",
            "learning_slope", "avg_success_rate", "avg_turns_per_ep",
        ]
        data = []
        for name, m in per_config_metrics.items():
            data.append([
                name,
                m.get("n_trajectories", 0),
                round(m.get("avg_trajectory_reward", 0), 4),
                round(m.get("std_trajectory_reward", 0), 4),
                round(m.get("avg_initial_reward", 0), 4),
                round(m.get("avg_final_reward", 0), 4),
                round(m.get("avg_improvement", 0), 4),
                round(m.get("learning_slope", 0), 4),
                round(m.get("avg_success_rate", 0), 4),
                round(m.get("avg_mean_turns_per_episode", 0), 2),
            ])

        table = wandb.Table(data=data, columns=columns)
        wandb.log({"summary/results_table": table})

    # ── Double-agent summary ──

    def log_double_agent_summary(
        self,
        metrics: Dict[str, Any],
    ):
        """Log double-agent metrics.

        Args:
            metrics: From compute_double_agent_metrics
        """
        # Scalar metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                wandb.log({f"double_agent/{k}": v})

        # Per-agent breakdown
        per_agent = metrics.get("per_agent", {})
        if per_agent:
            data = [
                [name, d["avg_reward"], d["std_reward"], d["success_rate"], d["n_episodes"]]
                for name, d in per_agent.items()
            ]
            table = wandb.Table(
                data=data,
                columns=["agent", "avg_reward", "std_reward", "success_rate", "n_episodes"]
            )
            wandb.log({
                "charts/per_agent_comparison": wandb.plot.bar(
                    table, "agent", "avg_reward",
                    title="Per-Agent Average Reward"
                )
            })

        # Per-episode curve with switch point
        per_ep = metrics.get("per_episode_avg_rewards", [])
        if per_ep:
            switch = metrics.get("switch_episode", 0)
            xs = [list(range(len(per_ep)))]
            ys = [per_ep]
            wandb.log({
                "charts/double_agent_episode_curve": wandb.plot.line_series(
                    xs=xs, ys=ys, keys=["avg_reward"],
                    title=f"Episode Rewards (switch at ep {switch})",
                    xname="Episode",
                )
            })

    # ── Comparison logging ──

    def log_comparison(
        self,
        comparison_metrics: Dict[str, Any],
    ):
        """Log comparison metrics (finetuned vs base).

        Args:
            comparison_metrics: From compute_comparison_metrics
        """
        for k, v in comparison_metrics.items():
            if k == "per_config":
                continue
            if isinstance(v, (int, float)):
                wandb.log({f"comparison/{k}": v})

        per_config = comparison_metrics.get("per_config", {})
        if per_config:
            self.log_single_agent_summary(per_config)

    # ── Artifacts ──

    def save_trajectories_artifact(
        self,
        results: BenchmarkResults,
        name: str = "eval_trajectories",
    ):
        """Save full trajectory data as W&B artifact."""
        if not self.config.save_trajectories:
            return

        artifact = wandb.Artifact(name, type="eval_results")

        with tempfile.TemporaryDirectory() as tmpdir:
            for model_name, env_results in results.results.items():
                for benchmark_id, trajectories in env_results.items():
                    safe_bid = benchmark_id.replace("/", "_")
                    filepath = Path(tmpdir) / f"{model_name}_{safe_bid}.json"
                    data = [t.to_dict() for t in trajectories]
                    with open(filepath, "w") as f:
                        json.dump(data, f, indent=2)
                    artifact.add_file(str(filepath))

            wandb.log_artifact(artifact)

        logger.info(f"Saved trajectories artifact: {name}")

    # ── Lifecycle ──

    def finish(self):
        """Finish the W&B run."""
        if self.run:
            self.run.finish()
            logger.info("W&B run finished")


class DummyTracker:
    """No-op tracker when wandb is disabled or unavailable."""

    def init(self, *a, **kw): return None
    def log_episode(self, *a, **kw): pass
    def log_trajectory(self, *a, **kw): pass
    def log_single_agent_summary(self, *a, **kw): pass
    def log_double_agent_summary(self, *a, **kw): pass
    def log_comparison(self, *a, **kw): pass
    def save_trajectories_artifact(self, *a, **kw): pass
    def finish(self): pass


def create_tracker(config: Optional[WandbConfig] = None) -> WandbTracker:
    """Create a tracker or dummy.

    Args:
        config: WandbConfig to enable tracking, None to disable.

    Returns:
        WandbTracker if config provided and wandb available, DummyTracker otherwise.
    """
    if config is None:
        return DummyTracker()
    if not WANDB_AVAILABLE:
        logger.warning("wandb not installed, using dummy tracker")
        return DummyTracker()
    return WandbTracker(config)
