"""
DoubleAgentReport — Compute and persist all double-agent / multi-agent metrics.

Owns:
    metrics/double_agent/per_schedule.json
    metrics/double_agent/per_agent.json
    tables/double_agent_summary.csv
    tables/per_agent_breakdown.csv

Usage:
    report = DoubleAgentReport(results, switch_episode=5)
    report.save("results/run_001")

    # Or layer onto an existing DataStore (e.g., after SingleAgentReport):
    report.save_to(store)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from latentgym.eval.results import BenchmarkResults
from latentgym.eval.double_agent.metrics import compute_double_agent_metrics
from latentgym.eval.single_agent.metrics import compute_per_agent_metrics
from .data_store import DataStore

logger = logging.getLogger(__name__)


class DoubleAgentReport:
    """Computes and writes all double-agent metrics and tables.

    Requires that trajectories have `agent_assignments` populated
    (set automatically by ScheduledRunner / DoubleAgentRunner).

    Metrics computed:
        - per_schedule: pre/post switch reward, transfer effect, adaptation speed
        - per_agent: per-agent reward, turns, success rate breakdown
        - per-episode curves with switch point marked
    """

    def __init__(
        self,
        results: BenchmarkResults,
        switch_episode: Optional[int] = None,
    ):
        """
        Args:
            results: BenchmarkResults containing double-agent trajectories.
            switch_episode: Episode where the agent switch occurs.
                If None, auto-detected from agent_assignments.
        """
        self.results = results
        self._explicit_switch = switch_episode
        self._per_schedule: Dict[str, Any] = {}
        self._per_agent: Dict[str, Any] = {}
        self._computed = False

    def compute(self):
        """Compute all double-agent metrics. Idempotent."""
        if self._computed:
            return

        for model_name, env_results in self.results.results.items():
            for bid, trajectories in env_results.items():
                if not trajectories:
                    continue

                switch_ep = self._resolve_switch_episode(trajectories)
                if switch_ep is None:
                    continue  # Not a double-agent trajectory

                key = f"{model_name}/{bid}"

                # Core double-agent metrics
                self._per_schedule[key] = compute_double_agent_metrics(
                    trajectories, switch_ep
                )

                # Per-agent breakdown
                agent_metrics = compute_per_agent_metrics({key: trajectories})
                if key in agent_metrics:
                    self._per_agent[key] = agent_metrics[key]

        self._computed = True

    def save(self, output_dir: str, run_name: str = "", extra_metadata: Optional[Dict[str, Any]] = None):
        """Compute and write to output_dir. Creates DataStore if needed."""
        self.compute()
        store = DataStore(output_dir)
        self.save_to(store)

        # Write trajectories
        store.write_trajectories(self.results)

        # Write metadata
        metadata = {
            "run_name": run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "report_type": "double_agent",
            "models": self.results.model_names,
            "benchmark_ids": self.results.benchmark_ids,
            "switch_episode": self._explicit_switch,
            "n_schedules": len(self._per_schedule),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        store.write_metadata(metadata)

        logger.info(f"DoubleAgentReport saved to {output_dir}")
        logger.info(f"  {len(self._per_schedule)} schedule configs recorded")

    def save_to(self, store: DataStore):
        """Write double-agent metrics and tables to an existing DataStore."""
        self.compute()

        if not self._per_schedule:
            logger.warning("No double-agent trajectories found — nothing to write")
            return

        # Metrics
        store.write_json(self._per_schedule, "metrics/double_agent/per_schedule.json")
        store.write_json(self._per_agent, "metrics/double_agent/per_agent.json")

        # Tables
        self._write_summary_csv(store)
        self._write_per_agent_csv(store)

    # -- Accessors --

    @property
    def per_schedule(self) -> Dict[str, Any]:
        self.compute()
        return self._per_schedule

    @property
    def per_agent(self) -> Dict[str, Any]:
        self.compute()
        return self._per_agent

    # -- Internal helpers --

    def _resolve_switch_episode(self, trajectories) -> Optional[int]:
        """Determine switch episode for a set of trajectories.

        Uses explicit switch_episode if provided, otherwise infers
        from agent_assignments.
        """
        if self._explicit_switch is not None:
            # Verify this is actually a double-agent trajectory
            for t in trajectories:
                if t.agent_assignments and len(set(t.agent_assignments)) > 1:
                    return self._explicit_switch
            return None

        # Auto-detect from agent_assignments
        for t in trajectories:
            if not t.agent_assignments:
                continue
            unique_agents = set(t.agent_assignments)
            if len(unique_agents) <= 1:
                continue
            first_agent = t.agent_assignments[0]
            for i, a in enumerate(t.agent_assignments):
                if a != first_agent:
                    return i
        return None

    def _write_summary_csv(self, store: DataStore):
        rows = []
        for key, m in sorted(self._per_schedule.items()):
            rows.append({
                "schedule": key,
                "switch_episode": m.get("switch_episode", 0),
                "n_trajectories": m.get("n_trajectories", 0),
                "n_episodes": m.get("n_episodes", 0),
                "avg_pre_switch_reward": round(m.get("avg_pre_switch_reward", 0), 4),
                "std_pre_switch_reward": round(m.get("std_pre_switch_reward", 0), 4),
                "avg_post_switch_reward": round(m.get("avg_post_switch_reward", 0), 4),
                "std_post_switch_reward": round(m.get("std_post_switch_reward", 0), 4),
                "avg_transfer_effect": round(m.get("avg_transfer_effect", 0), 4),
                "std_transfer_effect": round(m.get("std_transfer_effect", 0), 4),
                "avg_adaptation_speed": round(m.get("avg_adaptation_speed", 0), 4),
            })
        if rows:
            store.write_csv(
                rows, list(rows[0].keys()), "tables/double_agent_summary.csv"
            )

    def _write_per_agent_csv(self, store: DataStore):
        rows = []
        for key, agents in sorted(self._per_agent.items()):
            for agent_name, m in sorted(agents.items()):
                rows.append({
                    "schedule": key,
                    "agent": agent_name,
                    "avg_reward": round(m.get("avg_reward", 0), 4),
                    "std_reward": round(m.get("std_reward", 0), 4),
                    "avg_turns": round(m.get("avg_turns", 0), 2),
                    "success_rate": round(m.get("success_rate", 0), 4),
                    "n_episodes": m.get("n_episodes", 0),
                })
        if rows:
            store.write_csv(
                rows, list(rows[0].keys()), "tables/per_agent_breakdown.csv"
            )
