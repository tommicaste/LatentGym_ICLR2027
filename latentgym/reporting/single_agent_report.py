"""
SingleAgentReport — Compute and persist all single-agent metrics.

Owns:
    metrics/per_model.json
    metrics/per_env.json
    metrics/per_latent_complexity.json
    metrics/leaderboard.json
    metrics/detailed/exploration_exploitation.json
    tables/main_results.csv
    tables/leaderboard.csv

Usage:
    report = SingleAgentReport(results)
    report.save("results/run_001")          # writes to DataStore dir

    # Or with an existing DataStore:
    report.save_to(store)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from latentgym.eval.results import BenchmarkResults
from latentgym.eval.single_agent.metrics import (
    compute_single_agent_metrics,
    compute_detailed_metrics,
)
from .data_store import DataStore

logger = logging.getLogger(__name__)


class SingleAgentReport:
    """Computes and writes all single-agent metrics and tables.

    Metrics computed:
        - per_model: model → benchmark_id → full metric dict
        - per_env: env → model → averaged metrics across latents
        - per_latent_complexity: complexity → model → averaged metrics
        - leaderboard: ranked by avg_mean_reward
        - detailed: exploration/exploitation phase split per config
    """

    def __init__(self, results: BenchmarkResults):
        self.results = results
        self._per_model: Dict[str, Dict[str, Any]] = {}
        self._per_env: Dict[str, Any] = {}
        self._per_complexity: Dict[str, Any] = {}
        self._leaderboard: List[Dict[str, Any]] = []
        self._detailed: Dict[str, Any] = {}
        self._computed = False

    def compute(self):
        """Compute all metrics. Idempotent — safe to call multiple times."""
        if self._computed:
            return

        per_env_acc: Dict[str, Dict[str, Any]] = defaultdict(dict)
        per_model: Dict[str, Dict[str, Any]] = defaultdict(dict)
        per_complexity_acc: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: defaultdict(list)
        )

        for model_name, env_results in self.results.results.items():
            for bid, trajectories in env_results.items():
                if not trajectories:
                    continue

                m = compute_single_agent_metrics(trajectories)
                per_model[model_name][bid] = m

                # Accumulate per-env averages
                env_name = bid.split("/")[0]
                per_env_acc[env_name].setdefault(model_name, {})
                for key, val in m.items():
                    if isinstance(val, (int, float)):
                        bucket = per_env_acc[env_name][model_name]
                        if key not in bucket:
                            bucket[key] = {"sum": 0.0, "count": 0}
                        bucket[key]["sum"] += val
                        bucket[key]["count"] += 1

                # Accumulate per-complexity
                complexity = self._infer_complexity(trajectories)
                if complexity:
                    for key, val in m.items():
                        if isinstance(val, (int, float)):
                            per_complexity_acc[complexity][model_name].append(
                                (key, val)
                            )

        # Flatten per_env
        for env_name, model_data in per_env_acc.items():
            self._per_env[env_name] = {}
            for model_name, metrics_acc in model_data.items():
                self._per_env[env_name][model_name] = {
                    k: v["sum"] / v["count"] for k, v in metrics_acc.items()
                }

        # Flatten per_complexity
        for complexity, model_data in per_complexity_acc.items():
            self._per_complexity[complexity] = {}
            for model_name, kv_list in model_data.items():
                acc: Dict[str, list] = defaultdict(list)
                for k, v in kv_list:
                    acc[k].append(v)
                self._per_complexity[complexity][model_name] = {
                    k: sum(v) / len(v) for k, v in acc.items()
                }

        self._per_model = dict(per_model)
        self._leaderboard = self._build_leaderboard(self._per_model)

        # Detailed exploration/exploitation metrics
        for model_name, env_results in self.results.results.items():
            for bid, trajectories in env_results.items():
                if not trajectories:
                    continue
                key = f"{model_name}/{bid}"
                detailed = compute_detailed_metrics({key: trajectories})
                if key in detailed:
                    self._detailed[key] = detailed[key]

        self._computed = True

    def save(self, output_dir: str, run_name: str = "", extra_metadata: Optional[Dict[str, Any]] = None):
        """Compute metrics and write everything to output_dir.

        Creates a DataStore at output_dir and writes trajectories, metrics,
        tables, and metadata.
        """
        self.compute()
        store = DataStore(output_dir)
        self.save_to(store)

        # Write trajectories
        store.write_trajectories(self.results)

        # Write metadata
        metadata = {
            "run_name": run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "report_type": "single_agent",
            "models": self.results.model_names,
            "benchmark_ids": self.results.benchmark_ids,
            "env_names": self._extract_env_names(),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        store.write_metadata(metadata)

        logger.info(f"SingleAgentReport saved to {output_dir}")
        logger.info(
            f"  {len(self.results.model_names)} models × "
            f"{len(self.results.benchmark_ids)} configs"
        )

    def save_to(self, store: DataStore):
        """Write metrics and tables to an existing DataStore."""
        self.compute()

        # Metrics
        store.write_json(self._per_model, "metrics/per_model.json")
        store.write_json(self._per_env, "metrics/per_env.json")
        store.write_json(self._per_complexity, "metrics/per_latent_complexity.json")
        store.write_json(self._leaderboard, "metrics/leaderboard.json")
        if self._detailed:
            store.write_json(
                self._detailed, "metrics/detailed/exploration_exploitation.json"
            )

        # Tables
        self._write_main_results_csv(store)
        self._write_leaderboard_csv(store)

    # -- Accessors (for programmatic use without writing to disk) --

    @property
    def per_model(self) -> Dict[str, Dict[str, Any]]:
        self.compute()
        return self._per_model

    @property
    def per_env(self) -> Dict[str, Any]:
        self.compute()
        return self._per_env

    @property
    def leaderboard(self) -> List[Dict[str, Any]]:
        self.compute()
        return self._leaderboard

    @property
    def detailed(self) -> Dict[str, Any]:
        self.compute()
        return self._detailed

    # -- Internal helpers --

    def _build_leaderboard(self, per_model: Dict) -> List[Dict[str, Any]]:
        scores = {}
        for model_name, bid_metrics in per_model.items():
            all_rewards = [
                m.get("avg_mean_reward", 0.0)
                for m in bid_metrics.values()
                if m
            ]
            scores[model_name] = (
                sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
            )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"rank": i + 1, "model": name, "avg_reward": round(score, 4)}
            for i, (name, score) in enumerate(ranked)
        ]

    def _write_main_results_csv(self, store: DataStore):
        rows = []
        for model_name, bid_metrics in self._per_model.items():
            for bid, m in bid_metrics.items():
                rows.append({
                    "model": model_name,
                    "benchmark_id": bid,
                    "avg_mean_reward": round(m.get("avg_mean_reward", 0), 4),
                    "avg_improvement": round(m.get("avg_improvement", 0), 4),
                    "avg_success_rate": round(m.get("avg_success_rate", 0), 4),
                    "avg_initial_reward": round(m.get("avg_initial_reward", 0), 4),
                    "avg_final_reward": round(m.get("avg_final_reward", 0), 4),
                    "learning_slope": round(m.get("learning_slope", 0), 4),
                    "avg_total_turns": round(m.get("avg_total_turns", 0), 2),
                    "n_trajectories": m.get("n_trajectories", 0),
                })
        if rows:
            store.write_csv(rows, list(rows[0].keys()), "tables/main_results.csv")

    def _write_leaderboard_csv(self, store: DataStore):
        if self._leaderboard:
            store.write_csv(
                self._leaderboard,
                ["rank", "model", "avg_reward"],
                "tables/leaderboard.csv",
            )

    def _extract_env_names(self) -> List[str]:
        envs = set()
        for bid in self.results.benchmark_ids:
            envs.add(bid.split("/")[0])
        return sorted(envs)

    @staticmethod
    def _infer_complexity(trajectories) -> str:
        if trajectories and trajectories[0].metadata.get("latent_complexity"):
            return trajectories[0].metadata["latent_complexity"]
        return ""
