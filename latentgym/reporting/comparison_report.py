"""
ComparisonReport — Compute and persist cross-model comparison metrics.

Designed for the standard 4-schedule comparison experiment:
    P_f:      finetuned model only
    Q_b:      base model only
    P_f→Q_b:  finetuned then base (transfer schedule)
    Q_b→P_f:  base then finetuned (transfer schedule)

Owns:
    metrics/double_agent/comparisons.json

Usage:
    report = ComparisonReport(
        results,
        finetuned_key="P_f",
        base_key="Q_b",
        switch_episode=5,
    )
    report.save("results/run_001")

    # Or layer onto existing DataStore:
    report.save_to(store)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from latentgym.eval.results import BenchmarkResults
from latentgym.eval.single_agent.metrics import compute_comparison_metrics
from .data_store import DataStore

logger = logging.getLogger(__name__)


class ComparisonReport:
    """Computes and writes cross-model comparison metrics.

    Metrics computed per benchmark_id:
        - overall_improvement: finetuned final − base final
        - initial_prior_improvement: finetuned initial − base initial
        - icl_difference: finetuned ICL − base ICL
        - transfer_f_to_b: P_f→Q_b trajectory reward − Q_b-only trajectory reward
        - transfer_b_to_f: Q_b→P_f trajectory reward − P_f-only trajectory reward
        - turn_efficiency_difference: base turns − finetuned turns
        - per_config_summary: avg_mean_reward, avg_improvement, avg_success_rate per schedule
    """

    def __init__(
        self,
        results: BenchmarkResults,
        finetuned_key: str = "P_f",
        base_key: str = "Q_b",
        f_then_b_key: str = "P_f->Q_b",
        b_then_f_key: str = "Q_b->P_f",
        switch_episode: int = 5,
    ):
        self.results = results
        self.finetuned_key = finetuned_key
        self.base_key = base_key
        self.f_then_b_key = f_then_b_key
        self.b_then_f_key = b_then_f_key
        self.switch_episode = switch_episode
        self._comparisons: Dict[str, Any] = {}
        self._computed = False

    def compute(self):
        """Compute comparison metrics across all shared benchmark_ids. Idempotent."""
        if self._computed:
            return

        all_bids = set()
        for model_results in self.results.results.values():
            all_bids.update(model_results.keys())

        keys = [
            self.finetuned_key,
            self.base_key,
            self.f_then_b_key,
            self.b_then_f_key,
        ]

        for bid in sorted(all_bids):
            bid_results = {}
            for key in keys:
                trajs = self.results.get(key, bid)
                if trajs:
                    bid_results[key] = trajs

            if len(bid_results) < 2:
                continue

            comp = compute_comparison_metrics(
                bid_results,
                finetuned_key=self.finetuned_key,
                base_key=self.base_key,
                f_then_b_key=self.f_then_b_key,
                b_then_f_key=self.b_then_f_key,
            )

            # Serialize everything — per_config contains full 20-key metric
            # dicts per schedule, we keep them all (no lossy summarization)
            entry = {k: v for k, v in comp.items() if k != "per_config"}
            entry["per_config"] = {}
            for name, m in comp.get("per_config", {}).items():
                # Keep all scalar metrics; convert lists to JSON-safe form
                entry["per_config"][name] = {
                    k: v for k, v in m.items()
                    if isinstance(v, (int, float, str, bool, list))
                }
            self._comparisons[bid] = entry

        self._computed = True

    def save(self, output_dir: str, run_name: str = "", extra_metadata: Optional[Dict[str, Any]] = None):
        """Compute and write to output_dir."""
        self.compute()
        store = DataStore(output_dir)
        self.save_to(store)

        metadata = {
            "run_name": run_name or f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "report_type": "comparison",
            "schedule_keys": {
                "finetuned": self.finetuned_key,
                "base": self.base_key,
                "finetuned_then_base": self.f_then_b_key,
                "base_then_finetuned": self.b_then_f_key,
            },
            "switch_episode": self.switch_episode,
            "n_benchmark_ids": len(self._comparisons),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        store.write_metadata(metadata)

        logger.info(f"ComparisonReport saved to {output_dir}")
        logger.info(f"  {len(self._comparisons)} benchmark configs compared")

    def save_to(self, store: DataStore):
        """Write comparison metrics to an existing DataStore."""
        self.compute()

        if not self._comparisons:
            logger.warning("No comparison data — need at least 2 schedule keys with shared benchmark_ids")
            return

        output = {
            "switch_episode": self.switch_episode,
            "schedule_keys": {
                "finetuned": self.finetuned_key,
                "base": self.base_key,
                "finetuned_then_base": self.f_then_b_key,
                "base_then_finetuned": self.b_then_f_key,
            },
            "per_benchmark_id": self._comparisons,
        }
        store.write_json(output, "metrics/double_agent/comparisons.json")

    # -- Accessors --

    @property
    def comparisons(self) -> Dict[str, Any]:
        self.compute()
        return self._comparisons
