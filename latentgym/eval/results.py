"""
BenchmarkResults — Structured storage for all evaluation results.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import TrajectoryResult


@dataclass
class BenchmarkResults:
    """Stores all results in a structured, serializable format.

    Structure: model_name → benchmark_id → List[TrajectoryResult]
    """
    results: Dict[str, Dict[str, List[TrajectoryResult]]] = field(default_factory=dict)

    def add(self, model_name: str, benchmark_id: str, trajectories: List[TrajectoryResult]):
        """Add trajectories for a (model, env) pair."""
        self.results.setdefault(model_name, {})
        self.results[model_name].setdefault(benchmark_id, [])
        self.results[model_name][benchmark_id].extend(trajectories)

    def get(self, model_name: str, benchmark_id: str) -> List[TrajectoryResult]:
        """Get trajectories for a (model, env) pair."""
        return self.results.get(model_name, {}).get(benchmark_id, [])

    def slice_by_model(self, model_name: str) -> BenchmarkResults:
        """Get all results for one model."""
        sliced = BenchmarkResults()
        if model_name in self.results:
            sliced.results[model_name] = self.results[model_name]
        return sliced

    def slice_by_env(self, env_name: str) -> BenchmarkResults:
        """Get all results where benchmark_id starts with env_name."""
        sliced = BenchmarkResults()
        for model, env_results in self.results.items():
            for bid, trajs in env_results.items():
                if bid.startswith(env_name + "/"):
                    sliced.results.setdefault(model, {})[bid] = trajs
        return sliced

    @property
    def model_names(self) -> List[str]:
        return list(self.results.keys())

    @property
    def benchmark_ids(self) -> List[str]:
        ids = set()
        for env_results in self.results.values():
            ids.update(env_results.keys())
        return sorted(ids)

    def save(self, path: str):
        """Save to JSON."""
        data = {}
        for model, env_results in self.results.items():
            data[model] = {}
            for bid, trajs in env_results.items():
                data[model][bid] = [t.to_dict() for t in trajs]

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> BenchmarkResults:
        """Load from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        br = cls()
        for model, env_results in data.items():
            for bid, traj_dicts in env_results.items():
                trajs = [TrajectoryResult.from_dict(td) for td in traj_dicts]
                br.add(model, bid, trajs)
        return br
