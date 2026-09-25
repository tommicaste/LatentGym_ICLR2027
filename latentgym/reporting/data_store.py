"""
DataStore — Thin I/O layer for reading and writing benchmark results to disk.

DataStore handles ONLY serialization: writing JSONs, CSVs, trajectories,
and loading them back. It does NOT compute metrics — that is the job of
the Report classes (SingleAgentReport, DoubleAgentReport, ComparisonReport).

Output structure (fully populated by all three report types):
    output_dir/
        metadata.json
        trajectories/
            <model_name>/<benchmark_id>/traj_000.json ...
        metrics/
            per_env.json
            per_model.json
            per_latent_complexity.json
            leaderboard.json
            detailed/
                exploration_exploitation.json
            double_agent/
                per_schedule.json
                per_agent.json
                comparisons.json
        tables/
            main_results.csv
            leaderboard.csv
            double_agent_summary.csv
            per_agent_breakdown.csv
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from latentgym.eval.results import BenchmarkResults

logger = logging.getLogger(__name__)


class DataStore:
    """Thin I/O layer — write/read JSONs, CSVs, and trajectory files.

    Does NOT compute metrics. Use the Report classes for that.

    Usage (write):
        store = DataStore("results/run_001")
        store.write_trajectories(results)
        store.write_json(metrics, "metrics/per_model.json")
        store.write_csv(rows, fieldnames, "tables/main_results.csv")
        store.write_metadata({...})

    Usage (read):
        data = DataStore.load("results/run_001")
        data["metrics"]["per_model"]
        data["metrics"]["double_agent"]["per_schedule"]
    """

    def __init__(self, output_dir: str):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, data: Any, relative_path: str):
        """Write data as JSON to output_dir/relative_path."""
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def write_csv(self, rows: List[Dict[str, Any]], fieldnames: List[str], relative_path: str):
        """Write rows as CSV to output_dir/relative_path."""
        if not rows:
            return
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def write_trajectories(self, results: BenchmarkResults):
        """Write each trajectory to its own JSON file under trajectories/."""
        traj_dir = self.root / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        for model_name, env_results in results.results.items():
            for bid, trajectories in env_results.items():
                safe_model = model_name.replace("/", "__").replace(":", "_")
                safe_bid = bid.replace("/", "__")
                dest = traj_dir / safe_model / safe_bid
                dest.mkdir(parents=True, exist_ok=True)
                for i, traj in enumerate(trajectories):
                    path = dest / f"traj_{i:04d}.json"
                    self.write_json(traj.to_dict(), str(path.relative_to(self.root)))

    def write_metadata(self, metadata: Dict[str, Any]):
        """Write metadata.json at root."""
        self.write_json(metadata, "metadata.json")

    @staticmethod
    def load(data_dir: str) -> Dict[str, Any]:
        """Load all data from a DataStore output directory.

        Returns:
            Dict with keys: metadata, metrics, trajectory_index.
            Missing files return empty dicts — safe to call on partial outputs.
        """
        root = Path(data_dir)

        def _load_json(path: Path) -> Any:
            try:
                if path.exists():
                    with open(path) as f:
                        return json.load(f)
            except (PermissionError, OSError):
                pass
            return {}

        metadata = _load_json(root / "metadata.json")
        metrics = {
            "per_env": _load_json(root / "metrics" / "per_env.json"),
            "per_model": _load_json(root / "metrics" / "per_model.json"),
            "per_latent_complexity": _load_json(root / "metrics" / "per_latent_complexity.json"),
            "leaderboard": _load_json(root / "metrics" / "leaderboard.json"),
            "detailed": {
                "exploration_exploitation": _load_json(
                    root / "metrics" / "detailed" / "exploration_exploitation.json"
                ),
            },
            "double_agent": {
                "per_schedule": _load_json(
                    root / "metrics" / "double_agent" / "per_schedule.json"
                ),
                "per_agent": _load_json(
                    root / "metrics" / "double_agent" / "per_agent.json"
                ),
                "comparisons": _load_json(
                    root / "metrics" / "double_agent" / "comparisons.json"
                ),
            },
        }

        # Build trajectory index (lazy — don't load trajectory contents)
        # Keys use the original model name (from the trajectory JSON or reverse-sanitized
        # from directory name) so they match the model names in metrics.
        traj_index: Dict[str, Dict[str, List[str]]] = {}
        traj_dir = root / "trajectories"
        if traj_dir.exists():
            for model_dir in sorted(traj_dir.rglob("traj_0000.json")):
                # Found a trajectory file — extract model_name from JSON
                bid_dir = model_dir.parent
                try:
                    with open(model_dir) as f:
                        traj_data = json.load(f)
                    model_name = traj_data.get("model_name", bid_dir.parent.name)
                except (json.JSONDecodeError, IOError):
                    model_name = bid_dir.parent.name
                bid = traj_data.get("benchmark_id", bid_dir.name.replace("__", "/"))
                files = sorted(str(p) for p in bid_dir.glob("traj_*.json"))
                if model_name not in traj_index:
                    traj_index[model_name] = {}
                traj_index[model_name][bid] = files

        return {
            "metadata": metadata,
            "metrics": metrics,
            "trajectory_index": traj_index,
        }

    @staticmethod
    def load_trajectory(path: str) -> Dict[str, Any]:
        """Load a single trajectory JSON file."""
        with open(path) as f:
            return json.load(f)
