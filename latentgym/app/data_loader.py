"""
DataLoader for the Streamlit app.

Loads and caches benchmark results from a DataStore output directory.
All functions work with the DataStore.load() output format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st


@st.cache_data(ttl=60)
def load_data(data_dir: str) -> Dict[str, Any]:
    """Load all data from a DataStore output directory.

    Cached by Streamlit (TTL=60s so the app refreshes when results are added).
    """
    from latentgym.reporting.data_store import DataStore
    return DataStore.load(data_dir)


@st.cache_data(ttl=300)
def load_trajectory(path: str) -> Dict[str, Any]:
    """Load a single trajectory file."""
    with open(path) as f:
        return json.load(f)


def get_env_names(data: Dict[str, Any]) -> List[str]:
    """Extract unique env names from benchmark_ids."""
    benchmark_ids = []
    for bid_metrics in data["metrics"].get("per_model", {}).values():
        benchmark_ids.extend(bid_metrics.keys())
    return sorted({bid.split("/")[0] for bid in benchmark_ids})


def get_model_names(data: Dict[str, Any]) -> List[str]:
    return sorted(data["metrics"].get("per_model", {}).keys())


def get_benchmark_ids(data: Dict[str, Any], env_filter: Optional[str] = None) -> List[str]:
    bids: set = set()
    for bid_metrics in data["metrics"].get("per_model", {}).values():
        bids.update(bid_metrics.keys())
    bids = sorted(bids)
    if env_filter:
        bids = [b for b in bids if b.startswith(env_filter + "/")]
    return bids


def get_trajectory_paths(
    data: Dict[str, Any],
    model_name: str,
    benchmark_id: str,
) -> List[str]:
    """Get file paths for trajectories of a specific (model, benchmark_id) pair."""
    return (
        data.get("trajectory_index", {})
        .get(model_name, {})
        .get(benchmark_id, [])
    )


def scan_run_dirs(base_dir: str) -> List[str]:
    """Scan for DataStore directories (ones that have metadata.json)."""
    base = Path(base_dir)
    if not base.exists():
        return []
    runs = []
    # Check base_dir itself
    if (base / "metadata.json").exists():
        runs.append(str(base))
    # Check immediate subdirectories
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / "metadata.json").exists():
            runs.append(str(d))
    return runs


def has_double_agent_data(data: Dict[str, Any]) -> bool:
    """Check if DataStore has double-agent metrics."""
    da = data.get("metrics", {}).get("double_agent", {})
    return bool(da.get("per_schedule"))


def has_detailed_data(data: Dict[str, Any]) -> bool:
    """Check if DataStore has exploration/exploitation metrics."""
    detailed = data.get("metrics", {}).get("detailed", {})
    return bool(detailed.get("exploration_exploitation"))


def has_comparison_data(data: Dict[str, Any]) -> bool:
    """Check if DataStore has comparison metrics."""
    da = data.get("metrics", {}).get("double_agent", {})
    comp = da.get("comparisons", {})
    return bool(comp and comp.get("per_benchmark_id"))
