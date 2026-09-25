"""
Benchmark reporting: DataStore (I/O), Report classes (metrics), tables, plots, leaderboard.

Report classes (Option B architecture):
    SingleAgentReport  — standard single-model eval metrics
    DoubleAgentReport  — pre/post switch, transfer, per-agent breakdown
    ComparisonReport   — P_f vs Q_b cross-model comparison

All reports write to a shared DataStore directory. They can be used
independently or layered (single + double on the same output_dir).

Usage:
    # Single-agent eval
    report = SingleAgentReport(results)
    report.save("results/run_001")

    # Double-agent eval (same or different output_dir)
    report = DoubleAgentReport(results, switch_episode=5)
    report.save("results/run_001")

    # Comparison (P_f vs Q_b)
    report = ComparisonReport(results, finetuned_key="P_f", base_key="Q_b")
    report.save("results/run_001")

    # Layer onto same DataStore
    store = DataStore("results/run_001")
    SingleAgentReport(single_results).save_to(store)
    DoubleAgentReport(double_results, switch_episode=5).save_to(store)
    ComparisonReport(all_results).save_to(store)

    # Read back
    data = DataStore.load("results/run_001")
    data["metrics"]["per_model"]                          # single-agent
    data["metrics"]["double_agent"]["per_schedule"]       # double-agent
    data["metrics"]["double_agent"]["comparisons"]        # comparison
"""
from .data_store import DataStore
from .single_agent_report import SingleAgentReport
from .double_agent_report import DoubleAgentReport
from .comparison_report import ComparisonReport
from .tables import (
    make_results_table,
    make_leaderboard_table,
    make_per_env_table,
    make_complexity_table,
    make_double_agent_summary_table,
    make_per_agent_table,
    make_comparison_table,
    make_exploration_exploitation_table,
)
from .leaderboard import compute_leaderboard
from .dashboard import render_dashboard, render_dashboard_from_dir

__all__ = [
    # I/O
    "DataStore",
    # Reports
    "SingleAgentReport",
    "DoubleAgentReport",
    "ComparisonReport",
    # Tables
    "make_results_table",
    "make_leaderboard_table",
    "make_per_env_table",
    "make_complexity_table",
    "make_double_agent_summary_table",
    "make_per_agent_table",
    "make_comparison_table",
    "make_exploration_exploitation_table",
    # Dashboard
    "render_dashboard",
    "render_dashboard_from_dir",
    # Leaderboard
    "compute_leaderboard",
]
