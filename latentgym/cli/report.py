"""
Generate reports from benchmark results (tables, plots, dashboard, trajectory viewer).

Usage:
    # Generate all reports (tables + plots + dashboard + trajectory explorer)
    python -m latentgym.cli.report --data-dir results/run_001/ --output paper/

    # Just tables (markdown + LaTeX + CSV)
    python -m latentgym.cli.report --data-dir results/ --tables-only

    # Just plots (PNG/PDF/SVG)
    python -m latentgym.cli.report --data-dir results/ --plots-only

    # Interactive HTML dashboard (single file, open in browser)
    python -m latentgym.cli.report --data-dir results/ --dashboard

    # Interactive trajectory explorer (single file, open in browser)
    python -m latentgym.cli.report --data-dir results/ --trajectories

    # Print leaderboard to stdout
    python -m latentgym.cli.report --data-dir results/ --leaderboard
    python -m latentgym.cli.report --data-dir results/ --leaderboard --env bandits

    # Render a specific trajectory as HTML or text
    python -m latentgym.cli.report \\
        --trajectory results/trajectories/gpt-4o/bandits__loyal_favorite_0/traj_0000.json
    python -m latentgym.cli.report \\
        --trajectory results/trajectories/.../traj_0000.json --html trajectory.html

    # Compare two models head-to-head
    python -m latentgym.cli.report --data-dir results/ --compare gpt-4o claude-3.5
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def cmd_all(args):
    """Generate all reports: tables + plots + dashboard + trajectory explorer."""
    _write_tables(args.data_dir, args.output)
    _write_plots(args.data_dir, args.output, fmt=args.fmt)
    _write_dashboard(args.data_dir, args.output)
    _write_trajectory_explorer(args.data_dir, args.output)
    logger.info(f"All reports written to {args.output}")


def cmd_tables(args):
    _write_tables(args.data_dir, args.output)


def cmd_plots(args):
    _write_plots(args.data_dir, args.output, fmt=args.fmt)


def cmd_dashboard(args):
    _write_dashboard(args.data_dir, args.output)


def cmd_trajectories(args):
    _write_trajectory_explorer(args.data_dir, args.output)


def cmd_leaderboard(args):
    from latentgym.reporting.data_store import DataStore
    from latentgym.reporting.leaderboard import compute_leaderboard, format_leaderboard

    data = DataStore.load(args.data_dir)
    per_model = data["metrics"].get("per_model", {})
    lb = compute_leaderboard(per_model, env_filter=args.env)
    print(format_leaderboard(lb))


def cmd_trajectory(args):
    from latentgym.reporting.data_store import DataStore
    from latentgym.reporting.trajectory_viewer import TrajectoryViewer

    traj = DataStore.load_trajectory(args.trajectory)
    viewer = TrajectoryViewer()

    if args.html:
        html = viewer.render_html(traj)
        Path(args.html).write_text(html)
        logger.info(f"HTML written to {args.html}")
    else:
        print(viewer.render_text(traj))


def cmd_recompute(args):
    """Recompute metrics from saved trajectory JSONs on disk.

    Scans the trajectories/ directory, loads all trajectory JSONs,
    rebuilds BenchmarkResults, and recomputes metrics + saves them.
    Useful after running models separately to the same output dir.
    """
    import json
    from latentgym.eval.results import BenchmarkResults
    from latentgym.eval.types import TrajectoryResult
    from latentgym.reporting.single_agent_report import SingleAgentReport

    root = Path(args.data_dir)
    traj_dir = root / "trajectories"

    if not traj_dir.exists():
        logger.error(f"No trajectories/ directory found in {args.data_dir}")
        return

    # Load all trajectory JSONs into BenchmarkResults
    results = BenchmarkResults()
    n_loaded = 0
    for traj_path in sorted(traj_dir.rglob("traj_*.json")):
        try:
            with open(traj_path) as f:
                d = json.load(f)
            traj = TrajectoryResult.from_dict(d)
            model_name = traj.model_name
            bid = traj.benchmark_id
            if not model_name or not bid:
                logger.warning(f"  Skipping {traj_path} — missing model_name or benchmark_id")
                continue
            results.add(model_name, bid, [traj])
            n_loaded += 1
        except Exception as e:
            logger.warning(f"  Failed to load {traj_path}: {e}")

    if n_loaded == 0:
        logger.error("No trajectories loaded — nothing to recompute")
        return

    logger.info(f"Loaded {n_loaded} trajectories from {traj_dir}")
    logger.info(f"Models: {results.model_names}")
    logger.info(f"Configs: {len(results.benchmark_ids)}")

    # Recompute and save metrics only (don't overwrite trajectory files on disk)
    from latentgym.reporting.data_store import DataStore
    report = SingleAgentReport(results)
    report.compute()
    store = DataStore(str(root))
    report.save_to(store)  # Writes metrics + tables only, not trajectories
    store.write_metadata({
        "run_name": "recomputed",
        "report_type": "single_agent",
        "models": results.model_names,
        "benchmark_ids": results.benchmark_ids,
    })
    logger.info(f"Metrics recomputed and saved to {root}")

    # Only generate reports if --output was explicitly provided (not the default "paper/")
    # User should run a separate report command after recompute if they want reports


def cmd_compare(args):
    from latentgym.reporting.data_store import DataStore
    from latentgym.reporting.leaderboard import compute_model_comparison

    data = DataStore.load(args.data_dir)
    per_model = data["metrics"].get("per_model", {})

    model_a, model_b = args.compare
    comp = compute_model_comparison(per_model, model_a, model_b)

    print(f"\nHead-to-head: {model_a} vs {model_b}")
    print(f"  Configs compared: {comp['n_configs']}")
    print(f"  {model_a} wins: {comp['wins_a']}")
    print(f"  {model_b} wins: {comp['wins_b']}")
    print(f"  Ties:          {comp['ties']}")
    print(f"  Avg delta (A-B): {comp['avg_delta']:+.4f}")
    print("\nPer-config deltas (A - B):")
    for bid, delta in sorted(comp["per_config_delta"].items()):
        marker = "+" if delta > 0 else ("-" if delta < 0 else "=")
        print(f"  {marker} {bid:<50}  {delta:+.4f}")


# =============================================================================
# Internal helpers
# =============================================================================

def _write_tables(data_dir: str, output_dir: str):
    from latentgym.reporting.data_store import DataStore
    from latentgym.reporting.tables import (
        make_results_table, make_leaderboard_table,
        make_per_env_table, make_complexity_table,
        make_double_agent_summary_table, make_per_agent_table,
        make_comparison_table, make_exploration_exploitation_table,
    )

    data = DataStore.load(data_dir)
    per_model = data["metrics"].get("per_model", {})
    per_env = data["metrics"].get("per_env", {})
    per_complexity = data["metrics"].get("per_latent_complexity", {})
    leaderboard = data["metrics"].get("leaderboard", [])
    detailed = data["metrics"].get("detailed", {}).get("exploration_exploitation", {})
    da_schedule = data["metrics"].get("double_agent", {}).get("per_schedule", {})
    da_agent = data["metrics"].get("double_agent", {}).get("per_agent", {})
    da_comparisons = data["metrics"].get("double_agent", {}).get("comparisons", {})

    out = Path(output_dir) / "tables"
    out.mkdir(parents=True, exist_ok=True)

    # Main results table
    for fmt in ("markdown", "latex", "csv"):
        table = make_results_table(per_model, fmt=fmt)
        ext = {"markdown": "md", "latex": "tex", "csv": "csv"}[fmt]
        path = out / f"main_results.{ext}"
        path.write_text(table)
        logger.info(f"  Wrote {path}")

    # Leaderboard
    for fmt in ("markdown", "csv"):
        table = make_leaderboard_table(leaderboard, fmt=fmt)
        ext = "md" if fmt == "markdown" else "csv"
        path = out / f"leaderboard.{ext}"
        path.write_text(table)
        logger.info(f"  Wrote {path}")

    # Per-env table
    if per_env:
        table = make_per_env_table(per_env, fmt="markdown")
        (out / "per_env.md").write_text(table)
        logger.info(f"  Wrote {out}/per_env.md")

    # Complexity table
    if per_complexity:
        table = make_complexity_table(per_complexity, fmt="markdown")
        (out / "per_complexity.md").write_text(table)
        logger.info(f"  Wrote {out}/per_complexity.md")

    # Exploration/exploitation table
    if detailed:
        table = make_exploration_exploitation_table(detailed, fmt="markdown")
        (out / "exploration_exploitation.md").write_text(table)
        logger.info(f"  Wrote {out}/exploration_exploitation.md")

    # Double-agent tables
    if da_schedule:
        table = make_double_agent_summary_table(da_schedule, fmt="markdown")
        (out / "double_agent_summary.md").write_text(table)
        logger.info(f"  Wrote {out}/double_agent_summary.md")

        for fmt in ("csv",):
            table = make_double_agent_summary_table(da_schedule, fmt=fmt)
            (out / f"double_agent_summary.{fmt}").write_text(table)

    if da_agent:
        table = make_per_agent_table(da_agent, fmt="markdown")
        (out / "per_agent_breakdown.md").write_text(table)
        logger.info(f"  Wrote {out}/per_agent_breakdown.md")

    if da_comparisons:
        table = make_comparison_table(da_comparisons, fmt="markdown")
        (out / "comparison.md").write_text(table)
        logger.info(f"  Wrote {out}/comparison.md")


def _write_plots(data_dir: str, output_dir: str, fmt: str = "png"):
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        logger.warning("matplotlib not installed — skipping plots")
        return

    from latentgym.reporting.data_store import DataStore
    from latentgym.reporting.plots import save_all_plots

    data = DataStore.load(data_dir)
    plots_dir = Path(output_dir) / "plots"
    saved = save_all_plots(data, str(plots_dir), fmt=fmt)
    logger.info(f"  Saved {len(saved)} plots to {plots_dir}")


def _write_dashboard(data_dir: str, output_dir: str):
    from latentgym.reporting.dashboard import render_dashboard_from_dir

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    html = render_dashboard_from_dir(data_dir)
    path = out / "dashboard.html"
    path.write_text(html)
    logger.info(f"  Dashboard written to {path}")


def _write_trajectory_explorer(data_dir: str, output_dir: str):
    from latentgym.reporting.trajectory_viewer import TrajectoryViewer

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    viewer = TrajectoryViewer()
    html = viewer.render_interactive_html_from_dir(data_dir)
    path = out / "trajectory_explorer.html"
    path.write_text(html)
    logger.info(f"  Trajectory explorer written to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate reports from benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--data-dir", default=None, help="DataStore output directory")
    parser.add_argument("--output", default="paper/", help="Output directory for reports")
    parser.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"],
                        help="Plot image format")

    # Actions (mutually exclusive, default = all)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tables-only", action="store_true", help="Generate tables only")
    group.add_argument("--plots-only", action="store_true", help="Generate plots only")
    group.add_argument("--dashboard", action="store_true",
                        help="Generate interactive HTML dashboard")
    group.add_argument("--trajectories", action="store_true",
                        help="Generate interactive trajectory explorer HTML")
    group.add_argument("--leaderboard", action="store_true", help="Print leaderboard to stdout")
    group.add_argument("--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                        help="Compare two models head-to-head")
    group.add_argument("--trajectory", default=None,
                        help="Path to single trajectory JSON (render as text/HTML)")

    group.add_argument("--recompute", action="store_true",
                        help="Recompute metrics from saved trajectory JSONs on disk. "
                             "Use after running models separately to the same output dir.")

    parser.add_argument("--env", default=None, help="Filter leaderboard to this env")
    parser.add_argument("--html", default=None, help="Output HTML file (for --trajectory)")

    args = parser.parse_args()

    if args.trajectory:
        cmd_trajectory(args)
    elif args.leaderboard:
        if not args.data_dir:
            parser.error("--data-dir is required")
        cmd_leaderboard(args)
    elif args.compare:
        if not args.data_dir:
            parser.error("--data-dir is required")
        cmd_compare(args)
    elif args.tables_only:
        if not args.data_dir:
            parser.error("--data-dir is required")
        cmd_tables(args)
    elif args.plots_only:
        if not args.data_dir:
            parser.error("--data-dir is required")
        cmd_plots(args)
    elif args.dashboard:
        if not args.data_dir:
            parser.error("--data-dir is required")
        cmd_dashboard(args)
    elif args.trajectories:
        if not args.data_dir:
            parser.error("--data-dir is required")
        cmd_trajectories(args)
    elif args.recompute:
        if not args.data_dir:
            parser.error("--data-dir is required")
        cmd_recompute(args)
    else:
        if not args.data_dir:
            parser.error("--data-dir is required")
        cmd_all(args)


if __name__ == "__main__":
    main()
