"""
LatentGym: Streamlit App

Launch:
    streamlit run benchmark/app/app.py -- --data-dir results/
    streamlit run benchmark/app/app.py -- --data-dir results/ --scan
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import streamlit as st
import pandas as pd


# ── Args ─────────────────────────────────────────────────────────────────────
def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="results/")
    parser.add_argument("--scan", action="store_true")
    # Try to find our args after "--" separator
    try:
        idx = sys.argv.index("--")
        our_args = sys.argv[idx + 1:]
    except ValueError:
        # No "--" found — try parsing all args (streamlit passes script args directly)
        our_args = sys.argv[1:]
    args, _ = parser.parse_known_args(our_args)
    return args


ARGS = _get_args()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LatentGym Benchmark",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Tighter spacing */
    .block-container { padding-top: 1.5rem; }
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetric"] label { font-size: 0.8rem; color: #666; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.4rem; }
    /* Sidebar */
    [data-testid="stSidebar"] { background: #fafbfc; }
    [data-testid="stSidebar"] hr { margin: 8px 0; }
    /* Tables */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    /* Page title styling */
    .page-header { margin-bottom: 0.5rem; }
    .page-subtitle { color: #888; font-size: 0.9rem; margin-top: -0.8rem; margin-bottom: 1.2rem; }
    /* Filter bar */
    .filter-bar { background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 16px; border: 1px solid #eee; }
    /* Success/failure pills */
    .pill-success { background: #d4edda; color: #155724; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; }
    .pill-failure { background: #f8d7da; color: #721c24; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────────────────────
from latentgym.app.data_loader import (
    load_data, load_trajectory, get_model_names, get_env_names,
    get_benchmark_ids, get_trajectory_paths, scan_run_dirs,
    has_double_agent_data, has_detailed_data, has_comparison_data,
)

# ── Run selector ─────────────────────────────────────────────────────────────
if ARGS.scan:
    runs = scan_run_dirs(ARGS.data_dir)
    if not runs:
        st.error(f"No DataStore directories found in `{ARGS.data_dir}`")
        st.stop()
    DATA_DIR = st.sidebar.selectbox("📁 Select Run", runs, format_func=lambda x: Path(x).name) if len(runs) > 1 else runs[0]
else:
    DATA_DIR = ARGS.data_dir

if not Path(DATA_DIR).exists():
    st.error(f"Data directory not found: `{DATA_DIR}`")
    st.info("Generate data first: `python -m latentgym.cli.run_eval single --models ... --output results/`")
    st.stop()

data = load_data(DATA_DIR)

# Debug: show what was loaded
with st.sidebar.expander("🔧 Debug", expanded=True):
    st.write(f"Data dir: `{DATA_DIR}`")
    st.write(f"Data keys: {list(data.keys())}")
    st.write(f"Metrics keys: {list(data.get('metrics', {}).keys())}")
    st.write(f"per_model: {list(data.get('metrics', {}).get('per_model', {}).keys())}")
    st.write(f"traj_index: {list(data.get('trajectory_index', {}).keys())}")

metadata = data.get("metadata", {})
metrics = data.get("metrics", {})
per_model = metrics.get("per_model", {})
per_env = metrics.get("per_env", {})
per_complexity = metrics.get("per_latent_complexity", {})
leaderboard = metrics.get("leaderboard", [])
detailed = metrics.get("detailed", {}).get("exploration_exploitation", {})
da_schedule = metrics.get("double_agent", {}).get("per_schedule", {})
da_agent = metrics.get("double_agent", {}).get("per_agent", {})
da_comparisons = metrics.get("double_agent", {}).get("comparisons", {})

model_names = get_model_names(data)
env_names = get_env_names(data)
all_bids = get_benchmark_ids(data)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🎯 Benchmark")

run_name = metadata.get("run_name", Path(DATA_DIR).name)
created = metadata.get("created_at", "")[:16]
st.sidebar.caption(f"**{run_name}**")
if created:
    st.sidebar.caption(f"📅 {created}")
st.sidebar.caption(f"👥 {len(model_names)} models · 🎮 {len(env_names)} envs · 📊 {len(all_bids)} configs")

st.sidebar.divider()

# Pages
pages = {
    "Leaderboard": "🏆",
    "Results Table": "📋",
    "Per-Environment": "🌍",
    "By Complexity": "📊",
    "Learning Curves": "📈",
    "Trajectory Viewer": "🔍",
    "Model Comparison": "⚔️",
    "Custom Comparison": "🔧",
}
if has_detailed_data(data):
    pages["Exploration vs Exploitation"] = "🔬"
if has_double_agent_data(data):
    pages["Double-Agent"] = "🤝"
pages["Issues & Suggestions"] = "💡"

page = st.sidebar.radio(
    "Navigate",
    list(pages.keys()),
    format_func=lambda x: f"{pages[x]} {x}",
)

# Downloads
st.sidebar.divider()
with st.sidebar.expander("📥 Downloads"):
    try:
        from latentgym.reporting.dashboard import render_dashboard
        html = render_dashboard(data, title=run_name)
        st.download_button("Dashboard HTML", html, "dashboard.html", "text/html", use_container_width=True)
    except Exception:
        pass
    try:
        from latentgym.reporting.trajectory_viewer import TrajectoryViewer
        _v = TrajectoryViewer()
        _t = _v.browse(DATA_DIR, max_results=200)
        if _t:
            st.download_button("Trajectory Explorer HTML",
                               _v.render_interactive_html(_t),
                               "trajectories.html", "text/html", use_container_width=True)
    except Exception:
        pass


# ── Helpers ──────────────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str = ""):
    st.markdown(f"<h1 class='page-header'>{pages.get(title, '')} {title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p class='page-subtitle'>{subtitle}</p>", unsafe_allow_html=True)


def metric_row(cols_data: list):
    """Render a row of metric cards. cols_data = [(label, value, delta?), ...]"""
    cols = st.columns(len(cols_data))
    for col, item in zip(cols, cols_data):
        if len(item) == 3:
            col.metric(item[0], item[1], item[2])
        else:
            col.metric(item[0], item[1])


def styled_df(df: pd.DataFrame, **kwargs):
    st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def download_csv(df: pd.DataFrame, filename: str):
    st.download_button(f"⬇ Download CSV", df.to_csv(index=False), filename, "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

if page == "Leaderboard":
    page_header("Leaderboard", f"Ranked by average reward across {len(all_bids)} configs")

    if not leaderboard:
        st.warning("No results found.")
        st.stop()

    # Top model highlight
    top = leaderboard[0]
    metric_row([
        ("🥇 Top Model", top["model"]),
        ("Avg Reward", f"{top['avg_reward']:.4f}"),
        ("Models Evaluated", len(leaderboard)),
        ("Total Configs", len(all_bids)),
    ])

    st.subheader("Rankings")
    df = pd.DataFrame(leaderboard)
    styled_df(df)
    download_csv(df, "leaderboard.csv")

    # Per-env heatmap
    if per_env and len(model_names) > 0:
        st.subheader("Per-Environment Breakdown")
        table_data = {"Environment": env_names}
        for model in model_names:
            table_data[model] = [
                round(per_env.get(env, {}).get(model, {}).get("avg_mean_reward", 0), 4)
                for env in env_names
            ]
        styled_df(pd.DataFrame(table_data))

        try:
            from latentgym.reporting.plots import plot_env_bar_chart
            fig = plot_env_bar_chart(per_env)
            st.pyplot(fig)
        except ImportError:
            pass


elif page == "Results Table":
    page_header("Results Table", "Full model × config results with filtering")

    if not per_model:
        st.warning("No results found.")
        st.stop()

    # Filter bar
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_env = st.selectbox("Environment", ["All"] + env_names, help="Filter by environment")
        with col2:
            filter_model = st.selectbox("Model", ["All"] + model_names, help="Filter by model")
        with col3:
            sort_metric = st.selectbox("Sort by", [
                "Mean Reward", "Improvement", "Success Rate", "Final", "Slope"
            ], help="Sort the table by this metric")

    rows = []
    for model, bid_metrics in per_model.items():
        if filter_model != "All" and model != filter_model:
            continue
        for bid, m in bid_metrics.items():
            parts = bid.split("/")
            env = parts[0]
            if filter_env != "All" and env != filter_env:
                continue
            rows.append({
                "Model": model,
                "Env": env,
                "Latent": parts[1] if len(parts) > 1 else "",
                "Prompt": parts[2] if len(parts) > 2 else "",
                "Feedback": parts[3] if len(parts) > 3 else "",
                "Mean Reward": round(m.get("avg_mean_reward", 0), 4),
                "Improvement": round(m.get("avg_improvement", 0), 4),
                "Success Rate": round(m.get("avg_success_rate", 0), 4),
                "Initial": round(m.get("avg_initial_reward", 0), 4),
                "Final": round(m.get("avg_final_reward", 0), 4),
                "Slope": round(m.get("learning_slope", 0), 4),
                "Turns": round(m.get("avg_total_turns", 0), 1),
                "# Traj": m.get("n_trajectories", 0),
            })

    if rows:
        df = pd.DataFrame(rows).sort_values(sort_metric, ascending=False)
        metric_row([
            ("Rows", len(df)),
            ("Best " + sort_metric, f"{df[sort_metric].max():.4f}"),
            ("Worst " + sort_metric, f"{df[sort_metric].min():.4f}"),
            ("Mean " + sort_metric, f"{df[sort_metric].mean():.4f}"),
        ])
        styled_df(df)
        download_csv(df, "results.csv")
    else:
        st.info("No results match filters.")


elif page == "Per-Environment":
    page_header("Per-Environment", "Compare models within each environment")

    if not env_names:
        st.warning("No results.")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_env = st.selectbox("Environment", env_names)
    with col2:
        metric = st.selectbox("Metric", [
            "avg_mean_reward", "avg_improvement", "avg_success_rate",
            "avg_initial_reward", "avg_final_reward",
        ])

    env_data = per_env.get(selected_env, {})
    if not env_data:
        st.warning(f"No data for {selected_env}")
        st.stop()

    rows = []
    for model in sorted(env_data.keys()):
        m = env_data[model]
        rows.append({
            "Model": model,
            "Avg Reward": round(m.get("avg_mean_reward", 0), 4),
            "Improvement": round(m.get("avg_improvement", 0), 4),
            "Success": round(m.get("avg_success_rate", 0), 4),
            "Initial": round(m.get("avg_initial_reward", 0), 4),
            "Final": round(m.get("avg_final_reward", 0), 4),
        })
    df = pd.DataFrame(rows)

    # Best model highlight
    if not df.empty:
        best = df.loc[df["Avg Reward"].idxmax()]
        metric_row([
            ("Best Model", best["Model"]),
            ("Avg Reward", f"{best['Avg Reward']:.4f}"),
            ("Improvement", f"{best['Improvement']:+.4f}"),
            ("Success", f"{best['Success']:.1%}"),
        ])

    styled_df(df)

    try:
        from latentgym.reporting.plots import plot_env_bar_chart
        fig = plot_env_bar_chart({selected_env: env_data}, metric=metric,
                                 title=f"{selected_env}: {metric.replace('avg_', '')}")
        st.pyplot(fig)
    except ImportError:
        pass


elif page == "By Complexity":
    page_header("By Complexity", "Performance breakdown by latent difficulty tier")

    if not per_complexity:
        st.warning("No complexity data.")
        st.stop()

    complexities = sorted(per_complexity.keys())
    rows = []
    for c in complexities:
        row = {"Complexity": c}
        for model in model_names:
            val = per_complexity.get(c, {}).get(model, {}).get("avg_mean_reward", None)
            row[model] = round(val, 4) if val is not None else None
        rows.append(row)
    df = pd.DataFrame(rows)
    styled_df(df)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(complexities))
        width = 0.8 / max(len(model_names), 1)
        for i, model in enumerate(model_names):
            vals = [per_complexity.get(c, {}).get(model, {}).get("avg_mean_reward", 0) for c in complexities]
            offset = (i - len(model_names) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", " ").title() for c in complexities])
        ax.set_ylabel("Avg Mean Reward")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        st.pyplot(fig)
    except ImportError:
        pass


elif page == "Learning Curves":
    page_header("Learning Curves", "Per-episode reward progression showing in-context learning")

    if not env_names:
        st.warning("No results.")
        st.stop()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        selected_env = st.selectbox("Environment", env_names)
    with col2:
        bids = get_benchmark_ids(data, env_filter=selected_env)
        short_bids = {"/".join(b.split("/")[1:]): b for b in bids}
        selected_short = st.selectbox("Config", list(short_bids.keys()))
        selected_bid = short_bids[selected_short]
    with col3:
        curve_metric = st.radio("Metric", ["Rewards", "Turns"], horizontal=True)

    metric_key = "per_episode_avg_rewards" if curve_metric == "Rewards" else "per_episode_avg_turns"

    try:
        from latentgym.reporting.plots import plot_learning_curves
        fig = plot_learning_curves(per_model, selected_bid, metric=metric_key)
        st.pyplot(fig)
    except ImportError:
        st.info("Install matplotlib for charts")

    with st.expander("📊 Raw per-episode data"):
        rows = []
        max_eps = max(
            (len(per_model[m].get(selected_bid, {}).get("per_episode_avg_rewards", []))
             for m in model_names), default=0)
        for ep in range(max_eps):
            row = {"Episode": ep}
            for m in model_names:
                vals = per_model[m].get(selected_bid, {}).get("per_episode_avg_rewards", [])
                row[m] = round(vals[ep], 4) if ep < len(vals) else None
            rows.append(row)
        if rows:
            styled_df(pd.DataFrame(rows))


elif page == "Trajectory Viewer":
    page_header("Trajectory Viewer", "Inspect individual trajectories with full conversation and ground truth")

    if not model_names:
        st.warning("No results.")
        st.stop()

    # Filter bar
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_model = st.selectbox("Model", model_names)
    with col2:
        selected_env = st.selectbox("Environment", env_names)
    with col3:
        bids = get_benchmark_ids(data, env_filter=selected_env)
        short_bids = {"/".join(b.split("/")[1:]): b for b in bids}
        selected_short = st.selectbox("Config", list(short_bids.keys()))
        selected_bid = short_bids[selected_short]

    paths = get_trajectory_paths(data, selected_model, selected_bid)
    if not paths:
        st.warning(f"No trajectories for {selected_model} / {selected_bid}")
        st.stop()

    traj_idx = st.slider(f"Trajectory ({len(paths)} available)", 0, len(paths) - 1, 0)
    traj = load_trajectory(paths[traj_idx])

    # Summary
    ep_rewards = traj.get("episode_rewards", [])
    improvement = (ep_rewards[-1] - ep_rewards[0]) if len(ep_rewards) >= 2 else 0
    metric_row([
        ("Total Reward", f"{sum(ep_rewards):.4f}"),
        ("Improvement", f"{improvement:+.4f}"),
        ("Episodes", len(ep_rewards)),
        ("Success Rate", f"{traj.get('success_rate', 0):.1%}"),
    ])

    # Metadata
    with st.expander("ℹ️ Trajectory Metadata", expanded=False):
        mc = st.columns(3)
        mc[0].markdown(f"**Model:** `{traj.get('model_name', '?')}`")
        mc[0].markdown(f"**Env:** `{traj.get('env_name', '?')}`")
        mc[0].markdown(f"**Latent:** `{traj.get('latent_id', '?')}`")
        mc[1].markdown(f"**Prompt:** `{traj.get('prompt_id', '?')}`")
        mc[1].markdown(f"**Feedback:** `{traj.get('feedback_id', '?')}`")
        mc[1].markdown(f"**Seed:** `{traj.get('seed', 0)}`")
        ep = traj.get("env_params", {})
        if ep:
            mc[2].markdown("**Env Params:**")
            for k, v in sorted(ep.items()):
                mc[2].markdown(f"`{k}`: {v}")

    # Episode table
    outcomes = traj.get("episode_outcomes", [])
    assignments = traj.get("agent_assignments", [])
    episode_configs = traj.get("episode_configs", [])
    has_agents = assignments and len(set(assignments)) > 1
    has_gt = bool(episode_configs) or any(o.get("ground_truth") for o in outcomes)

    if outcomes:
        ep_rows = []
        for i, o in enumerate(outcomes):
            row = {
                "Ep": o.get("episode_idx", i),
                "Reward": round(o.get("reward", 0), 4),
                "Turns": o.get("turns", 0),
                "Eff": round(o.get("turn_efficiency", 0), 2),
                "Result": "✅" if o.get("success") else "❌",
                "Outcome": o.get("outcome_type", ""),
            }
            if has_agents:
                row["Agent"] = assignments[i] if i < len(assignments) else "?"
            if has_gt:
                gt = (episode_configs[i] if i < len(episode_configs) else None) or o.get("ground_truth", {})
                if gt:
                    if "target_word" in gt and "start_word" not in gt:
                        row["Ground Truth"] = gt["target_word"]
                    elif "ground_truth" in gt and isinstance(gt["ground_truth"], dict):
                        best = max(gt["ground_truth"], key=gt["ground_truth"].get)
                        row["Ground Truth"] = f"{best}={gt['ground_truth'][best]:.2f}"
                    elif "secret_code" in gt:
                        row["Ground Truth"] = str(gt["secret_code"])
                    elif "draws" in gt:
                        row["Ground Truth"] = f"max@{gt['draws'].index(max(gt['draws']))}"
                    elif "start_word" in gt:
                        row["Ground Truth"] = f"{gt['start_word']}→{gt.get('target_word', '?')}"
                    else:
                        row["Ground Truth"] = str(gt)[:30]
                else:
                    row["Ground Truth"] = ""
            ep_rows.append(row)
        styled_df(pd.DataFrame(ep_rows))

    if has_gt:
        with st.expander("🔑 Ground Truth Details"):
            configs = episode_configs or [o.get("ground_truth", {}) for o in outcomes]
            for i, gt in enumerate(configs):
                if gt:
                    st.markdown(f"**Episode {i}:**")
                    st.json(gt)

    # Conversation (with reasoning trace)
    reasoning_trace = traj.get("reasoning_trace", [])
    has_reasoning = any(r for r in reasoning_trace if r)
    if has_reasoning:
        n_with = sum(1 for r in reasoning_trace if r)
        st.subheader(f"💬 Conversation  🧠 {n_with}/{len(reasoning_trace)} turns with reasoning")
    else:
        st.subheader("💬 Conversation")

    conversation = traj.get("conversation", [])
    assistant_idx = 0
    for msg in conversation:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if role == "assistant":
            # Show reasoning before the action if available
            if assistant_idx < len(reasoning_trace) and reasoning_trace[assistant_idx]:
                with st.expander("🧠 Reasoning (internal thinking — not shown to env)", expanded=False):
                    st.code(reasoning_trace[assistant_idx], language=None)
            assistant_idx += 1
            with st.chat_message("assistant"):
                st.text(content)
        elif role == "system":
            with st.chat_message("user", avatar="⚙️"):
                st.text(content)
        else:
            with st.chat_message("user"):
                st.text(content)

    # ── Review & Annotation ──
    st.divider()
    st.subheader("📝 Review & Annotations")

    # Load/save annotations from a JSON file in the DataStore directory
    import json as _json
    annotations_path = Path(DATA_DIR) / "annotations.json"

    def _load_annotations() -> dict:
        if annotations_path.exists():
            with open(annotations_path) as f:
                return _json.load(f)
        return {}

    def _save_annotations(annot: dict):
        with open(annotations_path, "w") as f:
            _json.dump(annot, f, indent=2)

    traj_key = f"{selected_model}/{selected_bid}/traj_{traj_idx:04d}"
    all_annotations = _load_annotations()
    traj_annot = all_annotations.get(traj_key, {})

    # Status badge
    review_status = traj_annot.get("status", "unreviewed")
    status_options = ["unreviewed", "reviewed", "flagged", "approved", "rejected"]
    status_colors = {
        "unreviewed": "🔘", "reviewed": "✅", "flagged": "🚩",
        "approved": "👍", "rejected": "👎",
    }

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        new_status = st.selectbox(
            "Review Status",
            status_options,
            index=status_options.index(review_status),
            format_func=lambda x: f"{status_colors.get(x, '')} {x.title()}",
            key=f"status_{traj_key}",
        )
    with rc2:
        reviewer_name = st.text_input("Reviewer", value=traj_annot.get("reviewer", ""),
                                       placeholder="Your name", key=f"reviewer_{traj_key}")

    # Issue reporting
    issue_types = [
        "No issue",
        "Wrong reward / scoring bug",
        "Environment logic error",
        "Prompt unclear or misleading",
        "Feedback missing or incorrect",
        "Ground truth mismatch",
        "Agent produced invalid action",
        "Episode transition error",
        "Latent constraint not applied correctly",
        "Other",
    ]
    current_issue = traj_annot.get("issue_type", "No issue")
    issue_type = st.selectbox(
        "Report Issue",
        issue_types,
        index=issue_types.index(current_issue) if current_issue in issue_types else 0,
        key=f"issue_{traj_key}",
    )

    issue_detail = ""
    if issue_type != "No issue":
        issue_detail = st.text_area(
            "Issue Details",
            value=traj_annot.get("issue_detail", ""),
            placeholder="Describe the issue — which episode, what went wrong, expected vs actual behavior...",
            key=f"issue_detail_{traj_key}",
        )

    # Notes
    notes = st.text_area(
        "Notes",
        value=traj_annot.get("notes", ""),
        placeholder="Any observations about this trajectory...",
        key=f"notes_{traj_key}",
        height=80,
    )

    # Episode-level flags
    with st.expander("🏳️ Flag specific episodes"):
        flagged_episodes = traj_annot.get("flagged_episodes", {})
        for i, o in enumerate(outcomes):
            ep_flag = flagged_episodes.get(str(i), "")
            flag_val = st.text_input(
                f"Episode {i} (reward={o.get('reward', 0):.3f})",
                value=ep_flag,
                placeholder="Flag reason (leave empty if OK)",
                key=f"epflag_{traj_key}_{i}",
            )
            if flag_val:
                flagged_episodes[str(i)] = flag_val
            elif str(i) in flagged_episodes:
                del flagged_episodes[str(i)]

    # Save button
    if st.button("💾 Save Annotation", type="primary", key=f"save_{traj_key}"):
        from datetime import datetime
        all_annotations[traj_key] = {
            "status": new_status,
            "reviewer": reviewer_name,
            "issue_type": issue_type,
            "issue_detail": issue_detail if issue_type != "No issue" else "",
            "notes": notes,
            "flagged_episodes": flagged_episodes,
            "updated_at": datetime.now().isoformat(),
            "trajectory": {
                "model": selected_model,
                "benchmark_id": selected_bid,
                "traj_idx": traj_idx,
                "seed": traj.get("seed", 0),
                "total_reward": sum(ep_rewards),
                "success_rate": traj.get("success_rate", 0),
            },
        }
        _save_annotations(all_annotations)
        st.success("Annotation saved!")

    # Show annotation summary for this run
    n_annotated = sum(1 for v in all_annotations.values() if v.get("status") != "unreviewed")
    n_flagged = sum(1 for v in all_annotations.values() if v.get("issue_type", "No issue") != "No issue")
    n_total = len(all_annotations)

    st.caption(f"📊 This run: {n_annotated} reviewed, {n_flagged} with issues, {n_total} annotated total")

    # Export annotations
    with st.expander("📋 All Annotations for this Run"):
        if all_annotations:
            annot_rows = []
            for k, v in sorted(all_annotations.items()):
                annot_rows.append({
                    "Trajectory": k,
                    "Status": f"{status_colors.get(v.get('status', ''), '')} {v.get('status', '')}",
                    "Issue": v.get("issue_type", "No issue"),
                    "Reviewer": v.get("reviewer", ""),
                    "Updated": v.get("updated_at", "")[:16],
                    "Notes": v.get("notes", "")[:50] + ("..." if len(v.get("notes", "")) > 50 else ""),
                })
            df_annot = pd.DataFrame(annot_rows)
            styled_df(df_annot)
            download_csv(df_annot, "annotations.csv")
        else:
            st.info("No annotations yet.")

    st.divider()
    from latentgym.reporting.trajectory_viewer import TrajectoryViewer
    viewer = TrajectoryViewer()
    st.download_button("⬇ Download HTML", viewer.render_html(traj),
                        f"traj_{traj_idx:04d}.html", "text/html")


elif page == "Model Comparison":
    page_header("Model Comparison", "Head-to-head analysis between two models")

    if len(model_names) < 2:
        st.info("Need at least 2 models.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        model_a = st.selectbox("Model A", model_names, index=0)
    with col2:
        model_b = st.selectbox("Model B", model_names, index=min(1, len(model_names) - 1))

    if model_a == model_b:
        st.warning("Select two different models.")
        st.stop()

    from latentgym.reporting.leaderboard import compute_model_comparison
    comp = compute_model_comparison(per_model, model_a, model_b)

    metric_row([
        (f"🟢 {model_a} wins", str(comp["wins_a"])),
        (f"🔴 {model_b} wins", str(comp["wins_b"])),
        ("⚪ Ties", str(comp["ties"])),
        ("Avg Δ (A−B)", f"{comp['avg_delta']:+.4f}"),
    ])

    rows = [
        {"Config": bid, "Δ (A−B)": round(delta, 4),
         "Winner": model_a if delta > 0.001 else (model_b if delta < -0.001 else "tie")}
        for bid, delta in sorted(comp["per_config_delta"].items())
    ]
    if rows:
        styled_df(pd.DataFrame(rows))

    if per_complexity:
        st.subheader("By Complexity")
        crows = []
        for c, md in sorted(per_complexity.items()):
            a = md.get(model_a, {}).get("avg_mean_reward")
            b = md.get(model_b, {}).get("avg_mean_reward")
            if a is not None and b is not None:
                crows.append({"Complexity": c, model_a: round(a, 4), model_b: round(b, 4), "Δ": round(a - b, 4)})
        if crows:
            styled_df(pd.DataFrame(crows))

    if has_comparison_data(data):
        st.subheader("Finetuned vs Base (Transfer Analysis)")
        cd = da_comparisons.get("per_benchmark_id", {})
        if cd:
            cr = []
            for bid, m in sorted(cd.items()):
                cr.append({
                    "Config": bid,
                    "Overall Δ": round(m.get("overall_improvement", 0), 4),
                    "ICL Δ": round(m.get("icl_difference", 0), 4),
                    "F→B": round(m["transfer_f_to_b"], 4) if m.get("transfer_f_to_b") is not None else "—",
                    "B→F": round(m["transfer_b_to_f"], 4) if m.get("transfer_b_to_f") is not None else "—",
                })
            styled_df(pd.DataFrame(cr))


elif page == "Exploration vs Exploitation":
    page_header("Exploration vs Exploitation", "Phase split: first half vs second half of episodes")

    if not detailed:
        st.warning("No phase data.")
        st.stop()

    configs = sorted(detailed.keys())
    envs_in_detailed = sorted({k.split("/")[1] if "/" in k else k for k in configs})
    filter_env = st.selectbox("Filter Environment", ["All"] + envs_in_detailed)

    rows = []
    for key, m in sorted(detailed.items()):
        if filter_env != "All" and filter_env not in key:
            continue
        rows.append({
            "Config": key,
            "Explore Reward": round(m.get("exploration_mean_reward", 0), 4),
            "Exploit Reward": round(m.get("exploitation_mean_reward", 0), 4),
            "Improvement": round(m.get("improvement", 0), 4),
            "Slope": round(m.get("learning_slope", 0), 4),
            "Boundary": m.get("exploration_exploitation_boundary", 0),
        })

    if rows:
        df = pd.DataFrame(rows)
        styled_df(df)
        download_csv(df, "exploration_exploitation.csv")


elif page == "Double-Agent":
    page_header("Double-Agent", "Pre/post switch performance, transfer effects, per-agent breakdown")

    if not da_schedule:
        st.warning("No double-agent data.")
        st.stop()

    # Schedule summary
    st.subheader("Schedule Summary")
    srows = []
    for key, m in sorted(da_schedule.items()):
        srows.append({
            "Schedule": key,
            "Switch": m.get("switch_episode", 0),
            "Pre-Switch": round(m.get("avg_pre_switch_reward", 0), 4),
            "Post-Switch": round(m.get("avg_post_switch_reward", 0), 4),
            "Transfer": round(m.get("avg_transfer_effect", 0), 4),
            "Adaptation": round(m.get("avg_adaptation_speed", 0), 4),
            "# Traj": m.get("n_trajectories", 0),
        })
    styled_df(pd.DataFrame(srows))

    try:
        from latentgym.reporting.plots import (
            plot_pre_post_switch_comparison, plot_transfer_effects,
            plot_double_agent_learning_curve,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_pre_post_switch_comparison(da_schedule))
        with c2:
            st.pyplot(plot_transfer_effects(da_schedule))

        st.subheader("Learning Curve (with switch point)")
        sel = st.selectbox("Schedule", sorted(da_schedule.keys()))
        st.pyplot(plot_double_agent_learning_curve(da_schedule, sel))
    except ImportError:
        pass

    if da_agent:
        st.subheader("Per-Agent Breakdown")
        arows = []
        for key, agents in sorted(da_agent.items()):
            for an, m in sorted(agents.items()):
                arows.append({
                    "Schedule": key, "Agent": an,
                    "Reward": round(m.get("avg_reward", 0), 4),
                    "Success": round(m.get("success_rate", 0), 4),
                    "Turns": round(m.get("avg_turns", 0), 1),
                    "# Ep": m.get("n_episodes", 0),
                })
        styled_df(pd.DataFrame(arows))


elif page == "Custom Comparison":
    page_header("Custom Comparison", "Build custom views: pick axes, filters, and metrics")

    # Parse dimensions from benchmark_ids
    all_envs_set, all_latents_set, all_prompts_set, all_feedbacks_set = set(), set(), set(), set()
    for bid in all_bids:
        parts = bid.split("/")
        all_envs_set.add(parts[0])
        if len(parts) > 1: all_latents_set.add(parts[1])
        if len(parts) > 2: all_prompts_set.add(parts[2])
        if len(parts) > 3: all_feedbacks_set.add(parts[3])

    dims = {
        "Model": sorted(model_names),
        "Environment": sorted(all_envs_set),
        "Latent": sorted(all_latents_set),
        "Prompt": sorted(all_prompts_set) or ["default"],
        "Feedback": sorted(all_feedbacks_set) or ["default"],
        "Complexity": sorted(per_complexity.keys()) if per_complexity else [],
    }
    dim_col = {"Model": "model", "Environment": "env", "Latent": "latent",
               "Prompt": "prompt", "Feedback": "feedback"}

    METRICS = [
        "avg_mean_reward", "avg_improvement", "avg_success_rate",
        "avg_initial_reward", "avg_final_reward", "learning_slope",
        "avg_total_turns", "avg_mean_turns_per_episode",
    ]

    if "panels" not in st.session_state:
        st.session_state.panels = [0]
    if "next_id" not in st.session_state:
        st.session_state.next_id = 1

    for pid in list(st.session_state.panels):
        with st.container():
            st.divider()
            hc, rc = st.columns([9, 1])
            hc.markdown(f"### Comparison #{pid + 1}")
            if rc.button("🗑️", key=f"rm_{pid}"):
                st.session_state.panels.remove(pid)
                st.rerun()

            c1, c2, c3 = st.columns(3)
            with c1:
                axis = st.selectbox("Compare along", list(dims.keys()), key=f"ax_{pid}")
            with c2:
                met = st.selectbox("Metric", METRICS, key=f"met_{pid}",
                                   format_func=lambda x: x.replace("avg_", "").replace("_", " ").title())
            with c3:
                chart = st.selectbox("Chart", ["Bar", "Table", "Multi-metric"], key=f"ch_{pid}")

            # Filters — show all dimensions except the compare axis
            st.markdown("**Filters** (hold fixed):")
            filter_vals = {}
            fcols = st.columns(len(dims) - 1)
            fi = 0
            for dname, dvals in dims.items():
                if dname == axis or not dvals:
                    filter_vals[dname] = dvals
                    continue
                with fcols[fi]:
                    filter_vals[dname] = st.multiselect(dname, dvals, default=dvals, key=f"f_{dname}_{pid}")
                fi += 1

            # Build filtered data
            rows = []
            for model_i, bm_i in per_model.items():
                if model_i not in filter_vals.get("Model", model_names):
                    continue
                for bid_i, m_i in bm_i.items():
                    parts = bid_i.split("/")
                    env_i = parts[0]
                    lat_i = parts[1] if len(parts) > 1 else ""
                    pmt_i = parts[2] if len(parts) > 2 else "default"
                    fb_i = parts[3] if len(parts) > 3 else "default"

                    if env_i not in filter_vals.get("Environment", all_envs_set):
                        continue
                    if lat_i not in filter_vals.get("Latent", all_latents_set):
                        continue
                    if pmt_i not in filter_vals.get("Prompt", all_prompts_set or {"default"}):
                        continue
                    if fb_i not in filter_vals.get("Feedback", all_feedbacks_set or {"default"}):
                        continue

                    val = m_i.get(met)
                    if val is None:
                        continue
                    rows.append({"model": model_i, "env": env_i, "latent": lat_i,
                                 "prompt": pmt_i, "feedback": fb_i, "bid": bid_i, "value": float(val)})

            if not rows:
                st.info("No data matches filters.")
                continue

            df_raw = pd.DataFrame(rows)

            # Resolve group column
            if axis == "Complexity" and per_complexity:
                crows = []
                for c in filter_vals.get("Complexity", dims["Complexity"]):
                    for m in filter_vals.get("Model", model_names):
                        v = per_complexity.get(c, {}).get(m, {}).get(met)
                        if v is not None:
                            crows.append({"complexity": c, "model": m, "value": float(v)})
                if not crows:
                    st.info("No complexity data for filters.")
                    continue
                df_raw = pd.DataFrame(crows)
                gcol = "complexity"
            elif axis in dim_col:
                gcol = dim_col[axis]
            else:
                gcol = "model"

            df_agg = df_raw.groupby(gcol)["value"].agg(["mean", "std", "count"]).reset_index()
            df_agg.columns = [gcol, "Mean", "Std", "Count"]
            df_agg = df_agg.round(4).sort_values("Mean", ascending=False)

            # Render
            if chart == "Table":
                styled_df(df_agg)
            elif chart == "Bar":
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(max(6, len(df_agg) * 0.7), 3.5))
                    colors = ["#4CAF50" if v >= df_agg["Mean"].median() else "#2196F3" for v in df_agg["Mean"]]
                    bars = ax.bar(range(len(df_agg)), df_agg["Mean"], yerr=df_agg["Std"],
                                  color=colors, alpha=0.85, capsize=3)
                    ax.set_xticks(range(len(df_agg)))
                    ax.set_xticklabels(df_agg[gcol], rotation=25, ha="right", fontsize=9)
                    ax.set_ylabel(met.replace("avg_", "").replace("_", " ").title(), fontsize=10)
                    ax.grid(True, alpha=0.2, axis="y")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except ImportError:
                    styled_df(df_agg)
            elif chart == "Multi-metric":
                extra = st.multiselect("Additional metrics",
                                       [m for m in METRICS if m != met], key=f"mm_{pid}",
                                       format_func=lambda x: x.replace("avg_", "").replace("_", " ").title())
                all_mets = [met] + extra
                mrows = []
                for _, r in df_raw.iterrows():
                    for mt in all_mets:
                        v = per_model.get(r["model"], {}).get(r["bid"], {}).get(mt)
                        if v is not None:
                            mrows.append({gcol: r[gcol], "metric": mt.replace("avg_", ""), "value": float(v)})
                if mrows:
                    dfm = pd.DataFrame(mrows).groupby([gcol, "metric"])["value"].mean().reset_index()
                    dfp = dfm.pivot(index=gcol, columns="metric", values="value").round(4)
                    styled_df(dfp.reset_index())
                    try:
                        import matplotlib.pyplot as plt
                        import numpy as np
                        fig, ax = plt.subplots(figsize=(max(8, len(dfp) * 1), 3.5))
                        x = np.arange(len(dfp))
                        w = 0.8 / max(len(all_mets), 1)
                        for j, mt in enumerate([m.replace("avg_", "") for m in all_mets]):
                            if mt in dfp.columns:
                                ax.bar(x + (j - len(all_mets)/2 + 0.5) * w, dfp[mt].values, w, label=mt)
                        ax.set_xticks(x)
                        ax.set_xticklabels(dfp.index, rotation=25, ha="right", fontsize=9)
                        ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.2, axis="y")
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    except ImportError:
                        pass

            st.caption(f"Aggregated from {len(df_raw)} data points across {df_raw[gcol].nunique()} {axis.lower()}(s)")

    st.divider()
    def _add():
        st.session_state.panels.append(st.session_state.next_id)
        st.session_state.next_id += 1
    st.button("➕ Add another comparison", on_click=_add, type="primary")


elif page == "Issues & Suggestions":
    page_header("Issues & Suggestions", "Report issues, suggest improvements, and track feedback")

    import json as _json
    from datetime import datetime

    issues_path = Path(DATA_DIR) / "issues.json"

    def _load_issues() -> list:
        if issues_path.exists():
            with open(issues_path) as f:
                return _json.load(f)
        return []

    def _save_issues(issues: list):
        with open(issues_path, "w") as f:
            _json.dump(issues, f, indent=2)

    all_issues = _load_issues()

    # ── Summary cards ──
    open_count = sum(1 for i in all_issues if i.get("status") == "open")
    resolved_count = sum(1 for i in all_issues if i.get("status") == "resolved")
    total_count = len(all_issues)

    metric_row([
        ("📨 Open", str(open_count)),
        ("✅ Resolved", str(resolved_count)),
        ("📝 Total", str(total_count)),
    ])

    # ── Submit new issue/suggestion ──
    st.subheader("Submit New")

    with st.form("new_issue_form", clear_on_submit=True):
        fc1, fc2 = st.columns(2)
        with fc1:
            issue_category = st.selectbox("Category", [
                "Bug — Environment logic",
                "Bug — Reward computation",
                "Bug — Prompt / feedback text",
                "Bug — Data generation",
                "Bug — Evaluation pipeline",
                "Bug — Reporting / dashboard",
                "Suggestion — New environment",
                "Suggestion — New latent",
                "Suggestion — New metric",
                "Suggestion — UI improvement",
                "Suggestion — Documentation",
                "Question",
                "Other",
            ])
        with fc2:
            priority = st.selectbox("Priority", ["low", "medium", "high", "critical"])

        title = st.text_input("Title", placeholder="Brief summary of the issue or suggestion")

        description = st.text_area(
            "Description",
            placeholder="Detailed description. Include:\n"
                        "- Steps to reproduce (for bugs)\n"
                        "- Expected vs actual behavior\n"
                        "- Which env/latent/model is affected\n"
                        "- Trajectory reference if applicable (model/config/traj_idx)",
            height=150,
        )

        dc1, dc2 = st.columns(2)
        with dc1:
            affected_env = st.selectbox("Affected Environment", ["N/A"] + env_names)
        with dc2:
            submitter = st.text_input("Your Name", placeholder="Who is submitting this")

        submitted = st.form_submit_button("📨 Submit", type="primary")

        if submitted:
            if not title.strip():
                st.error("Title is required.")
            else:
                new_issue = {
                    "id": total_count + 1,
                    "status": "open",
                    "category": issue_category,
                    "priority": priority,
                    "title": title.strip(),
                    "description": description.strip(),
                    "affected_env": affected_env,
                    "submitter": submitter.strip() or "anonymous",
                    "created_at": datetime.now().isoformat(),
                    "resolved_at": None,
                    "resolution": "",
                    "comments": [],
                }
                all_issues.append(new_issue)
                _save_issues(all_issues)
                st.success(f"Issue #{new_issue['id']} submitted!")
                st.rerun()

    # ── Issue list ──
    st.divider()
    st.subheader("All Issues & Suggestions")

    if not all_issues:
        st.info("No issues submitted yet.")
    else:
        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            filter_status = st.selectbox("Filter Status", ["all", "open", "resolved", "wontfix"])
        with fc2:
            filter_priority = st.selectbox("Filter Priority", ["all", "critical", "high", "medium", "low"])
        with fc3:
            filter_cat = st.selectbox("Filter Category", ["all"] + sorted({i["category"] for i in all_issues}))

        # Apply filters
        filtered = all_issues
        if filter_status != "all":
            filtered = [i for i in filtered if i.get("status") == filter_status]
        if filter_priority != "all":
            filtered = [i for i in filtered if i.get("priority") == filter_priority]
        if filter_cat != "all":
            filtered = [i for i in filtered if i.get("category") == filter_cat]

        # Priority icons
        pri_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        status_icon = {"open": "📨", "resolved": "✅", "wontfix": "⏭️"}

        for issue in reversed(filtered):
            iid = issue.get("id", "?")
            istatus = issue.get("status", "open")
            ipri = issue.get("priority", "medium")
            header = (f"{status_icon.get(istatus, '📨')} {pri_icon.get(ipri, '🟡')} "
                      f"**#{iid}: {issue.get('title', 'Untitled')}**")

            with st.expander(header, expanded=(istatus == "open" and ipri in ("critical", "high"))):
                ic1, ic2, ic3, ic4 = st.columns(4)
                ic1.markdown(f"**Category:** {issue.get('category', '')}")
                ic2.markdown(f"**Priority:** {pri_icon.get(ipri, '')} {ipri}")
                ic3.markdown(f"**By:** {issue.get('submitter', 'anonymous')}")
                ic4.markdown(f"**Created:** {issue.get('created_at', '')[:16]}")

                if issue.get("affected_env", "N/A") != "N/A":
                    st.markdown(f"**Affected Env:** `{issue['affected_env']}`")

                if issue.get("description"):
                    st.markdown("**Description:**")
                    st.text(issue["description"])

                if issue.get("resolution"):
                    st.markdown(f"**Resolution:** {issue['resolution']}")

                # Comments
                comments = issue.get("comments", [])
                if comments:
                    st.markdown("**Comments:**")
                    for c in comments:
                        st.markdown(f"> **{c.get('author', '?')}** ({c.get('time', '')[:16]}): {c.get('text', '')}")

                # Actions
                st.markdown("---")
                ac1, ac2, ac3 = st.columns(3)

                with ac1:
                    new_status = st.selectbox(
                        "Status", ["open", "resolved", "wontfix"],
                        index=["open", "resolved", "wontfix"].index(istatus),
                        key=f"is_{iid}",
                    )

                with ac2:
                    resolution_text = st.text_input(
                        "Resolution note",
                        value=issue.get("resolution", ""),
                        placeholder="How was this resolved?",
                        key=f"ir_{iid}",
                    )

                with ac3:
                    comment_text = st.text_input(
                        "Add comment",
                        placeholder="Your comment...",
                        key=f"ic_{iid}",
                    )
                    comment_author = st.text_input("Name", placeholder="Your name", key=f"ica_{iid}")

                if st.button("💾 Update", key=f"iu_{iid}"):
                    # Find and update the issue in the list
                    for idx_i, orig in enumerate(all_issues):
                        if orig.get("id") == iid:
                            all_issues[idx_i]["status"] = new_status
                            if resolution_text:
                                all_issues[idx_i]["resolution"] = resolution_text
                            if new_status == "resolved" and not all_issues[idx_i].get("resolved_at"):
                                all_issues[idx_i]["resolved_at"] = datetime.now().isoformat()
                            if comment_text.strip():
                                all_issues[idx_i].setdefault("comments", []).append({
                                    "author": comment_author.strip() or "anonymous",
                                    "text": comment_text.strip(),
                                    "time": datetime.now().isoformat(),
                                })
                            break
                    _save_issues(all_issues)
                    st.success(f"Issue #{iid} updated!")
                    st.rerun()

        # Export
        st.divider()
        if all_issues:
            export_rows = []
            for i in all_issues:
                export_rows.append({
                    "ID": i.get("id"), "Status": i.get("status"), "Priority": i.get("priority"),
                    "Category": i.get("category"), "Title": i.get("title"),
                    "Env": i.get("affected_env"), "Submitter": i.get("submitter"),
                    "Created": i.get("created_at", "")[:16], "Resolution": i.get("resolution", ""),
                })
            download_csv(pd.DataFrame(export_rows), "issues.csv")
