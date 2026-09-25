# Reporting Module

Computes metrics, generates tables/charts, and renders interactive HTML dashboards from benchmark evaluation results. Fully independent of the Streamlit app — everything works as standalone Python or CLI.

## How It Gets Called

The reporting module is a **standalone Python library** — the CLI is just one of several consumers.

| Caller | How | When |
|--------|-----|------|
| **CLI** (`cli/report.py`) | `python -m latentgym.cli.report --data-dir ...` | After eval, to generate paper-ready outputs |
| **Streamlit app** (`app/app.py`) | `from latentgym.reporting import ...` | Live in the web UI (reads DataStore, renders tables/charts) |
| **Eval orchestrator** | `SingleAgentReport(results).save(...)` | Auto-save results at end of eval run |
| **Your own scripts** | `from latentgym.reporting import render_dashboard` | Custom analysis, Jupyter notebooks, etc. |

The CLI doesn't add any logic — it just parses args and calls the same functions:

```python
# What the CLI does internally:
from latentgym.reporting import DataStore, render_dashboard, make_results_table

data = DataStore.load("results/run_001/")
print(make_results_table(data["metrics"]["per_model"]))
html = render_dashboard(data)
```

You can call reporting from anywhere — Python scripts, notebooks, other tools. The CLI is a convenience wrapper, not a requirement.

## Architecture

```
reporting/
├── data_store.py            # Thin I/O layer (read/write JSON, CSV, trajectories)
├── single_agent_report.py   # Computes + writes all single-agent metrics
├── double_agent_report.py   # Computes + writes all double-agent metrics
├── comparison_report.py     # Computes + writes P_f vs Q_b cross-model comparison
├── tables.py                # Table generators (markdown, LaTeX, CSV)
├── plots.py                 # Chart generators (matplotlib → Figure objects)
├── leaderboard.py           # Leaderboard ranking + model-vs-model comparison
├── dashboard.py             # Interactive HTML dashboard (all tables + charts in one file)
├── trajectory_viewer.py     # Trajectory renderer (single, batch, interactive HTML)
└── __init__.py              # Public API
```

## Pipeline

```
run_eval.py                                  report.py / dashboard / your scripts
───────────────────────────                  ──────────────────────────────────────
1. Eval runners produce BenchmarkResults
2. Report classes compute metrics
3. DataStore writes to output_dir
         ↓
   output_dir/          ← the "dataset" →    4. DataStore.load(output_dir)
   ├── metadata.json                         5. Generate tables, plots, HTML
   ├── trajectories/
   ├── metrics/*.json
   └── tables/*.csv
```

### What `run_eval.py` writes (the input to reporting)

When `run_eval.py` finishes, it calls Report classes which write a **DataStore directory** — this is the intermediate "dataset" that `report.py` reads. There is no separate database or binary format — everything is JSON and CSV files.

```bash
# run_eval.py produces this:
results/run_001/
├── metadata.json               # Run info, model names, benchmark IDs
├── trajectories/               # One JSON per trajectory (full conversation + ground truth)
│   └── gpt-4o/
│       └── bandits__loyal_favorite_0__full_info__standard/
│           ├── traj_0000.json
│           ├── traj_0001.json
│           └── ...
├── metrics/                    # Pre-computed aggregated metrics
│   ├── per_model.json          # model → benchmark_id → 20 metric keys
│   ├── per_env.json            # env → model → averaged metrics
│   ├── per_latent_complexity.json
│   ├── leaderboard.json
│   ├── detailed/
│   │   └── exploration_exploitation.json
│   └── double_agent/           # Only if double-agent eval was run
│       ├── per_schedule.json
│       └── per_agent.json
└── tables/
    ├── main_results.csv
    └── leaderboard.csv

# report.py then reads from that same directory and produces:
paper/
├── tables/                     # Formatted tables (markdown, LaTeX, CSV)
├── plots/                      # Chart images (PNG/PDF/SVG)
├── dashboard.html              # Interactive HTML dashboard
└── trajectory_explorer.html    # Interactive HTML trajectory viewer
```

### How `run_eval.py` calls reporting

`run_eval.py` calls Report classes directly — not the CLI:

```python
# Single-agent eval (in run_eval.py):
results = await orchestrator.run()
report = SingleAgentReport(results)
report.save("results/run_001/")          # writes DataStore directory

# Double-agent eval (in run_eval.py):
results = await orchestrator.run_double_agent(model_a, model_b, switch_ep)
store = DataStore("results/run_001/")
SingleAgentReport(results).save_to(store)
DoubleAgentReport(results, switch_episode=5).save_to(store)
store.write_trajectories(results)
```

Then `report.py` reads from the same directory:

```python
# In report.py (or CLI):
data = DataStore.load("results/run_001/")
# data["metrics"]["per_model"], data["metrics"]["double_agent"], etc.
```

## Quick Start

### Single-agent eval

```python
from latentgym.reporting import SingleAgentReport

report = SingleAgentReport(results)
report.save("results/run_001/")
```

Writes:
```
results/run_001/
├── metadata.json
├── trajectories/...
├── metrics/
│   ├── per_model.json
│   ├── per_env.json
│   ├── per_latent_complexity.json
│   ├── leaderboard.json
│   └── detailed/exploration_exploitation.json
└── tables/
    ├── main_results.csv
    └── leaderboard.csv
```

### Double-agent eval

```python
from latentgym.reporting import DoubleAgentReport

report = DoubleAgentReport(results, switch_episode=5)
report.save("results/run_001/")
```

Writes:
```
metrics/double_agent/
├── per_schedule.json      # pre/post switch, transfer, adaptation
└── per_agent.json         # per-agent reward/turns/success
tables/
├── double_agent_summary.csv
└── per_agent_breakdown.csv
```

### Comparison (P_f vs Q_b)

```python
from latentgym.reporting import ComparisonReport

report = ComparisonReport(results, finetuned_key="P_f", base_key="Q_b", switch_episode=5)
report.save("results/run_001/")
```

Writes:
```
metrics/double_agent/comparisons.json
```

### Layering reports on the same directory

```python
from latentgym.reporting import DataStore, SingleAgentReport, DoubleAgentReport, ComparisonReport

store = DataStore("results/run_001")
SingleAgentReport(single_results).save_to(store)
DoubleAgentReport(double_results, switch_episode=5).save_to(store)
ComparisonReport(all_results).save_to(store)
```

### Generate interactive HTML dashboard

```python
from latentgym.reporting import render_dashboard_from_dir

html = render_dashboard_from_dir("results/run_001/")
with open("dashboard.html", "w") as f:
    f.write(html)
# Open dashboard.html in any browser
```

### Generate interactive trajectory explorer

```python
from latentgym.reporting.trajectory_viewer import TrajectoryViewer

viewer = TrajectoryViewer()
html = viewer.render_interactive_html_from_dir("results/run_001/")
with open("trajectories.html", "w") as f:
    f.write(html)
# Open trajectories.html in any browser
```

## Output Directory Structure (fully populated)

```
output_dir/
├── metadata.json
├── trajectories/
│   └── <model_name>/
│       └── <benchmark_id>/
│           └── traj_0000.json ...
├── metrics/
│   ├── per_model.json                       # SingleAgentReport
│   ├── per_env.json                         # SingleAgentReport
│   ├── per_latent_complexity.json           # SingleAgentReport
│   ├── leaderboard.json                     # SingleAgentReport
│   ├── detailed/
│   │   └── exploration_exploitation.json    # SingleAgentReport
│   └── double_agent/
│       ├── per_schedule.json                # DoubleAgentReport
│       ├── per_agent.json                   # DoubleAgentReport
│       └── comparisons.json                 # ComparisonReport
└── tables/
    ├── main_results.csv                     # SingleAgentReport
    ├── leaderboard.csv                      # SingleAgentReport
    ├── double_agent_summary.csv             # DoubleAgentReport
    └── per_agent_breakdown.csv              # DoubleAgentReport
```

## Reading Results Back

```python
from latentgym.reporting import DataStore

data = DataStore.load("results/run_001/")

# Single-agent metrics
data["metrics"]["per_model"]                          # model → bid → 20 metric keys
data["metrics"]["per_env"]                            # env → model → averaged metrics
data["metrics"]["leaderboard"]                        # ranked list
data["metrics"]["detailed"]["exploration_exploitation"]

# Double-agent metrics
data["metrics"]["double_agent"]["per_schedule"]       # pre/post switch, transfer
data["metrics"]["double_agent"]["per_agent"]          # per-agent breakdown
data["metrics"]["double_agent"]["comparisons"]        # P_f vs Q_b

# Trajectory files (lazy index, not loaded)
data["trajectory_index"]                              # model → bid → [file paths]

# Load individual trajectory
traj = DataStore.load_trajectory(data["trajectory_index"]["gpt-4o"]["bandits/..."][0])
```

## Metrics Reference

### Single-Agent (per_model.json — 20 keys per config)

| Metric | Description |
|--------|-------------|
| `avg_trajectory_reward` | Mean cumulative reward across trajectories |
| `std_trajectory_reward` | Std of cumulative reward |
| `min_trajectory_reward` | Min cumulative reward |
| `max_trajectory_reward` | Max cumulative reward |
| `avg_initial_reward` | Mean first-episode reward |
| `avg_final_reward` | Mean last-episode reward |
| `avg_improvement` | Mean (last - first episode reward) |
| `std_improvement` | Std of improvement |
| `avg_cumulative_reward` | Same as avg_trajectory_reward |
| `avg_mean_reward` | Mean per-episode reward (total / n_episodes) |
| `avg_success_rate` | Fraction of episodes with reward >= threshold |
| `avg_total_turns` | Mean total turns per trajectory |
| `std_total_turns` | Std of total turns |
| `avg_mean_turns_per_episode` | Mean turns per episode |
| `per_episode_avg_rewards` | List: avg reward at each episode position |
| `per_episode_std_rewards` | List: std reward at each episode position |
| `per_episode_avg_turns` | List: avg turns at each episode position |
| `learning_slope` | improvement / (n_episodes - 1) |
| `n_trajectories` | Number of trajectories |
| `n_episodes` | Episodes per trajectory |

### Detailed (exploration_exploitation.json — 15 keys per config)

| Metric | Description |
|--------|-------------|
| `exploration_mean_reward` | Mean reward in first half of episodes |
| `exploration_std_reward` | Std reward in first half |
| `exploration_mean_turns` | Mean turns in first half |
| `exploitation_mean_reward` | Mean reward in second half of episodes |
| `exploitation_std_reward` | Std reward in second half |
| `exploitation_mean_turns` | Mean turns in second half |
| `exploration_exploitation_boundary` | Episode index splitting the two phases |

### Double-Agent (per_schedule.json — 17 keys per schedule)

| Metric | Description |
|--------|-------------|
| `switch_episode` | Episode where agent switch occurred |
| `avg_pre_switch_reward` | Mean reward before switch |
| `std_pre_switch_reward` | Std before switch |
| `avg_post_switch_reward` | Mean reward after switch |
| `std_post_switch_reward` | Std after switch |
| `avg_transfer_effect` | Mean (post - pre reward) |
| `std_transfer_effect` | Std of transfer effect |
| `avg_adaptation_speed` | Reward change in first 2 episodes after switch |
| `per_episode_avg_rewards` | List: per-episode reward curve |
| `per_episode_std_rewards` | List: per-episode std |
| `per_episode_avg_turns` | List: per-episode turns |
| `per_agent` | Nested: agent_name → {avg_reward, std_reward, success_rate, avg_turns, n_episodes} |

### Comparison (comparisons.json — 7 keys per benchmark_id)

| Metric | Description |
|--------|-------------|
| `overall_improvement` | Finetuned final reward - base final reward |
| `initial_prior_improvement` | Finetuned initial - base initial |
| `icl_difference` | Finetuned ICL - base ICL |
| `transfer_f_to_b` | P_f→Q_b trajectory reward - Q_b-only reward |
| `transfer_b_to_f` | Q_b→P_f trajectory reward - P_f-only reward |
| `turn_efficiency_difference` | Base turns - finetuned turns |
| `per_config` | Full 20-key metric dict per schedule |

## Table Generators

All support `fmt="markdown"`, `fmt="latex"`, `fmt="csv"`:

```python
from latentgym.reporting import (
    make_results_table,          # model × config results
    make_leaderboard_table,      # ranked leaderboard
    make_per_env_table,          # env × model averages
    make_complexity_table,       # complexity × model averages
    make_double_agent_summary_table,   # pre/post switch + transfer
    make_per_agent_table,              # per-agent breakdown
    make_comparison_table,             # P_f vs Q_b per config
    make_exploration_exploitation_table,  # exploration vs exploitation phases
)
```

## Plot Functions

All return `matplotlib.Figure` objects (caller saves/shows):

```python
from latentgym.reporting.plots import (
    # Single-agent
    plot_learning_curves,          # per-episode reward curve
    plot_env_bar_chart,            # models grouped by env
    plot_heatmap,                  # model × config heatmap
    plot_improvement_distribution, # improvement bar chart

    # Double-agent
    plot_double_agent_learning_curve,  # curve with switch line
    plot_pre_post_switch_comparison,   # pre vs post bar chart
    plot_per_agent_comparison,         # per-agent metric bars
    plot_transfer_effects,             # transfer effect bars

    # Batch
    save_all_plots,                # generate + save all plots to directory
)
```

## Interactive HTML Outputs

### Dashboard (`dashboard.py`)

Single HTML file with all tables, charts (as base64 PNGs), and leaderboard. 6 tabs: Leaderboard, Results Table, Per-Environment, By Complexity, Learning Curves, Double-Agent. Sortable columns, environment filter dropdown. No server required.

### Trajectory Explorer (`trajectory_viewer.py`)

Single HTML file with all trajectories embedded as JSON. Filter dropdowns for model, env, latent, prompt, feedback. Prev/Next navigation. Shows: header with metrics (including reasoning badge), env params, episode table with ground truth, expandable ground truth details, color-coded conversation with reasoning traces. No server required.

Reasoning/thinking is shown as collapsible purple blocks before each assistant message (when available). A badge in the header shows how many turns had reasoning (e.g., "3/10 Reasoning"). Reasoning content is never mixed with the conversation — it's clearly labeled as internal thinking not shown to the environment.

## CLI Usage

The reporting module is wired into `latentgym/cli/report.py`. All commands read from a DataStore output directory (`--data-dir`).

### Generate everything (default)

```bash
python -m latentgym.cli.report --data-dir results/run_001/ --output paper/
```

Produces:
```
paper/
├── tables/
│   ├── main_results.md / .tex / .csv
│   ├── leaderboard.md / .csv
│   ├── per_env.md
│   ├── per_complexity.md
│   ├── exploration_exploitation.md
│   ├── double_agent_summary.md / .csv      (if double-agent data present)
│   ├── per_agent_breakdown.md              (if double-agent data present)
│   └── comparison.md                       (if comparison data present)
├── plots/
│   ├── bar_env_avg_reward.png
│   ├── learning_curve_*.png                (one per config)
│   ├── heatmap_*.png                       (one per env)
│   ├── bar_pre_post_switch.png             (if double-agent)
│   ├── bar_transfer_effects.png            (if double-agent)
│   ├── da_learning_curve_*.png             (if double-agent)
│   ├── bar_per_agent_reward.png            (if double-agent)
│   └── bar_per_agent_success.png           (if double-agent)
├── dashboard.html                          (interactive, open in browser)
└── trajectory_explorer.html                (interactive, open in browser)
```

### Individual commands

```bash
# Tables only (markdown + LaTeX + CSV)
python -m latentgym.cli.report --data-dir results/ --tables-only --output paper/

# Plots only (PNG by default, or --fmt pdf / --fmt svg)
python -m latentgym.cli.report --data-dir results/ --plots-only --output paper/ --fmt pdf

# Interactive HTML dashboard (single file)
python -m latentgym.cli.report --data-dir results/ --dashboard --output paper/

# Interactive trajectory explorer (single file)
python -m latentgym.cli.report --data-dir results/ --trajectories --output paper/

# Print leaderboard to stdout
python -m latentgym.cli.report --data-dir results/ --leaderboard
python -m latentgym.cli.report --data-dir results/ --leaderboard --env bandits

# View a specific trajectory (text to stdout, or HTML to file)
python -m latentgym.cli.report --trajectory results/trajectories/.../traj_0000.json
python -m latentgym.cli.report --trajectory results/trajectories/.../traj_0000.json --html traj.html

# Compare two models head-to-head
python -m latentgym.cli.report --data-dir results/ --compare gpt-4o claude-3.5
```

### What gets called automatically

When you run `--data-dir results/ --output paper/` (no flags), the CLI calls all four generators:
1. `_write_tables()` — all table formats for all metric types (single + double + comparison)
2. `_write_plots()` — all charts via `save_all_plots()` (single + double-agent if present)
3. `_write_dashboard()` — `render_dashboard_from_dir()` → `dashboard.html`
4. `_write_trajectory_explorer()` — `render_interactive_html_from_dir()` → `trajectory_explorer.html`

### Typical workflow

```bash
# 1. Run eval
python -m latentgym.cli.run_eval single \
    --models openai:gpt-4o openai:gpt-4o-mini \
    --envs bandits wordle \
    --output results/run_001/

# 2. Generate all reports
python -m latentgym.cli.report --data-dir results/run_001/ --output paper/

# 3. Open dashboard in browser
open paper/dashboard.html             # Mac
xdg-open paper/dashboard.html         # Linux

# 4. Open trajectory explorer
open paper/trajectory_explorer.html
```

## Dependencies

- **Required**: None (tables, reports, data_store work with stdlib only)
- **Optional**: `matplotlib` + `numpy` for charts and chart embedding in dashboard
- **Not required**: Streamlit (that's in `latentgym/app/`, a separate consumer)
