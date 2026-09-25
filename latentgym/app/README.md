# Streamlit App

Interactive web dashboard for exploring benchmark results. Provides the same data as the static HTML dashboard (`reporting/dashboard.py`) but with full interactivity — live filtering, sorting, custom comparisons, trajectory annotation, and issue tracking.

## Files

```
app/
├── app.py            # Main Streamlit app (all 11 pages)
├── data_loader.py    # Cached data loading + helper functions
└── pages/            # (reserved for future multi-file Streamlit pages)
```

### app.py

Single-file Streamlit app with 11 pages. All pages read from the same `DataStore.load()` output cached with 60s TTL.

### data_loader.py

Thin wrapper around `DataStore.load()` with Streamlit caching. Also provides helper functions: `get_model_names()`, `get_env_names()`, `get_benchmark_ids()`, `get_trajectory_paths()`, `scan_run_dirs()`, `has_double_agent_data()`, `has_detailed_data()`, `has_comparison_data()`.

## How to Run

```bash
# Basic — point at a DataStore output directory
streamlit run latentgym/app/app.py -- --data-dir results/run_001/

# With run selector — scans for multiple runs in a directory
streamlit run latentgym/app/app.py -- --data-dir results/ --scan
```

The `--` separator is required — Streamlit args go before it, app args after.

### Prerequisites

```bash
pip install streamlit pandas
pip install matplotlib numpy   # optional: for charts
```

## Data Source

The app reads from a **DataStore output directory** — the same directory produced by `run_eval` or Report classes. It does NOT read YAML configs, parquets, or trajectory generator output directly.

```
results/run_001/         ← this is what --data-dir points to
├── metadata.json
├── metrics/
│   ├── per_model.json
│   ├── per_env.json
│   ├── leaderboard.json
│   ├── detailed/
│   └── double_agent/
├── tables/
├── trajectories/
├── annotations.json     ← created by trajectory viewer (review/annotation)
└── issues.json          ← created by issues page
```

### Auto-refresh

Data is cached with `@st.cache_data(ttl=60)` — the app re-reads the DataStore every 60 seconds. If `run_eval` is still running, new results appear within a minute. Refresh the browser to see updates immediately.

## Pages

| Page | Icon | What it shows |
|------|------|---------------|
| **Leaderboard** | 🏆 | Ranked model table + per-env breakdown + bar chart |
| **Results Table** | 📋 | Full model × config table with env/model/sort filters |
| **Per-Environment** | 🌍 | Single env, all models, selectable metric + chart |
| **By Complexity** | 📊 | Easy/medium/hard/very_hard breakdown + grouped bar |
| **Learning Curves** | 📈 | Per-episode reward/turns progression |
| **Trajectory Viewer** | 🔍 | Full trajectory: metadata, ground truth, conversation, review/annotation |
| **Model Comparison** | ⚔️ | Head-to-head A vs B + P_f vs Q_b transfer analysis |
| **Custom Comparison** | 🔧 | Build custom views: pick axes, filters, metrics, add/remove panels |
| **Exploration vs Exploitation** | 🔬 | Phase split metrics (only if data exists) |
| **Double-Agent** | 🤝 | Pre/post switch, transfer effects, per-agent (only if data exists) |
| **Issues & Suggestions** | 💡 | Submit/track bugs, suggestions, and feedback |

Pages marked "only if data exists" appear in the sidebar only when the corresponding metrics are present in the DataStore.

## Key Features

### Every page
- Metric cards at the top (key stats)
- Download CSV button on tables
- Consistent styling and layout

### Results Table
- Filter by environment and model
- Sort by any metric column
- Shows all dimensions: model, env, latent, prompt, feedback

### Trajectory Viewer
- Full metadata panel (model, env, latent, prompt, feedback, seed, env_params)
- Episode table with ground truth column + agent column (double-agent)
- Expandable ground truth details (JSON per episode)
- Color-coded conversation (system=orange, assistant=blue, user=grey)
- **Reasoning traces** — when models produce thinking/reasoning (Anthropic thinking, Gemini thoughts, OpenRouter reasoning, vLLM reasoning), it's shown as expandable blocks before each assistant message. Header shows "N/M turns with reasoning".
- **Review & Annotation** — set status (reviewed/flagged/approved/rejected), report issues (9 categories), flag specific episodes, add notes. Persists to `annotations.json`.
- Download trajectory as standalone HTML

### Custom Comparison
- **Compare along**: Model, Environment, Latent, Prompt, Feedback, or Complexity
- **Hold fixed**: Multi-select filters for every other dimension
- **Metrics**: Any of 8 metrics (reward, improvement, success, turns, etc.)
- **Chart types**: Bar chart (with error bars), Table, Grouped bar (multi-metric)
- **Dynamic panels**: Add/remove comparison views with ➕/🗑️ buttons
- Each panel has independent configuration

### Issues & Suggestions
- Submit with category (9 bug types + 5 suggestion types), priority, description
- Per-env affected environment tag
- Comment threads per issue
- Status tracking (open/resolved/wontfix)
- Filter by status, priority, category
- Export all issues as CSV
- Persists to `issues.json`

## Sidebar

- Run info (name, date, model/env/config counts)
- Run selector (with `--scan` flag)
- Page navigation with icons
- Download buttons: Dashboard HTML, Trajectory Explorer HTML

## Persistence

The app writes two files to the DataStore directory:

| File | Created by | Content |
|------|-----------|---------|
| `annotations.json` | Trajectory Viewer → Save Annotation | Per-trajectory review status, issues, notes, episode flags |
| `issues.json` | Issues & Suggestions → Submit | Bug reports, suggestions, comments |

Both are plain JSON — readable without the app, and safe to check into git.

## Relationship to reporting/

The Streamlit app is a **consumer** of the reporting module, not a replacement:

```
reporting/ (Python library)
    ├── DataStore.load()        ← app reads this
    ├── plots.py                ← app calls plot functions
    ├── leaderboard.py          ← app calls compute_model_comparison
    ├── dashboard.py            ← app generates downloadable HTML
    └── trajectory_viewer.py    ← app generates downloadable HTML

app/ (Streamlit frontend)
    ├── Reads from DataStore
    ├── Calls reporting functions for charts/comparisons
    ├── Adds interactivity (filters, dynamic panels, annotations)
    └── Writes annotations.json + issues.json (app-only features)
```

The static HTML outputs (`dashboard.html`, `trajectory_explorer.html`) are generated by `reporting/` and can be downloaded from the app's sidebar. They work without Streamlit — shareable as standalone files.

## Alternatives

| Option | When to use |
|--------|------------|
| **Streamlit app** (this) | Interactive exploration, annotation, live during eval |
| **Static HTML dashboard** (`reporting/dashboard.py`) | Sharing results, archiving, no server needed |
| **Static trajectory explorer** (`reporting/trajectory_viewer.py`) | Sharing trajectories, offline viewing |
| **CLI reports** (`cli/report.py`) | Generating paper-ready tables/plots/LaTeX |
| **Python API** (`from latentgym.reporting import ...`) | Custom analysis in scripts/notebooks |

All read from the same DataStore directory.
