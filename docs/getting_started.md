# Getting Started

This guide walks you through installing LatentGym and running your first evaluation.

## Installation

### Prerequisites

- Python 3.12.7 (pinned via `skyrl-train/pyproject.toml`)
- [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- ~5 GB of scratch storage for the venv (on HPC, put it on `$SCRATCH`, not `$HOME`)

### 1. Clone

```bash
git clone <anonymized-repo-url> LatentGym
cd LatentGym
```

### 2. Configure your environment

```bash
cp .env.example .env
$EDITOR .env
```

**Required:** `VENV_DIR` — absolute path where the uv venv will live (e.g. `$SCRATCH/latentgym`).

**Set whichever of these you need:**
- `OPENROUTER_API_KEY` — primary path for evaluating models (used with specs like `openrouter/openai:gpt-4o`)
- `OPENAI_API_KEY`, `CLAUDE_API_KEY`, `GEMINI_API_KEY` — for direct provider access
- `HF_TOKEN`, `WANDB_API_KEY` — Hugging Face downloads, W&B logging

### 3. Install

```bash
bash setup.sh
```

Creates a Python 3.12.7 venv at `$VENV_DIR` and runs `uv sync --active --extra vllm` from `skyrl-train/`.

### 4. Activate in every new shell

```bash
source project_config.sh
source $VENV_DIR/bin/activate
```

### 5. Sanity check

```bash
python -c "import benchmark; print('OK')"
```

## Repo Structure

### Top-level layout

```
LatentGym/
├── latentgym/                  Our research code (envs, eval, training, reporting, app)
├── skyrl-train/                Vendored fork of SkyRL-Train v0.2.0 (RL training framework)
├── skyrl-gym/                  Vendored fork of SkyRL-gym (env registration)
├── TextArena/                  Vendored copy of TextArena with multi-episode adaptations
├── docs/                       This guide and other top-level documentation
├── assets/                     Images / figures for the README
├── README.md, LICENSE
├── .env.example, project_config.sh, setup.sh, .gitignore
```

### Benchmark module layout

```
latentgym/
├── core/                       Core abstractions (registry, env config, make_env)
├── envs/                       7 environments (bandits, wordle, hangman, mastermind,
│                               secretary, wordladder, number_guessing) + adding_environments.md
├── eval/                       Evaluation pipeline (single + double agent runners)
├── training/                   Training scripts + multi-GPU / FSDP guide
├── data/                       Data generation (trajectory JSONs + parquets)
├── cli/                        CLI: generate_data, run_eval, report
├── reporting/                  Tables, plots, dashboard, trajectory viewer
├── app/                        Streamlit app (11 interactive pages)
├── configs/                    Optional YAML eval-suite configs
├── register_skyrl.py           Registers benchmark envs with skyrl_gym
├── LOCAL_EVAL_GUIDE.md         Focused walkthrough for local-model eval
└── __init__.py
```

### How the pieces connect

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           latentgym/                                    │
│                                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐              │
│  │  core/   │   │  envs/   │   │  eval/   │   │  data/   │              │
│  │ ABCs     │◄──│ bandits  │──►│ runners  │◄──│ parquet  │              │
│  │ registry │   │ wordle   │   │ metrics  │   │ generate │              │
│  │ env mgmt │   │ hangman… │   │ results  │   │          │              │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘              │
│       ▲                              ▲              │                   │
│       │              ┌───────────────┐              │                   │
│       └──────────────│register_skyrl │◄─────────────┘                   │
│                      └───────────────┘  registers envs with skyrl_gym   │
└─────────────────────────────────────────────────────────────────────────┘
           │                                      │
           ▼                                      ▼
    ┌──────────────┐                     ┌──────────────────┐
    │ skyrl-gym    │                     │ skyrl-train      │
    │ BaseTextEnv  │                     │ SkyRLGymGenerator│
    │ registration │                     │ PromptDataset    │
    └──────────────┘                     └──────────────────┘
```

## Common Commands

```bash
# List available latents for an env
python -m latentgym.cli.generate_data list --env bandits

# Generate eval data (with dry-run preview)
python -m latentgym.cli.generate_data eval --env bandits --dry-run
python -m latentgym.cli.generate_data eval --env bandits --complexity easy --output latentgym/data/eval/

# Generate training data
python -m latentgym.cli.generate_data train --env number_guessing --latent set_of_3 \
    --n-trajectories 500 --num-episodes 10 --seed 10000 --output latentgym/data/train/

# Pre-filter candidate pools (one-time, speeds up filter-based envs)
python -m latentgym.cli.generate_data filter-pool --env wordle --output latentgym/data/pools/

# Run single-agent evaluation
python -m latentgym.cli.run_eval single --models openrouter/openai:gpt-4o \
    --env bandits --latent loyal_favorite_0 \
    --trajectory-dir latentgym/data/eval/ --output latentgym/results/

# Run double-agent evaluation (switch model at episode K)
python -m latentgym.cli.run_eval double --model-a openrouter/openai:gpt-4o \
    --model-b openrouter/openai:o4-mini --switch-episode 5 \
    --env bandits --latent loyal_favorite_0 \
    --trajectory-dir latentgym/data/eval/ --output latentgym/results/

# Generate reports (tables + plots + dashboard + trajectory explorer)
python -m latentgym.cli.report --data-dir latentgym/results/ --output paper/
python -m latentgym.cli.report --data-dir latentgym/results/ --leaderboard

# Interactive Streamlit app (11 pages)
streamlit run latentgym/app/app.py -- --data-dir latentgym/results/

# Train a model (see latentgym/training/README.md for full setup)
bash latentgym/training/train_minimal.sh    # single GPU
bash latentgym/training/train_fsdp.sh       # 4-GPU FSDP
```

## End-to-End Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 1: DEFINE                                                         │
│   FullyDefinedEnv(env_name="bandits", latent_id="loyal_favorite_0",    │
│                   prompt_id="full_info", feedback_id="standard", ...)  │
└──────────────┬─────────────────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 2: GENERATE TRAJECTORIES                                          │
│   trajectory_generator(latent, seed) ──► traj_000.json, manifest.json  │
│     generator_fn ──► {"ground_truth": {"red": 0.7, ...}}               │
│     filter_fn    ──► {"target_word": "crane"}                          │
└──────────────┬─────────────────────────────┬───────────────────────────┘
               │                             │
               ▼                             ▼
┌──────────────────────────┐   ┌──────────────────────────────────┐
│ STEP 3a: EVAL            │   │ STEP 3b: TRAIN                   │
│                          │   │                                  │
│ Orchestrator             │   │ generate_parquet()               │
│   loads manifest         │   │   wraps JSONs + varies           │
│   for each traj_file:    │   │   prompt × feedback × reward     │
│     make_env(fd, path)   │   │                                  │
│     runner.run()         │   │ SkyRL reads parquet              │
│     → TrajectoryResult   │   │   → skyrl_gym.make() → train     │
└──────────────┬───────────┘   └──────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 4: ANALYZE                                                        │
│   compute_single_agent_metrics(results) ──► tables, plots              │
│   compute_double_agent_metrics(results) ──► transfer effects           │
│   BenchmarkResults.save("results.json") ──► persistent storage         │
│   WandbTracker ──► optional cloud logging                              │
└────────────────────────────────────────────────────────────────────────┘
```

## Quick Start: Run an Eval in 3 Steps

### Step 1 — Generate eval trajectories

```bash
python -m latentgym.cli.generate_data eval \
    --env bandits \
    --latent loyal_favorite_0 \
    --n-trajectories 10 \
    --num-episodes 10 \
    --output data/eval/
```

This writes `data/eval/bandits_loyal_favorite_0/` containing trajectory JSON files.

Preview what would be generated without writing files:
```bash
python -m latentgym.cli.generate_data eval \
    --env bandits --latent loyal_favorite_0 \
    --n-trajectories 10 --num-episodes 10 \
    --dry-run
```

Use `--complexity` to select latents by difficulty tier:
```bash
python -m latentgym.cli.generate_data eval \
    --env bandits --complexity easy \
    --n-trajectories 10 --output data/eval/
```

Override env-specific parameters via `--env-param`:
```bash
python -m latentgym.cli.generate_data eval \
    --env bandits --latent loyal_favorite_0 \
    --env-param max_turns_per_episode=15 \
    --env-param n_buttons=8 \
    --n-trajectories 10 --output data/eval/
```

### Step 2 — Run evaluation

```bash
python -m latentgym.cli.run_eval single \
    --models openai:gpt-4o-mini \
    --env bandits \
    --latent loyal_favorite_0 \
    --prompt full_info \
    --feedback standard \
    --num-episodes 10 \
    --n-trajectories 10 \
    --trajectory-dir data/eval/ \
    --output results/quick_run/
```

Set your API key first:
```bash
export OPENAI_API_KEY=sk-...
```

### Step 3 — View results

```bash
# Print leaderboard
python -m latentgym.cli.report --data-dir results/quick_run/ --leaderboard

# Generate all tables and plots
python -m latentgym.cli.report --data-dir results/quick_run/ --output paper/

# Generate interactive dashboard and trajectory explorer
python -m latentgym.cli.report --data-dir results/quick_run/ --dashboard --output paper/
python -m latentgym.cli.report --data-dir results/quick_run/ --trajectories --output paper/

# Launch interactive Streamlit app (11 pages)
streamlit run latentgym/app/app.py -- --data-dir results/quick_run/
```

The Streamlit app includes 11 pages: Leaderboard, Per-Environment, Learning Curves,
Trajectory Viewer, Comparisons, Dashboard, Latent Analysis, Prompt Ablation,
Feedback Ablation, Double-Agent Results, and Export.

## Available Environments

| Environment     | # Latents | Type             | Dependency     |
|-----------------|-----------|------------------|----------------|
| `bandits`       | 28        | Generator-based  | None           |
| `wordle`        | 165       | Filter-based     | TextArena      |
| `hangman`       | 105       | Filter-based     | TextArena      |
| `mastermind`    | 54        | Filter-based     | TextArena      |
| `secretary`     | 30        | Generator-based  | None           |
| `wordladder`    | 53        | Filter-based     | TextArena      |
| `number_guessing` | 7       | Generator-based  | None           |

List available latents for any env:
```bash
python -m latentgym.cli.generate_data list --env bandits
```

## Model Specs

The `--models` argument accepts any `provider:model` string:

```
openai:gpt-4o
openai:gpt-4o-mini
anthropic:claude-sonnet-4-6
anthropic:claude-haiku-4-5-20251001
google/gemini-pro
vllm:http://localhost:8000    # local vLLM server
mock:random                   # deterministic mock (for testing)
```

## Evaluating Multiple Models

```bash
python -m latentgym.cli.run_eval single \
    --models openai:gpt-4o openai:gpt-4o-mini anthropic:claude-haiku-4-5-20251001 \
    --env bandits --latent loyal_favorite_0 \
    --prompt full_info --feedback standard \
    --n-trajectories 20 \
    --trajectory-dir data/eval/ \
    --output results/multi_model/
```

## Evaluating All Latents for an Env

```bash
# Generate all latents (use --val-ratio to split train/val)
python -m latentgym.cli.generate_data eval \
    --env wordle --n-trajectories 50 --val-ratio 0.2 --output data/eval/

# Run eval against all generated configs
python -m latentgym.cli.run_eval single \
    --models openai:gpt-4o \
    --config configs/eval_suites/full.yaml \
    --trajectory-dir data/eval/ \
    --output results/full_wordle/
```

## Next Steps

- [Evaluation guide](../latentgym/eval/README.md) — single/double-agent, OpenRouter/local models, reports, dashboard
- [Training guide](../latentgym/training/README.md) — single-GPU + FSDP, parquet format, adding algorithms
- [Environments](../latentgym/envs/README.md) — 5 axes, 7 envs, per-env pages
- [Adding a new environment](../latentgym/envs/adding_environments.md)
