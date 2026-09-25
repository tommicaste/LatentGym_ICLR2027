# CLI Module

Command-line interface for the benchmark: data generation, evaluation, and reporting.

## How to Run

All CLI commands are run as Python modules from the repository root (`skyrl-train/`):

```bash
cd skyrl-train

# Set PYTHONPATH if not installed as a package
export PYTHONPATH=.:$PYTHONPATH

# Or if using a virtualenv
source .venv/bin/activate

# General pattern:
python -m latentgym.cli.<module> <subcommand> [args]
```

The three entry points:

```bash
python -m latentgym.cli.generate_data <subcommand>   # list / eval / parquet / train
python -m latentgym.cli.run_eval <subcommand>         # single / double
python -m latentgym.cli.report [flags]                # --tables-only / --dashboard / etc.
```

### Prerequisites

```bash
# Required for all commands
pip install pandas pyarrow    # parquet generation

# Required for run_eval (depending on model provider)
pip install litellm           # OpenAI, Anthropic, Google API models
pip install vllm              # local vLLM models

# Required for report --plots-only and dashboard charts
pip install matplotlib numpy

# Required for report --dashboard / --trajectories (optional, for Streamlit app)
pip install streamlit

# TextArena (required for trajectory generation and evaluation)
pip install textarena
```

### Quick test (no API keys needed)

```bash
# List environments
python -m latentgym.cli.generate_data list

# Dry run to see what would be generated
python -m latentgym.cli.generate_data eval --env bandits --complexity easy --dry-run --output /tmp/test/

# Dry run to see what would be evaluated
python -m latentgym.cli.run_eval single --models mock:random --env bandits --latent loyal_favorite_0 --dry-run --output /tmp/test/
```

## Files

```
cli/
├── generate_data.py   # Generate trajectory JSONs + SkyRL parquets
├── run_eval.py        # Run single-agent or double-agent evaluation
├── report.py          # Generate reports (tables, plots, dashboard, trajectory viewer)
└── __init__.py
```

### generate_data.py

Generates benchmark datasets in two formats:
- **Trajectory JSONs** — ground truth per episode (target words, button probabilities, etc.)
- **SkyRL parquets** — wrappers around JSONs with prompt/feedback/reward config for training

Subcommands:
- `list` — list registered environments, latents, prompts, feedbacks
- `eval` — generate JSONs + parquets for all (env, latent) pairs with train/val split
- `parquet` — convert existing JSONs to parquets (standalone)
- `train` — shortcut: generate training JSONs + parquets in one command
- `filter-pool` — pre-filter candidate pools by latent (one-time, speeds up trajectory generation)

### run_eval.py

Runs models against benchmark environments and saves results.

Subcommands:
- `single` — single-agent evaluation (one model plays all episodes)
- `double` — double-agent evaluation (model A plays first K episodes, model B plays the rest)

Saves results via Report classes → DataStore directory (JSONs, metrics, tables).

### report.py

Generates reports from evaluation results stored in a DataStore directory.

Actions:
- Default (no flag) — generate everything: tables + plots + dashboard + trajectory explorer
- `--tables-only` — markdown, LaTeX, CSV tables
- `--plots-only` — matplotlib charts (PNG/PDF/SVG)
- `--dashboard` — interactive HTML dashboard
- `--trajectories` — interactive HTML trajectory explorer
- `--leaderboard` — print leaderboard to stdout
- `--compare MODEL_A MODEL_B` — head-to-head comparison
- `--trajectory PATH` — render a single trajectory

## Data Generation Pipeline

### Pipeline flow (Stage 0 is optional but recommended for filter-based envs)

```
Stage 1: Trajectory JSONs          Stage 2: Parquets               Stage 3: Consume
────────────────────────           ──────────────────               ────────────────

generate_data eval                 (generated automatically)
    │                                  │
    ├─ train/                          ├─ train/parquets/
    │   ├─ manifest.json               │   ├─ full_info_standard_cumulative.parquet
    │   ├─ traj_000.json               │   ├─ no_info_standard_terminal.parquet
    │   └─ ...                         │   └─ ... (P × F × R per combo)
    │                                  │
    └─ val/                            └─ val/parquets/
        ├─ manifest.json                   ├─ full_info_standard.parquet
        ├─ traj_000.json                   ├─ no_info_with_stats.parquet
        └─ ...                             └─ ... (P × F, reward=per_episode)

                                   Train parquets ──→ SkyRL training
                                   Val parquets   ──→ SkyRL eval (LocalRunner)
                                   Val JSONs      ──→ Benchmark eval (APIRunner)
```

### What varies where

```
Trajectory JSONs (ground truth — fixed at generation time):
    env, latent, num_episodes, seed → different episode configs

Parquets (eval-time config — same JSONs, different parquet rows):
    prompt_id, feedback_id, reward_type → different rows pointing to same JSONs

Train parquets: one file per (prompt, feedback, reward) combo
Val parquets:   one file per (prompt, feedback), reward fixed to per_episode
```

### Defaults (when flags are omitted)

| Flag | Default |
|------|---------|
| `--env` | All 7 registered environments |
| `--latent` | All latents in the env (requires `--env`) |
| `--prompt` | All prompts registered for the env |
| `--feedback` | All feedbacks registered for the env |
| `--reward-type` | All 4 types: cumulative, terminal, improvement, per_episode |
| `--complexity` | No filter (all complexities) |
| `--n-trajectories` | 100 per (env, latent) |
| `--num-episodes` | 10 per trajectory |
| `--seed` | 42 for eval, 10000 for train |
| `--env-param KEY=VALUE` | No overrides (uses registry defaults). Overrides env_params from registry. Auto-converts int/float/bool. Available on `eval` and `train` commands. |

### Dependency rules (same across all CLI commands)

| You specify | Requires |
|------------|----------|
| `--latent` | `--env` (latents are env-specific) |
| `--prompt` | `--env` (prompts are env-specific) |
| `--feedback` | `--env` (feedbacks are env-specific) |
| `--complexity` | nothing (works with or without `--env`) |

## generate_data Examples

```bash
# List all environments
python -m latentgym.cli.generate_data list

# List latents for an env
python -m latentgym.cli.generate_data list --env bandits
python -m latentgym.cli.generate_data list --env bandits --complexity easy

# Generate everything for all envs, train/val split
python -m latentgym.cli.generate_data eval --val-ratio 0.2 --output data/eval/

# Generate for one env, easy latents only
python -m latentgym.cli.generate_data eval --env bandits --complexity easy --val-ratio 0.2 --output data/eval/

# Generate for specific latents
python -m latentgym.cli.generate_data eval --env bandits --latent loyal_favorite_0,clockwise_rotation --val-ratio 0.2 --output data/eval/

# Specific prompt/feedback (instead of all)
python -m latentgym.cli.generate_data eval --env bandits --prompt full_info --feedback standard --val-ratio 0.2 --output data/eval/

# Training shortcut (JSONs + parquets, seed=10000)
python -m latentgym.cli.generate_data train --env bandits --latent loyal_favorite_0 --output data/train/

# Convert existing JSONs to parquets
python -m latentgym.cli.generate_data parquet --source data/eval/bandits/loyal_favorite_0/train/ --mode train

# Dry run (preview without generating)
python -m latentgym.cli.generate_data eval --env bandits --dry-run --output data/eval/

# Pre-filter candidate pools (one-time, speeds up wordle/hangman/wordladder generation)
python -m latentgym.cli.generate_data filter-pool --env wordle --raw-pool word_lists/5letter.txt --output data/pools/
python -m latentgym.cli.generate_data filter-pool --env hangman --raw-pool word_lists/all_words.txt --output data/pools/
python -m latentgym.cli.generate_data filter-pool --env wordladder --raw-pool word_lists/pairs.txt --output data/pools/

# Filter only easy latents
python -m latentgym.cli.generate_data filter-pool --env wordle --raw-pool word_lists/5letter.txt --output data/pools/ --complexity easy

# Override registry env_params (auto-converts types)
python -m latentgym.cli.generate_data eval --env bandits --env-param num_turns=30 --env-param buttons=3 --output data/eval/
python -m latentgym.cli.generate_data train --env wordle --env-param max_turns_per_episode=8 --output data/train/
```

## run_eval Examples

```bash
# Single model, single env
python -m latentgym.cli.run_eval single \
    --models openai:gpt-4o \
    --env bandits --latent loyal_favorite_0 \
    --prompt full_info --feedback standard \
    --trajectory-dir data/eval/ \
    --output results/run_001/

# Multiple models, all latents for an env
python -m latentgym.cli.run_eval single \
    --models openai:gpt-4o anthropic:claude-sonnet-4-6 \
    --env bandits \
    --trajectory-dir data/eval/ \
    --output results/run_001/

# All envs, easy latents only
python -m latentgym.cli.run_eval single \
    --models openai:gpt-4o \
    --complexity easy \
    --trajectory-dir data/eval/ \
    --output results/run_001/

# Double-agent eval
python -m latentgym.cli.run_eval double \
    --model-a openai:gpt-4o --model-b openai:gpt-4o-mini \
    --switch-episode 5 \
    --env bandits \
    --trajectory-dir data/eval/ \
    --output results/double/

# From YAML config
python -m latentgym.cli.run_eval single \
    --config configs/eval_suites/quick.yaml \
    --output results/run_001/

# Dry run
python -m latentgym.cli.run_eval single --models openai:gpt-4o --env bandits --dry-run --output results/

# With reasoning/thinking enabled (Anthropic, Google)
python -m latentgym.cli.run_eval single \
    --models anthropic:claude-sonnet-4-6 \
    --env bandits --latent loyal_favorite_0 \
    --enable-thinking --thinking-budget 10000 \
    --trajectory-dir data/eval/ \
    --output results/with_thinking/

# Via OpenRouter (reasoning captured automatically, no --enable-thinking needed)
python -m latentgym.cli.run_eval single \
    --models openrouter/anthropic:claude-sonnet-4-6 \
    --env bandits \
    --trajectory-dir data/eval/ \
    --output results/openrouter/
```

### Thinking/Reasoning Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-thinking` | off | Enable extended thinking for Anthropic and Google models |
| `--thinking-budget` | 10000 | Max tokens for thinking/reasoning |

Reasoning support by provider route:
- **OpenRouter** (`openrouter/...`): reasoning captured automatically via `include_reasoning: True`
- **Anthropic direct** (`anthropic:...`): requires `--enable-thinking` (forces temperature=1)
- **Google direct** (`google:...`): requires `--enable-thinking`
- **vLLM** (`vllm:...`): reasoning captured if server uses `--reasoning-parser`
- **OpenAI direct** (`openai:...`): reasoning not available (ChatCompletions API discards it)

Reasoning is stored in `reasoning_trace` on each trajectory result and displayed in the trajectory viewer/explorer.

## report Examples

```bash
# Generate everything (tables + plots + dashboard + trajectory explorer)
python -m latentgym.cli.report --data-dir results/run_001/ --output paper/

# Individual outputs
python -m latentgym.cli.report --data-dir results/ --tables-only --output paper/
python -m latentgym.cli.report --data-dir results/ --plots-only --output paper/ --fmt pdf
python -m latentgym.cli.report --data-dir results/ --dashboard --output paper/
python -m latentgym.cli.report --data-dir results/ --trajectories --output paper/

# Leaderboard
python -m latentgym.cli.report --data-dir results/ --leaderboard
python -m latentgym.cli.report --data-dir results/ --leaderboard --env bandits

# Single trajectory
python -m latentgym.cli.report --trajectory results/trajectories/.../traj_0000.json
python -m latentgym.cli.report --trajectory results/trajectories/.../traj_0000.json --html traj.html

# Model comparison
python -m latentgym.cli.report --data-dir results/ --compare gpt-4o claude-3.5
```

## Training on Multiple Parquets

After generating data, you'll have individual parquets per (env, latent, prompt, feedback, reward) combination. To train on multiple environments/latents together, there are three approaches:

### Approach 1: Pass multiple parquets to SkyRL directly (recommended)

SkyRL's `PromptDataset` natively accepts a list of files and concatenates them internally:

```yaml
# skyrl config
data:
  train_data:
    - data/train/bandits/loyal_favorite_0/train/parquets/full_info_standard_cumulative.parquet
    - data/train/bandits/clockwise_rotation/train/parquets/full_info_standard_cumulative.parquet
    - data/train/wordle/vowel_count_2/train/parquets/full_info_standard_cumulative.parquet
```

No combining needed. SkyRL loads each file and samples uniformly from all rows.

### Approach 2: Concatenate parquets with pandas

If you prefer a single file (or SkyRL config only accepts one path):

```python
import pandas as pd
from pathlib import Path

parquets = list(Path("data/train/").rglob("parquets/full_info_standard_cumulative.parquet"))
df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
df.to_parquet("data/train/combined.parquet", index=False)
```

```yaml
data:
  train_data: data/train/combined.parquet
```

### Approach 3: Use generate_multi_env_train_parquet (from Python)

Generates a combined parquet directly from trajectory JSONs without creating individual parquets first:

```python
from latentgym.data.train.generate import generate_multi_env_train_parquet

generate_multi_env_train_parquet(
    configs=[
        {
            "env_name": "bandits",
            "latent_id": "loyal_favorite_0",
            "trajectory_dir": "data/train/bandits/loyal_favorite_0/train/",
            "prompt_ids": ["no_info", "full_info"],
            "feedback_ids": ["standard"],
            "reward_types": ["cumulative"],
        },
        {
            "env_name": "wordle",
            "latent_id": "vowel_count_2",
            "trajectory_dir": "data/train/wordle/vowel_count_2/train/",
            # Omit prompt_ids etc. to use defaults
        },
    ],
    output_path="data/train/mixed.parquet",
)
```

### When to use which

| Approach | Best for |
|----------|----------|
| **1. Multiple files in SkyRL** | Simplest — no extra step, add/remove envs by editing YAML |
| **2. pd.concat** | When you want one file, or need to filter/sample specific parquets |
| **3. generate_multi_env** | When generating from scratch and want fine control over prompt/feedback per env |

All three produce identical training behavior — SkyRL sees the same rows regardless.

## Typical End-to-End Workflow

```bash
# 0. (One-time) Pre-filter candidate pools for filter-based envs
python -m latentgym.cli.generate_data filter-pool \
    --env wordle --raw-pool word_lists/5letter_hardcore.txt --output data/pools/
python -m latentgym.cli.generate_data filter-pool \
    --env hangman --raw-pool word_lists/all_words.txt --output data/pools/

# 1. Generate data (JSONs + parquets) with train/val split
python -m latentgym.cli.generate_data eval \
    --env bandits,wordle --complexity easy \
    --val-ratio 0.2 --n-trajectories 200 \
    --output data/eval/

# 2. Train with SkyRL (using the train parquets)
python -m skyrl_train.entrypoints.main_base \
    data.train_data=data/eval/bandits/loyal_favorite_0/train/parquets/full_info_standard_cumulative.parquet \
    environment.env_class=latentgym_bandits

# 3. Evaluate the trained model
python -m latentgym.cli.run_eval single \
    --models vllm:http://localhost:8000 \
    --env bandits --complexity easy \
    --trajectory-dir data/eval/ \
    --output results/run_001/

# 4. Generate reports
python -m latentgym.cli.report --data-dir results/run_001/ --output paper/

# 5. Open in browser
open paper/dashboard.html
open paper/trajectory_explorer.html
```
