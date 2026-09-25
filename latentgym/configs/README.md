# Configs

Pre-defined configuration files for the benchmark. **All configs are optional** — every command works fully through CLI args without any YAML files.

## Are configs required?

**No.** The CLI supports full hierarchical expansion through flags:

```bash
# These work without any YAML files:
python -m latentgym.cli.generate_data eval --env bandits --complexity easy --output data/eval/
python -m latentgym.cli.run_eval single --models openai:gpt-4o --env bandits --output results/
python -m latentgym.cli.report --data-dir results/ --output paper/
```

YAML configs are **convenience presets** — they save you from typing long commands and make experiments reproducible by checking a file into git.

## What reads YAML vs what doesn't

| Command | Uses YAML? | Notes |
|---------|-----------|-------|
| `generate_data list` | No | Reads from registry |
| `generate_data eval` | No | Uses `--env`, `--latent`, `--complexity`, etc. |
| `generate_data parquet` | No | Uses `--source`, `--mode`, `--prompt`, etc. |
| `generate_data train` | No | Uses `--env`, `--latent`, etc. |
| `run_eval single --env ...` | No | CLI args build configs directly |
| `run_eval single --config X.yaml` | **Yes** | Loads eval suite from YAML |
| `run_eval double` | No | Uses `--model-a`, `--model-b`, etc. |
| `report` (all modes) | No | Reads from DataStore directory |

Only `run_eval --config <file>` actually loads a YAML file. Everything else uses CLI args.

## Directory Structure

```
configs/
├── eval_suites/              # Pre-defined evaluation configs (loaded by --config)
│   ├── quick.yaml            # Fast sanity check (6 configs)
│   ├── easy.yaml             # All envs, easy latents (~38 configs)
│   ├── full.yaml             # Reference — use CLI expansion instead
│   ├── bandits_only.yaml     # Bandits deep dive (all latents × prompts)
│   ├── wordle_only.yaml      # Wordle deep dive (24 latents across tiers)
│   ├── number_guessing_only.yaml  # All 7 latents × 3 prompts
│   └── prompt_ablation.yaml  # Compare no_info vs some_info vs full_info
│
├── envs/                     # Environment reference docs (NOT loaded by code)
│   ├── bandits.yaml          # 28 latents, 5 buttons, 20 turns
│   ├── wordle.yaml           # 165 latents, 5-letter, 6 attempts
│   ├── hangman.yaml          # 105 latents, 6 attempts
│   ├── mastermind.yaml       # 54 latents, 4-digit code
│   ├── secretary.yaml        # 30 latents, 10 draws
│   ├── wordladder.yaml       # 53 latents, word pairs
│   └── number_guessing.yaml  # 7 latents, range 1-1000
│
└── models/                   # Model provider reference docs (NOT loaded by code)
    ├── openai.yaml           # gpt-4o, gpt-4o-mini, gpt-3.5-turbo
    ├── anthropic.yaml        # claude-opus, claude-sonnet, claude-haiku
    └── local_vllm.yaml       # local vLLM server setup
```

## eval_suites/ — Loaded by `run_eval --config`

These are the only configs that are actually loaded by code. Each file defines a list of (env, latent, prompt, feedback, num_episodes) combinations.

### Format

```yaml
configs:
  - env: bandits
    latent: loyal_favorite_0
    prompt: full_info            # optional, defaults to full_info
    feedback: standard           # optional, defaults to standard
    num_episodes: 10             # optional, defaults to 10
  - env: wordle
    latent: vowel_count_2
    num_episodes: 5
```

### Usage

```bash
python -m latentgym.cli.run_eval single \
    --config latentgym/configs/eval_suites/quick.yaml \
    --models openai:gpt-4o \
    --output results/quick/
```

### Available suites

| Suite | What it covers | Size | Use case |
|-------|---------------|------|----------|
| `quick.yaml` | bandits (3 easy) + number_guessing (3) | 6 | Sanity check, CI |
| `easy.yaml` | All 7 envs, ~5 easy latents each | ~38 | Starting benchmark |
| `full.yaml` | Placeholder — use CLI instead | 0 | Reference only |
| `bandits_only.yaml` | All bandits latents, 3 prompts on key ones | ~21 | Bandits deep dive |
| `wordle_only.yaml` | 24 wordle latents across all tiers | ~24 | Wordle deep dive |
| `number_guessing_only.yaml` | All 7 latents × 3 prompts | 21 | Quick full-env eval |
| `prompt_ablation.yaml` | 4 envs × 2 latents × 3 prompts | 24 | Prompt effect study |

### CLI equivalent

Every eval suite has a CLI equivalent. For example:

```bash
# quick.yaml is equivalent to:
python -m latentgym.cli.run_eval single \
    --models openai:gpt-4o \
    --env bandits --latent loyal_favorite_0,clockwise_rotation,p_gap_large \
    --prompt full_info --feedback standard --num-episodes 5 \
    --output results/quick/

# easy.yaml is equivalent to:
python -m latentgym.cli.run_eval single \
    --models openai:gpt-4o \
    --complexity easy --prompt full_info \
    --output results/easy/
```

The YAML version is useful when:
- You want to check a specific set of configs into git
- You want to share an exact experiment setup with collaborators
- You want to run the same setup across multiple machines

## envs/ — Reference only

These files document each environment's parameters, latent counts, and notes. **No code reads them.** They serve as quick-reference documentation alongside the full READMEs in `latentgym/envs/<env>/README.md`.

Each file lists:
- `env_params` — default parameters (buttons, num_turns, word_length, etc.)
- `prompts` / `feedbacks` — registered prompt and feedback IDs
- `latent_summary` — count per complexity tier
- `notes` — env-specific caveats (word pool requirements, duplicate modes, etc.)

## models/ — Reference only

These files document available model providers and their specs. **No code reads them.** Models are passed via the `--models` CLI flag:

```bash
# Model spec format: provider:model_name
--models openai:gpt-4o                     # OpenAI
--models anthropic:claude-sonnet-4-6       # Anthropic
--models vllm:http://localhost:8000        # Local vLLM
--models mock:random                       # Mock (no API key needed)
```

The YAML files show available models per provider, recommended sampling parameters, and required environment variables (API keys).

## Adding a new eval suite

Create a new YAML in `eval_suites/`:

```yaml
# configs/eval_suites/my_experiment.yaml
configs:
  - { env: bandits, latent: loyal_favorite_0, prompt: full_info, num_episodes: 10 }
  - { env: bandits, latent: fibonacci, prompt: full_info, num_episodes: 10 }
  - { env: wordle, latent: vowel_count_2, prompt: no_info, num_episodes: 5 }
```

Run it:

```bash
python -m latentgym.cli.run_eval single \
    --config latentgym/configs/eval_suites/my_experiment.yaml \
    --models openai:gpt-4o \
    --output results/my_experiment/
```
