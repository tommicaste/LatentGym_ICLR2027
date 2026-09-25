# Local Model Evaluation — Complete Guide

Two paths for running a local model against the benchmark:

1. **Option 1**: vLLM as an API server, driven by the benchmark CLI (simple, sequential).
2. **Option 2**: SkyRL local runner with batched inference (faster for large N).
   - **2A**: You don't have a parquet yet (build one from trajectory JSONs).
   - **2B**: You already have a parquet.

---

## Prerequisites (both options)

Activate the project venv:

```bash
cd <repo-root>/skyrl-train
source <repo-root>/.venv/bin/activate
```

You need at least one of:

- A directory of trajectory JSONs (`manifest.json` + `traj_*.json`), e.g. at `data/eval/number_guessing/dynamic_range/`.
- A SkyRL parquet built from those trajectories.

---

## Option 1 — vLLM as an API server (via `run_eval` CLI)

Simplest path. Start a vLLM server, then point the benchmark CLI at it.

### Step 1: Start the vLLM server

In a separate terminal (or background):

```bash
vllm serve /path/to/your-model --port 8000

# For reasoning models (DeepSeek-R1, QwQ):
vllm serve /path/to/model --port 8000 --reasoning-parser

# For multi-GPU:
vllm serve /path/to/model --port 8000 --tensor-parallel-size 2
```

Verify it's up:

```bash
curl http://localhost:8000/v1/models
```

### Step 2: Run the eval

```bash
cd <repo-root>/skyrl-train

python -m latentgym.cli.run_eval single \
    --models vllm:http://localhost:8000 \
    --env number_guessing \
    --latent dynamic_range \
    --prompt full_info \
    --feedback information \
    --num-episodes 10 \
    --n-trajectories 50 \
    --trajectory-dir data/eval/ \
    --output results/local_vllm_run/ \
    --temperature 0.7 \
    --max-tokens 512 \
    --max-retries 3 \
    --request-timeout 120
```

**What this does**: parses `vllm:http://localhost:8000` → instantiates `VLLMModel` (`latentgym/eval/model_interface.py`, lines 522–581) → uses OpenAI-compatible API → loads trajectory JSONs from `data/eval/number_guessing/dynamic_range/` → runs them sequentially → writes per-trajectory results to `results/local_vllm_run/trajectories/...`.

**Inputs needed**: trajectory JSONs only. Parquet is NOT used here.

**Resume after interruption**: add `--resume`.

---

## Option 2 — SkyRL local runner (batched inference, ~32× faster)

Uses SkyRL's `SkyRLGymGenerator` for batched vLLM inference with shared KV cache. Not wired into the CLI — you must drive it programmatically.

### Note on env registration (read first)

Both 2A and 2B below show an explicit `register(...)` call. **You only need this if your eval process hasn't already registered the env via some other path.** Registration with `skyrl_gym` is a side-effect on a global registry: once anything in the process runs `register(id="benchmark_multi_episode", entry_point="...")`, you can `skyrl_gym.make("benchmark_multi_episode", ...)` from anywhere.

Three equivalent ways to get it registered — pick whichever matches your setup:

1. **Explicit `register()` call** (what `local_runner.py` does, used in the examples below) — most portable, works from a clean-slate script.

2. **Import the module that already registers it.** If your training pipeline has an `__init__.py` (e.g. `latentgym/envs/__init__.py`) that calls `register(...)` at import time, a single import does the same job:
   ```python
   import latentgym  # or whatever your training entrypoint imports
   ```
   To find which file does this:
   ```bash
   grep -rn "skyrl_gym.envs.registration" latentgym/ skyrl-train/
   ```

3. **SkyRL's config-driven discovery.** Some SkyRL setups register envs from a Hydra config field (e.g. `environment.entry_point` or a plugin list). If your training config already wires this, your eval config can do the same and SkyRL handles registration for you.

**Bottom line**: if your eval runs in the same process / from the same entrypoint as training, registration likely already happened — you can drop the explicit `register(...)` call from the snippets below. If you launch eval as a separate process or a clean script, keep it (or replace it with the equivalent import / config line your training uses).

### 2A. You DON'T have a parquet yet

Build one from your trajectory JSONs, then run SkyRL.

Save as `scripts/run_local_skyrl.py`:

```python
import asyncio
from pathlib import Path
from skyrl_gym.envs.registration import register
from latentgym.core.env_config import FullyDefinedEnv, RewardType
from latentgym.eval.single_agent.local_runner import (
    _build_eval_parquet, register_with_skyrl,
)

# 1. Define the eval config
fully_defined = FullyDefinedEnv(
    env_name="number_guessing",
    latent_id="dynamic_range",
    prompt_id="full_info",
    feedback_id="information",
    num_episodes=10,
    reward_type=RewardType.CUMULATIVE,
)

# 2. Register MultiEpisodeEnv with skyrl_gym
register_with_skyrl("benchmark_multi_episode")

# 3. Build the parquet from trajectory JSONs
parquet_path = "results/local_skyrl/eval.parquet"
Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
_build_eval_parquet(
    fully_defined=fully_defined,
    trajectory_dir="data/eval/number_guessing/dynamic_range/",
    env_class_id="benchmark_multi_episode",
    output_path=parquet_path,
)
print(f"Parquet written to {parquet_path}")
```

Run it:

```bash
python scripts/run_local_skyrl.py
```

Then run SkyRL's eval entrypoint with this Hydra config:

```yaml
# configs/skyrl_eval.yaml
environment:
  env_class: benchmark_multi_episode

dataset:
  path: results/local_skyrl/eval.parquet
  prompt_key: prompt
  env_class_key: env_class

generator:
  backend: vllm
  model_path: /path/to/your-model
  tensor_parallel_size: 1
  eval_batch_size: 16
  eval_n_samples_per_prompt: 1
  eval_sampling_params:
    temperature: 0.7
    max_tokens: 512
```

Invoke (substitute your SkyRL eval entrypoint — `skyrl_train.evaluate` or equivalent in your SkyRL install):

```bash
python -m skyrl_train.evaluate --config-path=configs --config-name=skyrl_eval
```

### 2B. You ALREADY have a parquet

Skip the parquet build. The parquet must have this schema:

```
columns: data_source, prompt, env_class, reward_spec, extra_info

extra_info fields:
  trajectory_path, env_name, latent_id, prompt_id,
  feedback_id, num_episodes, reward_type
```

The `env_class` column must match the ID you register (default: `"benchmark_multi_episode"`).

Verify your parquet schema matches:

```bash
python -c "
import pandas as pd
df = pd.read_parquet('/path/to/your.parquet')
print('Columns:', df.columns.tolist())
print('Sample extra_info:', df.iloc[0]['extra_info'])
print('env_class values:', df['env_class'].unique())
"
```

Register the env with skyrl_gym before running:

```python
# scripts/register_env.py
from skyrl_gym.envs.registration import register

register(
    id="benchmark_multi_episode",   # must match your parquet's env_class column
    entry_point="latentgym.core.multi_episode_env:MultiEpisodeEnv",
)
```

Point SkyRL's Hydra config at the existing parquet:

```yaml
# configs/skyrl_eval.yaml
environment:
  env_class: benchmark_multi_episode      # must match parquet's env_class column

dataset:
  path: /path/to/your.parquet              # <-- your existing parquet
  prompt_key: prompt
  env_class_key: env_class

generator:
  backend: vllm
  model_path: /path/to/your-model
  tensor_parallel_size: 1
  eval_batch_size: 16
  eval_n_samples_per_prompt: 1
  eval_sampling_params:
    temperature: 0.7
    max_tokens: 512
```

Run:

```bash
python -m skyrl_train.evaluate --config-path=configs --config-name=skyrl_eval
```

> **Important**: The trajectory JSONs referenced by `extra_info.trajectory_path` in each parquet row **must still exist on disk**. `MultiEpisodeEnv._init_from_skyrl_extras()` (`latentgym/core/multi_episode_env.py`, lines 108–110) reads them at episode-config load time. If the parquet was built on another machine, sync the trajectory directory to the same paths.

---

## When to use which

| Scenario                                                       | Use                                          |
| -------------------------------------------------------------- | -------------------------------------------- |
| One-off eval, single GPU, small N                              | **Option 1** (vLLM API)                      |
| Need resume / per-trajectory output / `report.py` integration  | **Option 1**                                 |
| Large N, want batched throughput                               | **Option 2**                                 |
| Already have parquet from training pipeline                    | **Option 2B**                                |
| No parquet, only JSONs, but need throughput                    | **Option 2A**                                |

---

## Timing & Parallelism

### Sequential timing estimate

Per-trajectory time is roughly `num_episodes × turns_per_episode × time_per_turn`.

For a typical setup (10 episodes × ~15 turns avg):

| Model                                  | Time per turn  | Time per traj | 50 trajs       |
| -------------------------------------- | -------------- | ------------- | -------------- |
| Mock                                   | ~0.01s         | ~1.5s         | ~1 minute      |
| Local vLLM (8B model, single GPU)      | 0.5–2s         | 75–300s       | **1–4 hours**  |
| Local vLLM (70B model, single GPU)     | 3–8s           | 7–20 min      | **6–17 hours** |
| API model (GPT-4o, Claude)             | 1–3s           | 2.5–7.5 min   | **2–6 hours**  |

**Dominant variables:**

1. **Average turns per trajectory** — varies hugely by env. Number guessing `dynamic_range` often runs out the 30-turn budget on hard episodes; wordladder may finish in 5. Check your generated trajectories for `max_turns_per_episode` and the model's actual turn count from prior runs.
2. **Output token count** — reasoning models with `--reasoning-parser` can be 5–20× slower per turn (long thinking traces).
3. **Network latency** — for `vllm:http://localhost:8000` it's negligible (~1ms); for OpenRouter it's 50–200ms per turn × ~150 turns per traj = an extra ~10s/traj.

**Quick way to estimate yours**: run with `--n-trajectories 2` first, time it, multiply.

```bash
time python -m latentgym.cli.run_eval single \
    --models vllm:http://localhost:8000 \
    --env number_guessing --latent dynamic_range \
    --prompt full_info --feedback information \
    --num-episodes 10 --n-trajectories 2 \
    --trajectory-dir data/eval/ --output /tmp/timing_test/
```

Then `total_seconds / 2 × 50` is your estimate for 50 trajectories.

### Parallelism: multiple GPUs

Three different ways to use multiple GPUs — they help in different situations.

#### 1. Tensor parallelism (one big model spread across GPUs)

Use this if your **model doesn't fit on one GPU** (70B+ on consumer cards, 405B anywhere).

```bash
vllm serve /path/to/model --port 8000 --tensor-parallel-size 4
```

- Splits each layer's weights across GPUs.
- Latency per turn improves modestly (~1.3–1.8× on 4 GPUs due to NCCL communication).
- **Throughput per request doesn't really change** — still one logical engine.
- Best for fitting big models, not for speed on small ones.

#### 2. Data parallelism (multiple servers, multiple clients)

Use this if your **model fits on one GPU** and you want raw throughput. This is the biggest win.

```bash
# Start one vLLM server per GPU
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/model --port 8000 &
CUDA_VISIBLE_DEVICES=1 vllm serve /path/to/model --port 8001 &
CUDA_VISIBLE_DEVICES=2 vllm serve /path/to/model --port 8002 &
CUDA_VISIBLE_DEVICES=3 vllm serve /path/to/model --port 8003 &
```

Then split trajectories across `run_eval` processes:

```bash
# Process 1
python -m latentgym.cli.run_eval single --models vllm:http://localhost:8000 \
    --start-trajectory 0  --n-trajectories 12  --env number_guessing \
    --latent dynamic_range --prompt full_info --feedback information \
    --num-episodes 10 --trajectory-dir data/eval/ --output results/run/ &

# Process 2
python -m latentgym.cli.run_eval single --models vllm:http://localhost:8001 \
    --start-trajectory 12 --n-trajectories 12  --env number_guessing \
    --latent dynamic_range --prompt full_info --feedback information \
    --num-episodes 10 --trajectory-dir data/eval/ --output results/run/ &

# Process 3
python -m latentgym.cli.run_eval single --models vllm:http://localhost:8002 \
    --start-trajectory 24 --n-trajectories 13  --env number_guessing \
    --latent dynamic_range --prompt full_info --feedback information \
    --num-episodes 10 --trajectory-dir data/eval/ --output results/run/ &

# Process 4
python -m latentgym.cli.run_eval single --models vllm:http://localhost:8003 \
    --start-trajectory 37 --n-trajectories 13  --env number_guessing \
    --latent dynamic_range --prompt full_info --feedback information \
    --num-episodes 10 --trajectory-dir data/eval/ --output results/run/ &

wait
```

- Each `run_eval` process points at a different vLLM port.
- All four write to the same `--output` dir; per-trajectory result files don't collide because file names include the trajectory index.
- **Speedup**: near-linear with GPU count (~3.5–3.9× on 4 GPUs).

#### 3. Concurrent clients on one server (no extra GPUs needed)

vLLM **continuously batches** concurrent requests on a single GPU. So even with one GPU, launching multiple `run_eval` processes against the same port helps:

```bash
for i in 0 12 24 37; do
  python -m latentgym.cli.run_eval single \
      --models vllm:http://localhost:8000 \
      --start-trajectory $i --n-trajectories 12 \
      --env number_guessing --latent dynamic_range \
      --prompt full_info --feedback information \
      --num-episodes 10 --trajectory-dir data/eval/ \
      --output results/run/ &
done
wait
```

Typical: 3–5× throughput on one GPU. **Combine with #1 or #2** for compounding gains.

#### Combining them — example for 70B on 8 GPUs

For best throughput on a 70B model with 8 GPUs:

```bash
# Two servers, each using tensor-parallel 4 to fit the 70B model
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/70b --port 8000 --tensor-parallel-size 4 &
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /path/to/70b --port 8001 --tensor-parallel-size 4 &
```

Then run 4–8 `run_eval` clients distributed across both ports. Effective speedup over single-GPU sequential: **~6–10×**.

#### Decision tree

| Does the model fit on 1 GPU? | Setup                                                       |
| ---------------------------- | ----------------------------------------------------------- |
| Yes                          | Data parallelism (#2) — N servers, N clients                |
| No                           | Tensor parallelism (#1) for the model + concurrent clients (#3) |
| Partially (e.g. needs 2 GPUs) | TP=2 servers, run multiple of them (#1 + #2)               |

#### Updated time estimate (8B model, 4 GPUs, data-parallel + concurrent clients)

From the 1–4 hour single-GPU estimate above, expect **15–60 minutes** for 50 trajectories.

#### Caveat: avoiding race conditions on `--output`

`run_eval` writes to one `--output` directory and updates a checkpoint file. Concurrent processes writing to the same checkpoint can race. Two safe approaches:

- **Disjoint trajectory ranges via `--start-trajectory`** (shown in the snippets above) — each process owns a slice, no overlap.
- **Disjoint output directories then merge**:
  ```bash
  --output results/run/shard_0/
  --output results/run/shard_1/
  ```
  Then move the per-trajectory files into one directory before running `report.py`.

---

## Reference: Key Files

- CLI entry point: `latentgym/cli/run_eval.py`
- Model interface (vLLM client): `latentgym/eval/model_interface.py` (lines 522–581)
- Local runner (SkyRL bridge): `latentgym/eval/single_agent/local_runner.py`
- Multi-episode env: `latentgym/core/multi_episode_env.py`
- Example local vLLM config: `latentgym/configs/models/local_vllm.yaml`
