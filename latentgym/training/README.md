# Training

RL training on benchmark environments via SkyRL. This guide covers **single-node GPU training only** — no HPC / SLURM / Shifter assumptions. The canonical example is `number_guessing` with `Qwen/Qwen2.5-1.5B-Instruct` and GRPO.

Two ready-to-run scripts:

| Script | What | Use when |
|---|---|---|
| [`train_minimal.sh`](train_minimal.sh) | Single-GPU, ~30 lines, GRPO, no critic, console logger | Smoke test, debugging, learning the pipeline |
| [`train_fsdp.sh`](train_fsdp.sh) | 4-GPU FSDP, colocated policy+ref, multi-engine vLLM | Real runs |

Both call `python -m skyrl_train.entrypoints.main_base` with Hydra config overrides. Defaults come from [`skyrl-train/skyrl_train/config/ppo_base_config.yaml`](../../skyrl-train/skyrl_train/config/ppo_base_config.yaml).

---

## 1. Quick Start (Single GPU)

### Prerequisites

- Project installed (see [docs/getting_started.md](../../docs/getting_started.md))
- `.env` has `VENV_DIR` set
- At least one GPU visible to the venv
- Optional: `HF_TOKEN` in `.env` if the model is gated

### Step 1 — Generate training data

Trajectories + parquets are produced in one CLI call:

```bash
python -m latentgym.cli.generate_data train \
    --env number_guessing \
    --latent set_of_3 \
    --n-trajectories 500 \
    --num-episodes 10 \
    --seed 10000 \
    --output latentgym/data/train/
```

This creates `latentgym/data/train/number_guessing/set_of_3/parquets/full_info_standard_cumulative.parquet` (and a validation split under `val/`). One parquet per `(prompt × feedback × reward_type)` combination is auto-generated. The training script uses the `full_info_standard_cumulative` parquet by default.

For envs that need word pools (`wordle`, `hangman`, `wordladder`), generate the pools first:

```bash
python -m latentgym.data.generate_word_lists --env all --output latentgym/data/pools/
```

### Step 2 — Run training

```bash
bash latentgym/training/train_minimal.sh
```

Edit the variables at the top of the script (`ENV`, `LATENT`, `MODEL`, `SEED`, `NUM_EPISODES`, `MAX_TURNS_PER_EPISODE`) before running. The script:

1. Sources `project_config.sh` and activates the venv.
2. Sets `SKYRL_REGISTER_MODULES=latentgym.register_skyrl` so benchmark envs register with `skyrl_gym`.
3. Clears any stale Ray cluster.
4. Calls `python -m skyrl_train.entrypoints.main_base` with the GRPO + single-GPU Hydra overrides.

### What gets produced

- **Checkpoints** under `~/ckpts/` (default `trainer.ckpt_path`). Override with `trainer.ckpt_path=<path>`.
- **HF-format exports** under `~/exports/` (default `trainer.export_path`).
- **Console logs** because the minimal script uses `trainer.logger=console`. Swap to `wandb` to log runs to W&B (see Section 2).

---

## 2. Multi-GPU Training (FSDP)

Use FSDP when the model + activations don't fit on one GPU, or when you want batched training to be faster. The `train_fsdp.sh` script uses 4 GPUs by default and is configured for FSDP2 + colocated placement.

### Difference vs the minimal script

| Setting | Minimal (1 GPU) | FSDP (4 GPUs) |
|---|---|---|
| `trainer.placement.policy_num_gpus_per_node` | `1` | `4` |
| `trainer.placement.ref_num_gpus_per_node` | `1` | `4` |
| `trainer.placement.critic_num_gpus_per_node` | `1` | `4` |
| `trainer.placement.colocate_all` | (default true) | explicitly `true` |
| `trainer.strategy` | (default `fsdp2`) | explicitly `fsdp2` |
| `trainer.policy.fsdp_config.cpu_offload` | `true` | `true` |
| `trainer.ref.fsdp_config.cpu_offload` | `true` | `true` |
| `generator.num_inference_engines` | `1` | `4` |
| `generator.inference_engine_tensor_parallel_size` | `1` | `1` (one engine per GPU) |
| `generator.gpu_memory_utilization` | `0.5` | `0.7` |
| `trainer.train_batch_size` | `8` | `32` |
| `trainer.policy_mini_batch_size` | `4` | `8` |
| `generator.n_samples_per_prompt` | `4` | `8` |

### Run

```bash
bash latentgym/training/train_fsdp.sh
```

Edit the top-of-file vars (`NUM_GPUS`, `TRAIN_BATCH_SIZE`, `N_SAMPLES_PER_PROMPT`, `MINI_BATCH_SIZE`) for your hardware.

### GPU memory tuning

- **`generator.gpu_memory_utilization`** (0.0–1.0) caps how much of each GPU vLLM uses. Lower if you OOM, raise to improve throughput.
- **`trainer.policy.fsdp_config.cpu_offload=true`** offloads params + optimizer state to CPU during the forward pass. Slows training but unlocks larger models. Same flag for `trainer.ref`.
- **`trainer.placement.colocate_all=true`** packs policy + ref onto the same GPUs (memory-efficient; only one model resident at a time during phases).
- For tensor-parallel inference (one model split across multiple GPUs), raise `generator.inference_engine_tensor_parallel_size` and lower `generator.num_inference_engines` accordingly. The default (`num_inference_engines=$NUM_GPUS`, `tensor_parallel_size=1`) is one independent engine per GPU — best for batched throughput on small models.

### Enabling W&B

In `train_fsdp.sh`, change:

```bash
trainer.logger=wandb
+trainer.wandb_entity=<your-entity>
+trainer.wandb_group=<group-name>
```

The `+` prefix is required for keys not in the default config schema.

---

## 3. Data Format and Multi-Env Training

### Parquet schema

Each parquet row has this structure (produced by `latentgym.cli.generate_data train`):

```python
{
    "prompt": [...],                            # placeholder; env rebuilds at runtime
    "env_class": "latentgym_number_guessing",   # registered in latentgym.register_skyrl
    "reward_spec": {...},
    "extra_info": {
        "trajectory_path": "/abs/path/traj_000.json",   # ground-truth episode configs
        "env_name": "number_guessing",
        "latent_id": "set_of_3",
        "prompt_id": "full_info",
        "feedback_id": "standard",
        "reward_type": "cumulative",
        "num_episodes": 10
    }
}
```

At training time SkyRL reads each row, calls `skyrl_gym.make(extras=extra_info)`, which instantiates `MultiEpisodeEnv`, loads the trajectory JSON, and runs episodes. The parquet is a lightweight metadata wrapper; the actual ground truth lives in the trajectory JSONs that `trajectory_path` points to.

Parquets are generated in `latentgym/data/train/<env>/<latent>/parquets/`, one file per `<prompt_id>_<feedback_id>_<reward_type>` combination.

### What varies at which stage

Which knobs are baked into the trajectory JSONs vs. resolved at parquet (and thus training) time:

```
Varies at trajectory generation        Varies at parquet generation
(different ground truth)               (same ground truth, different config)
───────────────────────────            ─────────────────────────────────────
latent_id                              prompt_id
num_episodes                           feedback_id
seed                                   reward_type
```

So the same trajectory JSONs are reused across the 12 parquet files for an (env, latent): only `prompt_id × feedback_id × reward_type` differ. Changing `latent_id` or `num_episodes` requires re-running `generate_data train`.

### Single-env training (default)

Section 1 covers this. The training script's `data.train_data` is a list with one parquet path.

### Multi-env training

`data.train_data` accepts a list. SkyRL concatenates all rows; envs/latents/prompts can mix freely.

```bash
python -m skyrl_train.entrypoints.main_base \
    data.train_data="['latentgym/data/train/number_guessing/set_of_3/parquets/full_info_standard_cumulative.parquet','latentgym/data/train/number_guessing/range_100/parquets/full_info_standard_cumulative.parquet','latentgym/data/train/wordle/vowel_count_2/parquets/full_info_standard_cumulative.parquet']" \
    data.val_data="['latentgym/data/train/number_guessing/set_of_3/val/parquets/full_info_standard.parquet']" \
    environment.env_class=latentgym_number_guessing \
    ...  # rest of the Hydra overrides (use train_fsdp.sh as a template)
```

Notes:
- `environment.env_class` is still a single value, but `MultiEpisodeEnv` dispatches to the right per-row env via `extra_info.env_name`.
- Generate data for each env separately first (`generate_data train --env wordle ...`).
- Trajectory paths inside the parquets are absolute — moving parquets to another machine without their trajectory JSONs will break training.

---

## 4. Adding a New Algorithm

Built-in algorithms are selected via `trainer.algorithm.advantage_estimator`:

- `grpo` (default for our scripts)
- `gae` (uses a critic — set `trainer.critic.model.path` to a real model)
- `rloo`, `reinforce++`

For implementing a fully custom algorithm (e.g., new advantage estimator or policy loss), see the upstream **[SkyRL repository](https://github.com/NovaSky-AI/SkyRL)** — `trainer.algorithm.advantage_estimator` plugs into SkyRL's `AdvantageEstimatorRegistry`, and `trainer.algorithm.policy_loss_type` plugs into `PolicyLossRegistry`. The registries live in `skyrl-train/skyrl_train/`.

---

## Reference

- Entry point: [`skyrl-train/skyrl_train/entrypoints/main_base.py`](../../skyrl-train/skyrl_train/entrypoints/main_base.py)
- Hydra defaults: [`skyrl-train/skyrl_train/config/ppo_base_config.yaml`](../../skyrl-train/skyrl_train/config/ppo_base_config.yaml)
- Env registration: [`latentgym/register_skyrl.py`](../register_skyrl.py)
- Data generation CLI: [`latentgym/cli/generate_data.py`](../cli/generate_data.py)
