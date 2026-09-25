# Bandit Generator Latent Example

Demonstrates the **generator path** — running a bandit environment directly with a seed, without pre-generated trajectory JSON files.

## How It Differs from Standard Path

```
Standard path (all benchmark envs):
  trajectory_generator ──► traj.json ──► make_env(fd, trajectory_path="traj.json")
                                              │
                                              ▼
                                         MultiEpisodeEnv (loads configs from JSON)

Generator path (this example):
  make_env(fd, seed=42)
       │
       ├── generator_fn(seed) called per episode ──► episode_configs in memory
       │
       ▼
  MultiEpisodeEnv (configs generated on the fly, no JSON)
```

## Purpose

This is an **example**, not a standard benchmark env. It shows how `generator_fn` latents can produce episode configs on the fly:

```python
# Standard benchmark path (all envs):
env = make_env(fd, trajectory_path="traj.json")  # Load pre-generated ground truth

# Generator path (this example):
env = make_env(fd, seed=42)  # Ground truth generated on the fly from seed
```

## Usage

```python
import latentgym.envs.bandit_generator_latent_env
from latentgym.core import FullyDefinedEnv, make_env

fd = FullyDefinedEnv(
    env_name="bandit_generator_example",  # Note: different env_name
    latent_id="loyal_favorite_0",
    prompt_id="full_info",
    feedback_id="standard",
    num_episodes=5,
)
env = make_env(fd, seed=42)  # No trajectory file needed
```

## When to Use Generator Path

- Quick testing and debugging (no data generation step)
- Interactive exploration of latent patterns
- Unit tests

## When NOT to Use Generator Path

- Benchmarking (use trajectory files so all models see the same ground truth)
- Training (SkyRL requires parquets pointing to trajectory files)
- Comparing models (reproducibility requires shared trajectory files)
