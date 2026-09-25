# Adding Environments

This guide explains how to add a new environment to the benchmark.

## Overview

Each environment is a Python package under `latentgym/envs/<env_name>/`:

```
latentgym/envs/my_env/
    __init__.py
    core_env.py       # SingleEpisodeEnv — game dynamics
    latents.py        # LatentDefinition registrations
    prompts.py        # PromptTemplate registrations
    feedbacks.py      # FeedbackHandler registrations
    trajectory_generator.py  # Generates trajectory JSON files
```

Alternatively, put everything in one file (see `_monolithic_example/`).

## Step 1 — Implement SingleEpisodeEnv

```python
# core_env.py
from latentgym.core.single_episode_env import SingleEpisodeEnv
from typing import Any, Dict, Tuple

class MyGameEnv(SingleEpisodeEnv):
    """Describe your game here."""

    def __init__(self):
        """No-arg constructor. All game params come from episode_config in reset()."""
        super().__init__()
        self._target = None
        self._turns = 0

    def reset(self, episode_config: Dict[str, Any]) -> str:
        """Set up a new episode. Returns the initial observation.

        All game parameters (including max_turns_per_episode) come from
        episode_config, NOT from __init__ args.
        """
        self._target = episode_config["target"]
        self._max_turns = episode_config.get("max_turns_per_episode", 10)
        self._turns = 0
        return f"New game! Guess the target."

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Execute one action.

        Returns:
            (feedback_text, reward, episode_done, info_dict)
        """
        self._turns += 1
        guess = self._parse_action(action)

        if guess == self._target:
            return "Correct!", 1.0, True, {"success": True}

        if self._turns >= self._max_turns:
            return f"Out of turns! Answer was {self._target}.", 0.0, True, {}

        return f"Wrong, try again.", 0.0, False, {}

    def get_game_rules(self) -> str:
        """Return static rules text used in the system prompt."""
        return "You are playing My Game. Guess the target within 10 turns."

    def _parse_action(self, action: str) -> int:
        import re
        nums = re.findall(r"\d+", action)
        return int(nums[-1]) if nums else -1
```

**Key points:**
- `__init__(self)` takes **no arguments**. All game params come from `episode_config` in `reset()`
- `reset()` receives `episode_config` (from the latent) and returns initial observation text
- `episode_config` should include `max_turns_per_episode` as the standard key for turn limits
- `step()` receives raw LLM action string and returns `(feedback, reward, done, info)`
- Reward is `float` in `[0, 1]` (or higher for partial credit envs)
- `get_game_rules()` returns static text; it's called once per trajectory

**Env reuse pattern (`_env_params_key`):** If your env wraps a TextArena env or other
expensive-to-construct object, use `_env_params_key` to track constructor params and
only recreate the underlying env when they change. This avoids re-instantiation on
every `reset()` call. See `hangman/core_env.py` or `wordladder/core_env.py` for examples.

## Step 2 — Define Latents

Latents define the hidden constraint that varies across episodes/trajectories.
Three modes: `generator_fn`, `filter_fn`, or `episode_configs`.

### Generator-based latent (random draws)

```python
# latents.py
from latentgym.core.latent import LatentDefinition, LatentComplexity
from latentgym.core.registry import register_latent

def _easy_generator(env_params, ep_idx, n_eps, ctx):
    """Always pick from [1, 10]."""
    import random
    return {
        "target": random.randint(1, 10),
        "max_turns_per_episode": env_params.get("max_turns_per_episode", 5),
    }

register_latent("my_env", LatentDefinition(
    id="easy",
    name="Easy Range",
    complexity=LatentComplexity.EASY,
    description="Target is always in [1, 10]",
    generator_fn=_easy_generator,
))
```

### Filter-based latent (select from a pool)

```python
def _vowel_start_filter(word: str) -> bool:
    return word[0].lower() in "aeiou"

register_latent("my_env", LatentDefinition(
    id="starts_vowel",
    name="Starts with Vowel",
    complexity=LatentComplexity.EASY,
    description="Target word starts with a vowel",
    filter_fn=_vowel_start_filter,
))
```

The trajectory generator applies `filter_fn` to a pool of candidates to select valid targets.

### Static latent (predetermined configs)

```python
register_latent("my_env", LatentDefinition(
    id="fixed",
    name="Fixed Targets",
    complexity=LatentComplexity.EASY,
    description="Always the same 5 targets in order",
    episode_configs=[
        {"target": 7},
        {"target": 23},
        {"target": 42},
        {"target": 15},
        {"target": 99},
    ],
))
```

> **Cross-episode patterns:** a `generator_fn` receives `(env_params, ep_idx, n_eps, ctx)`, where
> `ctx` is a dict shared across all episodes in a trajectory — use it to make episode K depend on
> episode K−1 (rotations, random walks, etc.). The registry also exposes an experimental
> `register_cross_episode_latent` hook, but it is currently unused; prefer `generator_fn` + `ctx`.

## Step 3 — Add Prompt Templates

```python
# prompts.py
from latentgym.core.prompt import PromptTemplate
from latentgym.core.registry import register_prompt

class NoInfoPrompt(PromptTemplate):
    """No latent hints."""
    id = "no_info"

    def initial_system_prompt(self, game_rules, env_params, num_episodes):
        return f"{game_rules}\n\nYou will play {num_episodes} episode(s)."

    def episode_transition_message(self, ep_idx, num_episodes, prev_reward, prev_info):
        return f"\n--- Episode {ep_idx + 1}/{num_episodes} --- Previous reward: {prev_reward:.4f}"

class FullInfoPrompt(PromptTemplate):
    """Tell agent that the latent constraint is fixed across episodes."""
    id = "full_info"

    def initial_system_prompt(self, game_rules, env_params, num_episodes):
        return (
            f"{game_rules}\n\n"
            f"You will play {num_episodes} episodes. "
            "The hidden constraint stays the same across all episodes. "
            "Learn it from experience!"
        )

    def episode_transition_message(self, ep_idx, num_episodes, prev_reward, prev_info):
        return f"\n--- Episode {ep_idx + 1}/{num_episodes} --- Reward: {prev_reward:.4f}\nSame constraint applies."

register_prompt("my_env", NoInfoPrompt)
register_prompt("my_env", FullInfoPrompt)
```

## Step 4 — Add Feedback Handlers

Every environment registers **two** feedback handlers: `standard` (the **default** — reveals
only success and score) and `information` (additionally reveals the ground-truth outcome at the
end of each episode, regardless of success, for maximum cross-task learning signal).

```python
# feedbacks.py
from latentgym.core.feedback import FeedbackHandler
from latentgym.core.registry import register_feedback

class StandardFeedback(FeedbackHandler):
    """Default — no ground truth revealed."""
    id = "standard"

    def format_step_feedback(self, raw_feedback, episode_idx, turn, info):
        return raw_feedback  # pass through

    def format_episode_end_feedback(self, episode_idx, episode_reward, episode_info):
        return f"Episode {episode_idx} done. Reward: {episode_reward:.4f}"

class InformationFeedback(FeedbackHandler):
    """Reveals the ground-truth outcome each episode for maximum learning signal."""
    id = "information"

    def format_step_feedback(self, raw_feedback, episode_idx, turn, info):
        return raw_feedback

    def format_episode_end_feedback(self, episode_idx, episode_reward, episode_info):
        target = episode_info.get("target", "?")
        return f"Episode {episode_idx} done. Reward: {episode_reward:.4f}. Answer: {target}"

register_feedback("my_env", StandardFeedback)
register_feedback("my_env", InformationFeedback)
```

## Step 5 — Register the Env

```python
# __init__.py
from latentgym.core.registry import register_env
from .core_env import MyGameEnv

# Flat kwargs: all params are top-level, including max_turns_per_episode
register_env("my_env", MyGameEnv, max_turns_per_episode=10)
```

Then add your package to `latentgym/envs/__init__.py`:

```python
# latentgym/envs/__init__.py
from . import my_env  # noqa — triggers registration
```

## Step 5b — Register with SkyRL (for training)

Add your env to `latentgym/register_skyrl.py` so it can be used with the SkyRL training pipeline:

```python
# In latentgym/register_skyrl.py, add:
register_skyrl_env("benchmark_my_env", "my_env")
```

This makes the env available as `benchmark_my_env` in SkyRL's `skyrl_gym.make()` system.

## Step 6 — Add Trajectory Generator

The trajectory generator creates the ground-truth JSON files used by the orchestrator.

```python
# trajectory_generator.py
import json, random
from pathlib import Path
from latentgym.core.latent import LatentDefinition
from latentgym.core.trajectory_utils import TrajectoryManifest

def generate_trajectories(
    latent_def: LatentDefinition,
    n_trajectories: int,
    num_episodes: int,
    output_dir: str,
    seed: int = 42,
):
    random.seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    files = []
    for traj_idx in range(n_trajectories):
        ctx = {}
        episode_configs = []
        for ep_idx in range(num_episodes):
            if latent_def.generator_fn:
                cfg = latent_def.generator_fn({}, ep_idx, num_episodes, ctx)
            elif latent_def.filter_fn:
                cfg = _sample_filtered(latent_def.filter_fn)
            else:
                cfg = latent_def.episode_configs[ep_idx % len(latent_def.episode_configs)]
            episode_configs.append(cfg)

        traj_data = {"episode_configs": episode_configs, "latent_id": latent_def.id}
        fname = f"traj_{traj_idx:04d}.json"
        with open(Path(output_dir) / fname, "w") as f:
            json.dump(traj_data, f)
        files.append(fname)

    # Write manifest
    manifest = TrajectoryManifest(trajectory_files=files, latent_id=latent_def.id)
    manifest.save(output_dir)
```

## Step 7 — Test It

```python
# Quick sanity check
from latentgym.core.registry import make_env
from latentgym.core.env_config import FullyDefinedEnv
from latentgym.core.reward import RewardType
import latentgym.envs.my_env  # noqa

config = FullyDefinedEnv(
    env_name="my_env",
    latent_id="easy",
    prompt_id="full_info",
    feedback_id="standard",
    num_episodes=3,
    reward_type=RewardType.CUMULATIVE,
)

# Create a dummy trajectory file first
import json, tempfile, os
traj = {"episode_configs": [{"target": 7}, {"target": 3}, {"target": 9}]}
with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
    json.dump(traj, f)
    traj_path = f.name

env = make_env(config, trajectory_path=traj_path)
messages, info = env.init(prompt=None)
print("Initial messages:", len(messages))

# Simulate one step
output = env.step("[7]")
print("Step output done:", output.done)

os.unlink(traj_path)
```

## Monolithic Alternative

For simple envs, put everything in one file:

```
latentgym/envs/_monolithic_example/monolithic_env.py
```

This file shows the exact same pattern (env + latents + prompts + feedbacks + registration)
all in one place. See it as a reference template.

## File Checklist

- [ ] `core_env.py` — implements `SingleEpisodeEnv` with no-arg `__init__`, params from `episode_config`
- [ ] `latents.py` — calls `register_latent()` for all latents (include `max_turns_per_episode` in configs)
- [ ] `prompts.py` — calls `register_prompt()` (convention: `no_info`, `some_info`, `full_info`)
- [ ] `feedbacks.py` — registers both `standard` (default) and `information`
- [ ] `trajectory_generator.py` — `generate_trajectories()` function
- [ ] `__init__.py` — calls `register_env()` with flat kwargs
- [ ] Add `from . import my_env` in `latentgym/envs/__init__.py`
- [ ] Add `register_skyrl_env()` call in `latentgym/register_skyrl.py`
