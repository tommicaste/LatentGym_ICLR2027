# Number Guessing

Guess a hidden integer in a range; after each guess the agent is told whether the target is
greater than, less than, or equal to the guess. Across episodes the targets are drawn according
to a latent (a small fixed set, a narrow sub-range, etc.), so an agent that infers the latent can
skip binary search and guess a likely value directly.

- **Latent mode:** generator (self-contained — no TextArena dependency)
- **Defaults:** range `[1, 1000]`, up to 10 turns/episode, 7 episodes
- **Latents:** 7 · **Prompts:** `no_info`, `some_info`, `full_info` · **Feedbacks:** `standard`, `information`

## Game Dynamics

- Each turn the agent submits a guess, ideally bracketed (e.g. `[500]`); the parser takes the
  last bracketed number, falling back to the last bare number.
- Feedback is `greater than` / `less than` / `correct`. Out-of-range guesses are rejected and
  the turn is consumed.
- **Reward:** solved = `max(0, 1.0 − turns_used × 0.020)`; not solved within the turn limit = `0.0`.
  Solving in fewer turns scores higher, so exploiting the latent (fewer guesses) is rewarded.

## Latent Mode: Generator

Each latent generates the per-episode target (and the visible range), sharing context across the
sequence. The trajectory JSON stores `{"target_number", "min_range", "max_range"}` per episode.

## Latent Catalog (7)

| ID | Complexity | Description |
|---|---|---|
| `set_of_2` | Easy | All targets drawn from a fixed set of 2 specific numbers in `[1, 1000]` |
| `set_of_3` | Easy | All targets drawn from a fixed set of 3 specific numbers in `[1, 1000]` |
| `range_100` | Medium | All targets fall within a contiguous 100-number sub-range of `[1, 1000]` |
| `range_1000` | Medium | All targets fall within a contiguous 1000-number sub-range of `[1, 10000]` |
| `dynamic_range` | Hard | Targets within a 1000-number sub-range of a dynamic outer range |
| `dynamic_full_range` | Hard | Full prompt range is `[n, n+1000]` where `n` varies per trajectory |
| `two_ranges` | Hard | Targets from two disjoint 500-number ranges within `[1, 10000]` |

## Prompt Variants

| ID | Info level |
|---|---|
| `no_info` | Standard rules, no mention that targets follow a pattern |
| `some_info` | Hints that the numbers might follow a pattern across games |
| `full_info` | Explicitly states the numbers share a hidden pattern (with a latent-specific hint) |

## Feedback Variants

| ID | After each episode the agent sees… |
|---|---|
| `standard` (default) | Whether it found the number, and the score |
| `information` | The above **plus** the target number, every episode regardless of success |

## Episode Flow

```
Episode (up to 10 turns)
  Env:   "I'm thinking of a number between 1 and 1000. You have 10 guesses."
  Agent: "[500]"   Env: "The number is less than 500."
  Agent: "[250]"   Env: "Correct! You guessed the number 250 in 2 turns."
    reward = max(0, 1.0 - 2 × 0.020) = 0.96, done

  Ground truth (trajectory JSON): {"target_number": 250, "min_range": 1, "max_range": 1000}
```

## Usage

```python
import latentgym.envs.number_guessing
from latentgym.core import FullyDefinedEnv, make_env

fd = FullyDefinedEnv(env_name="number_guessing", latent_id="set_of_3",
                     prompt_id="full_info", feedback_id="standard", num_episodes=7)
env = make_env(fd, trajectory_path="traj.json")  # standard
env = make_env(fd, seed=42)                       # quick test (generator on the fly)
```

## Trajectory Generation

```python
from latentgym.envs.number_guessing.trajectory_generator import generate_number_guessing_trajectories

generate_number_guessing_trajectories(
    latent_id="set_of_3", num_episodes=7, n_trajectories=100,
    seed=42, output_dir="data/eval/number_guessing_set_of_3/",
)
```

## Latent Examples

**set_of_3** — every target is one of 3 fixed numbers (e.g. {592, 781, 926}). Early on the agent
binary-searches; once it notices the recurring values it solves later tasks in one guess.

**two_ranges** — targets come from two disjoint 500-number bands; the agent learns to probe both
bands rather than treat the full range as uniform.
