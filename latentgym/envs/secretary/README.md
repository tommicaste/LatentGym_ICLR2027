# Secretary

Optimal stopping. The agent sees values one at a time and must `[accept]` the current one or
`[continue]` to the next — with no going back. The goal is to accept the maximum of the
sequence. Across episodes, the position or magnitude of the max follows a latent, so an agent
that infers the latent can stop at the right moment instead of relying on the classic 1/e rule.

- **Latent mode:** generator
- **Defaults:** 10 draws/episode, up to 10 turns/episode, 10 episodes
- **Latents:** 41 · **Prompts:** `no_info`, `some_info`, `full_info` · **Feedbacks:** `standard`, `information`
- Wraps `textarena.envs.Secretary.env.SecretaryEnv`.

## Game Dynamics

- Each episode presents `num_draws` values (default 10), one per turn.
- Each turn the agent outputs `[accept]` (claim the current value, episode ends) or `[continue]`
  (skip to the next). Reaching the last value forces acceptance. Invalid output is treated as
  `[continue]`.
- **Reward:** `1.0` if the accepted value is the sequence maximum; otherwise partial credit
  `0.5 × (accepted_value / max_value)`. This leaves a clear gap between finding the max (1.0)
  and a near miss (~0.47).

## Latent Mode: Generator

Each latent builds the value sequence per episode, sharing a `context` dict across episodes so
cross-episode patterns can track state. The trajectory JSON stores `{"draws": [...]}` per episode.

## Latent Catalog (41)

### Easy (13)
| ID | Description |
|---|---|
| `best_is_first` | Optimal candidate is always the first presented |
| `best_is_last` | Optimal candidate is always the last presented |
| `fixed_high` | Best candidate always has value exactly 1.0 — accept any 1.0 immediately |
| `first_half_bias` | Best candidate always in the first 50% of the sequence |
| `second_half_bias` | Best candidate always in the second 50% of the sequence |
| `fixed_position_0`..`fixed_position_4` | Best candidate always at index 0,1,2,3,4 (mod N) |
| `best_at_even` | Best candidate always at an even index (0, 2, 4, …) |
| `best_at_odd` | Best candidate always at an odd index (1, 3, 5, …) |
| `threshold_05` | Max always above 0.5, all others below — accept any value > 0.5 |

### Medium (12)
| ID | Description |
|---|---|
| `alternating_position` | Best is early in even episodes, late in odd episodes |
| `position_shift` | Best position shifts +1 each episode (0→1→2→…) |
| `countdown` | Best position decreases by 1 each episode (9→8→7→…) |
| `third_rotation` | Best position rotates through thirds (first→middle→last→…) |
| `position_cycle_258` | Max position cycles 2→5→8→2→… |
| `position_cycle_1379` | Max position cycles 1→3→7→9→1→… |
| `threshold_08` | Max always above 0.8, all others below |
| `threshold_06` | Max always above 0.6, all others below |
| `best_in_quarter_0`..`best_in_quarter_3` | Max always in the first/second/third/fourth quarter |

### Hard (10)
| ID | Description |
|---|---|
| `step_function` | Values in sorted chunks (Low→Medium→High) — wait for the high chunk |
| `inverse_order` | Strictly decreasing — accept immediately |
| `sorted_order` | Strictly increasing — always continue to the last |
| `prime_positions` | Best always at a prime index (2, 3, 5, 7) |
| `valley_pattern` | Values form a valley (High→Low→High) — best is at one of the ends |
| `mountain_pattern` | Values form a mountain (Low→High→Low) — best is in the middle |
| `relative_jump` | Max always follows a large (>0.3) jump — accept after big jumps |
| `early_decoy` | First few values are high (~0.8) but the max is in the second half |
| `max_after_min` | Max appears 1–2 positions after the minimum |
| `ascending_spike` | Values rise slowly then spike at the max — accept the spike |

### Very Hard (6)
| ID | Description |
|---|---|
| `fibonacci_positions` | Best always at a Fibonacci index (1, 2, 3, 5, 8) |
| `modular_pattern` | Best position equals `episode_idx mod N` |
| `mirror_episodes` | If episode K has max at index i, episode K+1 has it at N−1−i |
| `random_walk_position` | Best position does a random walk (+1/−1) from the previous episode |
| `same_position_streak` | Max stays at the same position for 3 episodes, then shifts |
| `increasing_position` | Max position increases each episode (clamps at N−1) |

## Prompt Variants

| ID | Info level |
|---|---|
| `no_info` | Standard secretary rules, no mention of sequence patterns |
| `some_info` | Hints that the position of the maximum may follow a pattern |
| `full_info` | Explicitly states value sequences share a hidden structural pattern |

## Feedback Variants

| ID | After each episode the agent sees… |
|---|---|
| `standard` (default) | Whether it accepted the maximum, and the score |
| `information` | The above **plus** the ground-truth sequence/max, every episode regardless of success |

## Episode Flow

```
Episode (10 draws default)
  Env:   "The current value is 0.43. [accept] or [continue]?"
  Agent: "[continue]"        done = False
  ...
  Env:   "The current value is 0.91. [accept] or [continue]?"
  Agent: "[accept]"          Env: "You accepted 0.91 — the maximum!"  reward = 1.0, done

  Accept a non-max value → partial reward 0.5 × (accepted / max)
  Ground truth (trajectory JSON): {"draws": [0.43, 0.67, 0.91, ...]}
```

## Usage

```python
import latentgym.envs.secretary
from latentgym.core import FullyDefinedEnv, make_env

fd = FullyDefinedEnv(env_name="secretary", latent_id="best_is_last",
                     prompt_id="full_info", feedback_id="standard", num_episodes=10)
env = make_env(fd, trajectory_path="traj.json")  # standard
env = make_env(fd, seed=42)                       # quick test (generator on the fly)
```

## Trajectory Generation

```python
from latentgym.envs.secretary.trajectory_generator import generate_secretary_trajectories

generate_secretary_trajectories(
    latent_id="best_is_last", num_episodes=10, n_trajectories=100,
    seed=42, output_dir="data/eval/secretary_best_is_last/", env_params={"num_draws": 10},
)
```

## Latent Examples

**best_is_last (num_draws=5)** — max always at the final position; optimal: always continue:
```
Episode 0: [0.32, 0.45, 0.28, 0.61, 0.95]   max at idx 4
Episode 1: [0.41, 0.22, 0.67, 0.33, 0.88]   max at idx 4
```

**mountain_pattern** — values peak in the middle, so the max is near the center.
**threshold_06** — the max is always > 0.6 while everything else is below; accept the first value over 0.6.
