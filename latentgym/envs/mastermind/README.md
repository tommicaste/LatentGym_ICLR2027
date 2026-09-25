# Mastermind

Code-breaking. The agent guesses a hidden digit code and receives black-peg (right digit, right
position) and white-peg (right digit, wrong position) feedback. Across episodes the secret code
satisfies a structural latent, so an agent that infers the constraint can restrict its guesses
and crack the code in fewer turns.

- **Latent mode:** filter
- **Defaults:** code length 4, digits 1–6, no duplicates, up to 10 turns/episode, 10 episodes
- **Latents:** 54 · **Prompts:** `no_info`, `some_info`, `full_info` · **Feedbacks:** `standard`, `information`
- Wraps `textarena.envs.Mastermind.env.MastermindEnv`.

## Game Dynamics

- A secret code of `code_length` digits (each in `1..num_numbers`) is chosen by the latent.
- Each turn the agent guesses with `[1 2 3 4]` and receives black/white peg counts.
- **Reward:** `1.0` if the code is cracked; otherwise partial credit equal to the code's
  percentage completion at the turn limit.

## Latent Mode: Filter

The generator enumerates all valid codes for `(code_length, num_numbers, duplicates_allowed)`,
keeps those passing the latent's predicate, and samples secret codes from the filtered set. The
trajectory JSON stores `{"secret_code", "code_length", "num_numbers", "duplicates_allowed"}` per
episode. Some latents require duplicates (noted below); the generator sets `duplicates_allowed=True`
for them automatically.

## Latent Catalog (54)

### Easy (19)
`ascending`, `descending`, `consecutive`, `first_not_equals_last`, `all_same`†, `no_adjacent_same`,
`no_repeats`, `contains_1`…`contains_6`, `mixed_parity`, `has_extreme` (contains 1 or 6),
`first_less_than_last`, `first_greater_than_last`, `sum_even`, `sum_odd`

### Medium (32)
`strictly_ascending`, `strictly_descending`, `first_equals_last`†, `palindrome`†, `alternating`† (A-B-A-B),
`has_pair`†, `adjacent_same`†, `middle_same`†, `no_1`, `no_6`, `all_odd` (1/3/5), `all_even` (2/4/6),
`all_low` (1–3), `all_high` (4–6), `no_extremes` (no 1 or 6), `first_is_1`…`first_is_6`,
`last_is_1`…`last_is_6`, `first_is_min`, `last_is_max`, `sum_low` (≤ len×2.5), `sum_high` (≥ len×4.5),
`sum_divisible_by_3`

### Hard (3)
`all_prime` (2/3/5), `no_prime` (1/4/6), `has_triple`† (a digit appears 3×)

† requires `duplicates_allowed=True`.

## Prompt Variants

| ID | Info level |
|---|---|
| `no_info` | Standard mastermind rules, no mention of code patterns |
| `some_info` | Hints that secret codes may share a structural property across episodes |
| `full_info` | Explicitly states all codes share a hidden structural pattern |

## Feedback Variants

| ID | After each episode the agent sees… |
|---|---|
| `standard` (default) | Whether it cracked the code, and the score |
| `information` | The above **plus** the ground-truth secret code, every episode regardless of success |

## Episode Flow

```
Episode (up to 10 turns)
  Agent: "[1 2 3 4]"   Env: "Black pegs: 1, White pegs: 2"   done = False
  ...
  Agent: "[2 4 1 3]"   Env: "4 black peg(s) — you cracked the code!"   done = True, reward = 1.0
  Out of turns → partial reward = code percentage completion

  Ground truth (trajectory JSON): {"secret_code": [2,4,1,3], "code_length": 4, ...}
```

## Usage

```python
import latentgym.envs.mastermind
from latentgym.core import FullyDefinedEnv, make_env

fd = FullyDefinedEnv(env_name="mastermind", latent_id="ascending",
                     prompt_id="full_info", feedback_id="standard", num_episodes=10)
env = make_env(fd, trajectory_path="traj.json")  # standard
env = make_env(fd, seed=42)                       # quick test (codes on the fly)
```

## Trajectory Generation

```python
from latentgym.envs.mastermind.trajectory_generator import generate_mastermind_trajectories

generate_mastermind_trajectories(
    latent_id="ascending", num_episodes=10, n_trajectories=100,
    seed=42, output_dir="data/eval/mastermind_ascending/",
    env_params={"code_length": 4, "num_numbers": 6, "duplicates_allowed": False},
)
```

## Latent Examples

**ascending** — `[1, 2, 4, 6]`; restricting guesses to non-decreasing codes shrinks the search
space from 360 (no-dup, 4-of-6) to ~15.

**first_is_1** — `[1, ?, ?, ?]`; the agent should notice the fixed first digit after a few episodes.

**all_prime** — only digits 2/3/5 (requires duplicates); a very small code pool.
