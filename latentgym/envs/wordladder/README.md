# Word Ladder

Transform a start word into a target word by changing exactly one letter at a time, with each
intermediate word a valid English word. Across episodes, the structure of the solution path
(its hub word, allowed vocabulary, substitution style, or word family) follows a latent, so an
agent that infers the latent can plan paths faster.

- **Latent mode:** generator + filter
- **Defaults:** up to 10 turns/episode, 5 episodes
- **Latents:** 20 · **Prompts:** `no_info`, `some_info`, `full_info` · **Feedbacks:** `standard`, `information`
- Wraps `textarena.envs.WordLadder.env.WordLadderEnv`.

## Game Dynamics

- A start word and a target word of equal length are given. Each turn the agent submits one word
  `[word]` that differs from the current word by exactly one letter and is a valid English word.
- Invalid moves (wrong length, not a real word, more than one letter changed) are rejected.
- **Reward:** if solved, `max(0, 1.0 − (turns_used − optimal_steps) × 0.03)` — reaching the
  target in the optimal number of steps scores 1.0, with a small penalty per excess turn. If not
  solved by the turn limit, partial credit = `matching_letters / word_length`.

## Latent Mode: Generator + Filter

Latents constrain the **solution path**, not just the start/target pair. Generator latents build
pairs with a chosen structure (a shared hub word, a restricted vocabulary); filter latents select
pairs whose optimal path has a given property. The trajectory JSON stores
`{"start_word", "target_word", "optimal_path", "optimal_steps"}` per episode.

## Latent Catalog (20)

### Generator (6)
| ID | Complexity | Description |
|---|---|---|
| `hub_word_3letter` / `hub_word_4letter` / `hub_word_5letter` | Hard | All word pairs (of that length) share a common **hub word** on their optimal path; the hub varies per trajectory. Discover it to split each puzzle into two easier halves. |
| `restricted_vocab_3letter` / `restricted_vocab_4letter` / `restricted_vocab_5letter` | Medium | All pairs (of that length) are solvable using only ~40 specific words; the vocabulary varies per trajectory. Learn which words are "in play". |

### Filter (14)
| ID | Complexity | Description |
|---|---|---|
| `order_left_to_right` | Hard | An optimal path changes positions left-to-right (0,1,2,…) |
| `order_right_to_left` | Hard | An optimal path changes positions right-to-left (…,2,1,0) |
| `order_outside_in` | Hard | An optimal path changes outer positions first, then inner |
| `subs_vowel_swaps` | Hard | An optimal path only changes vowels (a↔e↔i↔o↔u) |
| `subs_consonant_swaps` | Hard | An optimal path only changes consonants |
| `subs_alternating` | Hard | An optimal path alternates vowel and consonant changes |
| `subs_phonetic_group` | Hard | Consonant swaps stay within phonetic groups (b/p/d/t, f/v/s/z, m/n/l/r) |
| `family_contains_or` / `family_contains_an` / `family_contains_at` | Medium | All intermediate words contain `or` / `an` / `at` |
| `family_ends_e` / `family_ends_d` | Medium | All intermediate words end with `e` / `d` |
| `family_pattern_cvcc` | Hard | All intermediate words follow a consonant-vowel-consonant-consonant pattern |
| `family_pattern_cvcv` | Hard | All intermediate words follow a consonant-vowel-consonant-vowel pattern |

## Prompt Variants

| ID | Info level |
|---|---|
| `no_info` | Standard word-ladder rules, no mention of path patterns |
| `some_info` | Hints that solution paths may share a structural property across episodes |
| `full_info` | Explicitly states all pairs share a hidden path constraint |

## Feedback Variants

| ID | After each episode the agent sees… |
|---|---|
| `standard` (default) | Whether it reached the target, and the score |
| `information` | The above **plus** ground-truth path information, every episode regardless of success |

## Episode Flow

```
Episode (up to 10 turns)   Start: "cord" → Target: "warm"
  Agent: "[word]" (cord→word, one letter)   Env: "Current: word → Target: warm"   done = False
  ...
  Agent: "[warm]"   Env: "You reached the target word!"   done = True
    reward = max(0, 1.0 - (turns - optimal_steps) × 0.03)
  Out of turns → reward = matching_letters / word_length

  Ground truth (trajectory JSON): {"start_word": "cord", "target_word": "warm",
                                   "optimal_path": [...], "optimal_steps": 4}
```

## Usage

```python
import latentgym.envs.wordladder
from latentgym.core import FullyDefinedEnv, make_env

fd = FullyDefinedEnv(env_name="wordladder", latent_id="hub_word_4letter",
                     prompt_id="full_info", feedback_id="standard", num_episodes=5)
env = make_env(fd, trajectory_path="traj.json")               # standard
env = make_env(fd, seed=42, candidate_pool_path="pairs.txt")  # quick test (filter pool)
```

## Trajectory Generation

```python
from latentgym.envs.wordladder.trajectory_generator import generate_wordladder_trajectories

generate_wordladder_trajectories(
    latent_id="hub_word_4letter", num_episodes=5, n_trajectories=100,
    seed=42, output_dir="data/eval/wordladder_hub4/", env_params={"max_turns": 20},
)
```

## Latent Examples

**hub_word_4letter** — every pair's optimal path passes through one shared hub word (e.g. all
paths route through `core`); once the agent finds the hub, each puzzle becomes start→hub→target.

**family_contains_or** — every intermediate word contains `or`, e.g. `cord → core → bore → born`;
the agent learns to stay inside the `or` family rather than searching the full graph.
