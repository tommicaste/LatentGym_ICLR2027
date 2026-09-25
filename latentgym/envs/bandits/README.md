# Bandits

Multi-armed bandit with 5 buttons. The agent explores buttons to learn their hidden reward
probabilities, then locks in a final answer — selecting the best button correctly and **early**
earns more reward. Across a sequence of episodes the best-button pattern follows a latent, so an
agent that infers the latent can skip exploration and select immediately.

- **Latent mode:** generator
- **Defaults:** 5 buttons (`red, blue, green, yellow, purple`), up to 10 turns/episode, 10 episodes
- **Latents:** 29 · **Prompts:** `no_info`, `some_info`, `full_info` · **Feedbacks:** `standard`, `information`
- Wraps `textarena.envs.Bandit.env.BanditEnv`.

## Game Dynamics

- Each button has a hidden probability of returning reward 1 (else 0). Probabilities are the
  episode ground truth, set by the latent.
- Each turn the agent either **explores** (`[red]` — press a button, observe 0/1) or **locks in**
  (`[select red]` — end the episode immediately).
- **Reward:** a correct final selection earns `max(0, 1.0 − turns_used × 0.015)`; a wrong
  selection earns `0.0`. Selecting the best button earlier therefore scores higher.
- If the agent never selects, its last explore action on the final turn counts as the selection.

## Latent Mode: Generator

Each latent is a rule for how the best button (and the probability vector) is assigned across
episodes. The generator produces `{"ground_truth": {"red": 0.7, "blue": 0.3, ...}}` per episode.

## Latent Catalog (29)

### Easy (14)
| ID | Description |
|---|---|
| `loyal_favorite_0`..`loyal_favorite_4` | Best arm is always button index 0,1,2,3,4 (mod num_buttons) |
| `binary_switch_0_1` | Best arm is randomly button 0 or button 1 |
| `binary_switch_0_last` | Best arm is randomly the first or last button |
| `clockwise_rotation` | Best arm shifts +1 index each episode (0→1→2→…) |
| `counterclockwise_rotation` | Best arm shifts −1 index each episode (…→2→1→0) |
| `even_indices_only` | Best arm is always at an even index (0, 2, 4, …) |
| `odd_indices_only` | Best arm is always at an odd index (1, 3, 5, …) |
| `one_hot` | Best button has probability 1.0, all others 0.0 — one sample per button suffices |
| `fixed_probabilities` | Exact same probability vector every episode — memorize once |
| `bottom_excluded` | One specific button always has probability 0 — learn never to pick it |

### Medium (6)
| ID | Description |
|---|---|
| `ping_pong` | Best arm oscillates between first and last (0→N−1→0→N−1) |
| `shadow` | Best arm in episode K is the worst arm from episode K−1 |
| `same_ranking` | Button ranking never changes, but exact probabilities vary |
| `top_two_fixed` | The top 2 buttons are always the same pair — only explore between them |
| `swap_top_two` | The top two buttons swap #1 vs #2 each episode — alternate |
| `cycle_length_5` | Each button is best exactly once per 5-episode cycle |

### Hard (5)
| ID | Description |
|---|---|
| `skip_2` | Best index skips by 2 each episode (0→2→4→…) |
| `skip_3` | Best index skips by 3 each episode (0→3→6→…) |
| `random_walk` | Best index does a random walk (+1 or −1) from the previous episode |
| `cold_hand` | The previous episode's best button is never the next best — exclude it |
| `hot_hand` | 80% chance the best button stays the same as last episode — trust recent history |

### Very Hard (4)
| ID | Description |
|---|---|
| `mirror_mode` | If episode K is button X, episode K+1 is button N−1−X |
| `fibonacci` | Best index follows the Fibonacci sequence mod N (1,1,2,3,5,8,…) |
| `prime_indices` | Best arm is always at a prime index (2, 3, 5, 7, …) |
| `triangular` | Best index follows triangular numbers mod N (0,1,3,6,10,…) |

## Prompt Variants

| ID | Info level |
|---|---|
| `no_info` | No mention of patterns or multi-episode structure |
| `some_info` | Hints that a recurring pattern may exist across episodes |
| `full_info` | Explicitly states there is a hidden pattern to discover |

## Feedback Variants

| ID | After each episode the agent sees… |
|---|---|
| `standard` (default) | Which button it selected, whether it was correct, and the score |
| `information` | The above **plus** the full ground-truth probability vector and the best button, every episode regardless of success |

## Episode Flow

```
Episode (up to 10 turns)
  Explore turns:
    Agent: "[red]"         Env: "You pressed red. Reward: 1"   (reward 0.0 so far)
    Agent: "[blue]"        Env: "You pressed blue. Reward: 0"
    ...
  Lock in:
    Agent: "[select red]"  Env: "You selected 'red' on turn 4. Correct!"
    reward = max(0, 1.0 - 4 × 0.015) = 0.94, episode ends

  Ground truth (from trajectory JSON): {"red": 0.9, "blue": 0.1, ...}
  Best button = red (highest probability)
```

## Usage

```python
import latentgym.envs.bandits
from latentgym.core import FullyDefinedEnv, make_env

fd = FullyDefinedEnv(env_name="bandits", latent_id="loyal_favorite_0",
                     prompt_id="full_info", feedback_id="standard", num_episodes=10)

env = make_env(fd, trajectory_path="traj.json")  # standard: pre-generated ground truth
env = make_env(fd, seed=42)                       # quick test: generator resolves on the fly
```

## Trajectory Generation

```python
from latentgym.envs.bandits.trajectory_generator import generate_bandit_trajectories

generate_bandit_trajectories(
    latent_id="loyal_favorite_0", num_episodes=10, n_trajectories=100,
    seed=42, output_dir="data/eval/bandits_loyal_favorite_0/",
)
```

## Latent Examples

**Clockwise Rotation** — best button advances one index per episode:
```
Episode 0: red best (idx 0)   Episode 1: blue best (idx 1)   Episode 2: green best (idx 2)
```

**Ping-Pong** — best arm bounces between the ends:
```
Episode 0: red (idx 0)   Episode 1: purple (idx 4)   Episode 2: red (idx 0)   ...
```

**Shadow** — best arm in episode K is the worst arm from episode K−1 (requires tracking the
previous episode's outcome).
