# Benchmark Environments

Each environment poses a *sequence* of related tasks whose shared structure is governed by a
hidden **latent**. The agent is rewarded for inferring that latent from experience and
exploiting it on later tasks. An environment is fully defined by five independently swappable
axes:

```
FullyDefinedEnv = core-env × latent × prompt × feedback × num_episodes
```

See [`../README.md`](../README.md) for the end-to-end pipeline (data generation → eval →
training) and the paper (§4) for the framing and metrics.

## The Five Axes

| Axis | What it controls | Values |
|---|---|---|
| **core-env** | The within-task game dynamics | `bandits`, `wordle`, `hangman`, `mastermind`, `secretary`, `wordladder`, `number_guessing` |
| **latent** | The hidden regularity shared across tasks in a sequence | per-env catalogs (see each env page) |
| **prompt** | How much the agent is told about the latent up front | `no_info`, `some_info`, `full_info` |
| **feedback** | What is added to the cross-task history after each task | `standard` (default), `information` |
| **num_episodes** | The horizon — how many tasks per sequence (adaptation budget) | integer (per-env default) |

**Latent modes.** A latent produces episode configs in one of two ways:
- **generator** — generates configs directly from a rule (e.g. bandit probability vectors).
- **filter** — selects targets from a candidate pool by a predicate (e.g. Wordle words with 2 vowels).

Both require a seed and are resolved into trajectory JSONs at data-generation time; at runtime
the env just loads the JSON.

**Prompt** sets the agent's prior over the latent: `no_info` gives no indication structure
exists, `some_info` hints that recurring structure may exist, `full_info` describes the latent
explicitly.

**Feedback** governs how fast evidence accumulates:
- `standard` — after each task the agent sees only success/failure and its score.
- `information` — additionally reveals the ground-truth outcome (e.g. the target number, the
  bandit probability vector and best button) **every** episode, regardless of success.

## Environment Suite

7 environments, **421 latents** total. All environments run **up to 10 turns per episode**;
episode counts are per-env registry defaults. Both are overridable per run.

| Env | Latent mode | # Latents | Default episodes | Core game |
|---|---|---:|---:|---|
| [bandits](bandits/README.md) | generator | 29 | 10 | 5-button multi-armed bandit |
| [wordle](wordle/README.md) | filter | 165 | 5 | Guess a 5-letter word (G/Y/X feedback) |
| [hangman](hangman/README.md) | filter | 105 | 10 | Reveal a word letter by letter |
| [mastermind](mastermind/README.md) | filter | 54 | 10 | Infer a 4-digit code (6 symbols, no dups) |
| [secretary](secretary/README.md) | generator | 41 | 10 | Optimal stopping over 10 draws |
| [wordladder](wordladder/README.md) | generator + filter | 20 | 5 | Transform one word into another |
| [number_guessing](number_guessing/README.md) | generator | 7 | 7 | Guess a number in a range |

> Note: the paper reports 442 latents; the suite has since changed (see each env page for the
> current catalog). Treat the registry as ground truth — query it with
> `registry.list_latents(env)`.

## Difficulty Axes

Three orthogonal difficulty dimensions (paper Table 1), illustrated for number guessing:

| Difficulty axis | Set by | Example (number guessing) |
|---|---|---|
| Within-task | Visible range | `[1,100]`, `[1,1000]`, `[1,10000]` |
| Latent-identification | Size of hidden set `z` | `|z|=2`, `|z|=5`, `|z|=10` |
| Cross-task | Latent + within-task difficulty | small `|z|` over a wide range = highly predictive |

Prompt and feedback modulate these orthogonally: a richer prompt lowers latent-identification
difficulty; richer feedback accelerates evidence accumulation. The horizon `N` sets the budget.

## Per-Environment Pages

Each environment has its own page with game dynamics, latent catalog, prompt/feedback variants,
episode flow, and usage:

- [bandits](bandits/README.md) · [wordle](wordle/README.md) · [hangman](hangman/README.md) ·
  [mastermind](mastermind/README.md) · [secretary](secretary/README.md) ·
  [wordladder](wordladder/README.md) · [number_guessing](number_guessing/README.md)

## Adding a New Environment

See [adding_environments.md](adding_environments.md) for the full step-by-step guide
(implement `SingleEpisodeEnv`, define latents/prompts/feedbacks, register, add a trajectory
generator, wire up SkyRL training).

---

## Developer Reference

### Package layout

Each environment is a self-contained package:

```
envs/<name>/
├── __init__.py               # Imports + registers all components (flat kwargs)
├── core_env.py               # SingleEpisodeEnv subclass (game dynamics)
├── latents.py                # LatentDefinition registrations
├── prompts.py                # PromptTemplate variants (no_info, some_info, full_info)
├── feedbacks.py              # FeedbackHandler variants (standard, information)
├── trajectory_generator.py   # Generates trajectory JSONs for this env
└── README.md                 # Env page (dynamics, latent catalog)
```

### How an env connects to the rest of the system

```
envs/bandits/__init__.py
    │
    ├── import core_env    ──► register_env("bandits", BanditSingleEpisodeEnv, ...)
    ├── import latents     ──► register_latent("bandits", LatentDefinition(...)) × 29
    ├── import prompts     ──► register_prompt("bandits", BanditNoInfoPrompt) × 3
    └── import feedbacks   ──► register_feedback("bandits", BanditStandardFeedback) × 2
                │
                ▼
        core/registry.py stores all registrations
                │
                ▼
        make_env(FullyDefinedEnv) resolves from registry
                │
                ▼
        MultiEpisodeEnv ready to play
```

### Core env architecture (Option B)

All core_envs follow Option B: **no-arg constructor, all params from `episode_config`**.

- `def __init__(self):` takes no arguments. No game params in the constructor.
- `def reset(self, episode_config)` receives all game parameters per episode.
- Registry stores `env_params` as flat kwargs for latent generators and metadata only.
- Env instances are **reused across episodes** — `_env_params_key` tracks constructor params
  and only recreates the underlying (e.g. TextArena) env when they change. `.reset()` is called
  per episode, not `__init__`.
- `max_turns_per_episode` is the standard key in `episode_config`. Old env-specific keys
  (`max_attempts`, `num_turns`, `max_turns`) still work as fallbacks.

### TextArena dependency

Six environments wrap TextArena game classes; `number_guessing` is self-contained.

- `BanditSingleEpisodeEnv` wraps `textarena.envs.Bandit.env.BanditEnv`
- `WordleSingleEpisodeEnv` wraps `textarena.envs.Wordle.env.WordleEnv`
- `HangmanSingleEpisodeEnv` wraps `textarena.envs.Hangman.env.HangmanEnv`
- `MastermindSingleEpisodeEnv` wraps `textarena.envs.Mastermind.env.MastermindEnv`
- `SecretarySingleEpisodeEnv` wraps `textarena.envs.Secretary.env.SecretaryEnv`
- `WordLadderSingleEpisodeEnv` wraps `textarena.envs.WordLadder.env.WordLadderEnv`

The dependency is contained within each `core_env.py` — nothing else in the benchmark touches
TextArena directly. Hangman, WordLadder, and Wordle use a class-level `_word_list_cache` so word
lists load once across instances, with try/except for the nltk `words` download.

### Special environments

- **`bandit_generator_latent_env/`** — example of the **generator path** without trajectory
  JSONs: `make_env(fd, seed=42)` resolves ground truth on the fly. Standard benchmark envs use
  the trajectory-JSON path; this shows the alternative for quick testing.
- **`_monolithic_example/`** — reference template implementing an entire env (core_env + latents
  + prompts + feedbacks + registration) in a single file, for cases where separate files are
  overkill.

### Three ways to run an environment

`make_env()` supports three modes:

```
1. make_env + trajectory_path (standard — all envs, eval + training)
   Trajectory JSONs contain pre-generated ground truth.
   make_env(fd, trajectory_path="traj.json")

2. make_env + seed (quick testing — generator latents only)
   Ground truth generated on the fly from seed.
   make_env(fd, seed=42)

3. make_env + seed + pool (quick testing — filter latents only)
   Target sampled from filtered pool on the fly.
   make_env(fd, seed=42, candidate_pool_path="words.txt")
```

Modes 2 and 3 skip the trajectory-JSON layer; they're for debugging and one-offs. The standard
benchmark pipeline (data generation → eval / training) always uses mode 1.

### Prompt composition (layered)

The system prompt is composed from independent layers:

- `core_env.get_game_rules()` — env-specific, constant across prompt variants.
- `prompt_template.initial_system_prompt()` — varies by `prompt_id` (how much latent info).
- Episode header + initial observation — structural.

Feedback has three distinct points:
- **Intra-episode**: `feedback_handler.format_step_feedback()` — after each turn.
- **End-of-episode**: `feedback_handler.format_episode_end_feedback()` — when an episode completes.
- **Episode transition**: `prompt_template.episode_transition_message()` — between episodes.

```
┌────────────────────────────────────────────────────────────────────┐
│                     SYSTEM PROMPT LAYERS                           │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ core_env.get_game_rules()                                │ fixed│
│  │ "You are playing a Multi-Armed Bandit game with 5..."    │      │
│  ├──────────────────────────────────────────────────────────┤      │
│  │ prompt_template.initial_system_prompt()                  │varies│
│  │ "Reward probabilities share a hidden pattern..."         │prompt│
│  ├──────────────────────────────────────────────────────────┤      │
│  │ "--- Episode 1 of 10 ---"                                │ auto │
│  │ "You are in a room with 5 buttons..."                    │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                    │
│  FEEDBACK POINTS:                                                  │
│  ┌──────────────────┐ ┌─────────────────┐ ┌────────────────────┐   │
│  │ Intra-episode    │ │ End-of-episode  │ │ Episode transition │   │
│  │ (every turn)     │ │ (last ep only)  │ │ (between episodes) │   │
│  │ feedback_handler │ │ feedback_handler│ │ prompt_template    │   │
│  │ varies by        │ │ varies by       │ │ varies by          │   │
│  │ feedback_id      │ │ feedback_id     │ │ prompt_id          │   │
│  └──────────────────┘ └─────────────────┘ └────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

### MultiEpisodeEnv levels

- **Homogeneous** (primary): the same `SingleEpisodeEnv` for all episodes, different configs.
- **Heterogeneous** (advanced): different `SingleEpisodeEnv` per episode via an `EpisodeDef` list.
  Only works through `from_episode_defs()`, not through `make_env` / SkyRL.

```
Homogeneous (primary):
  ep0: BanditEnv + config0    ep1: BanditEnv + config1    ep2: BanditEnv + config2
       ^^^^^^^^                     ^^^^^^^^                    ^^^^^^^^
       same env class               same env class              same env class
  Supported by: make_env, SkyRL, orchestrator, parquet

Heterogeneous (advanced):
  ep0: BanditEnv + config0    ep1: WordleEnv + config1    ep2: BanditEnv + config2
       ^^^^^^^^                     ^^^^^^^^                    ^^^^^^^^
       different env classes
  Supported by: from_episode_defs() only
```
