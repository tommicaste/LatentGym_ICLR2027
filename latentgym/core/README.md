# latentgym/core/ — Core Abstractions

All base classes and framework logic. Environments, eval, and data generation build on these.

## Module Structure and Connections

```
┌──────────────────────────────────────────────────────────────────────┐
│                            core/                                     │
│                                                                      │
│  ┌────────────────────┐          ┌─────────────────────┐            │
│  │ single_episode_env │◄─────────│ multi_episode_env   │            │
│  │ (ABC)              │  wraps   │ (BaseTextEnv)       │            │
│  └────────────────────┘          │                     │            │
│                                  │ uses all of:        │            │
│  ┌────────────┐ ┌────────────┐  │  - SingleEpisodeEnv │            │
│  │  prompt.py │ │ feedback.py│──►│  - PromptTemplate   │            │
│  │ (ABC)      │ │ (ABC)      │  │  - FeedbackHandler  │            │
│  └────────────┘ └────────────┘  │  - RewardAggregator │            │
│                                  │  - episode_configs  │            │
│  ┌────────────┐ ┌────────────┐  └──────────┬──────────┘            │
│  │  latent.py │ │ reward.py  │             │                        │
│  │ (dataclass)│ │ (class)    │─────────────┘                        │
│  └────────────┘ └────────────┘                                      │
│                                                                      │
│  ┌────────────────────┐   ┌──────────────────────┐                  │
│  │  env_config.py     │──►│  registry.py         │                  │
│  │  FullyDefinedEnv   │   │  make_env()          │                  │
│  └────────────────────┘   │  register_env/latent/ │                  │
│                           │  prompt/feedback      │                  │
│  ┌────────────────────┐   └──────────────────────┘                  │
│  │ trajectory_utils.py│                                             │
│  │ save/load helpers  │                                             │
│  └────────────────────┘                                             │
└──────────────────────────────────────────────────────────────────────┘
         │                              │                    │
         ▼                              ▼                    ▼
    ┌──────────┐                  ┌──────────┐         ┌──────────┐
    │  envs/   │                  │  eval/   │         │  data/   │
    │ implement│                  │ run envs │         │ generate │
    │ ABCs     │                  │ + collect│         │ parquets │
    └──────────┘                  └──────────┘         └──────────┘
```

## How make_env() Resolves Everything

```
make_env(FullyDefinedEnv, trajectory_path=None, seed=None, candidate_pool_path=None)
    │
    ├── Always from registry:
    │   ├── core_env      = _ENV_REGISTRY[env_name]()   # No-arg constructor (Option B)
    │   ├── prompt        = _PROMPT_REGISTRY[env_name][prompt_id]()
    │   ├── feedback      = _FEEDBACK_REGISTRY[env_name][feedback_id]()
    │   └── reward_agg    = RewardAggregator(reward_type)
    │
    │   Note: env_params stored in registry are for latent generators and
    │   metadata only — they are NOT passed to core_env constructor.
    │
    ├── Episode configs (depends on what's provided):
    │   ├── trajectory_path? ──► load from JSON file
    │   ├── seed + generator? ─► call generator_fn(seed) per episode
    │   └── seed + pool_path? ─► filter pool + sample(seed) per episode
    │
    └── MultiEpisodeEnv.from_configs(core_env, configs, prompt, feedback, reward)
```

## MultiEpisodeEnv Construction Paths

```
PATH 1: make_env()                     PATH 2: SkyRL __init__
(our eval/orchestrator)                (SkyRL training + eval)

  registry.py                           skyrl_gym.make()
       │                                     │
       ▼                                     ▼
  make_env(fd, traj_path)              __init__(env_config, extras)
       │                                     │
       ├─ resolve from registry              ├─ _init_from_skyrl_extras()
       ├─ load episode_configs               ├─ resolve from registry
       │                                     ├─ load trajectory JSON
       ▼                                     ▼
  from_configs(core_env, configs, ...)  self._core_env = env_cls()  # No args (Option B)
       │                                self._episode_configs = ...
       ▼                                     │
  MultiEpisodeEnv ready                      ▼
                                        MultiEpisodeEnv ready

Note: Both paths instantiate core_env with no arguments. All game
parameters come from episode_config in reset(). The env instance is
reused across episodes — _env_params_key tracks constructor params
and only recreates the env when they change.

Both paths produce IDENTICAL behavior (verified by tests).


PATH 3: from_episode_defs()
(heterogeneous — manual only)

  from_episode_defs([
      EpisodeDef(core_env=BanditEnv(), config={...}),
      EpisodeDef(core_env=WordleEnv(), config={...}),
  ], prompt, feedback, reward)
```

## Files

### single_episode_env.py — SingleEpisodeEnv ABC

Defines game dynamics for one episode. Does NOT inherit from skyrl's Env — only MultiEpisodeEnv does.

```python
class SingleEpisodeEnv(ABC):
    def reset(self, episode_config: Dict) -> str:     # Initial observation
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:  # feedback, reward, done, info
    def get_game_rules(self) -> str:                   # For layered prompt composition
    def close(self) -> None:
```

`get_game_rules()` returns the env-specific rules text used by the prompt composition system. It's the constant layer that doesn't change across prompt variants.

### Who Produces What Feedback

```
Source                   What                          When              Configurable by
──────                   ────                          ────              ───────────────
core_env.step()          raw game feedback              every turn       env_name (fixed)
feedback_handler         formatted step feedback        every turn       feedback_id
feedback_handler         end-of-episode summary         last ep only     feedback_id
prompt_template          episode transition message     between eps      prompt_id
core_env.reset()         next episode initial obs       between eps      env_name (fixed)
```

### latent.py — LatentDefinition + CrossEpisodeLatent

Two modes (no static mode — both need a seed):
- **generator_fn**: `(env_params, ep_idx, n_eps, context) → episode_config` (bandits)
- **filter_fn**: `(candidate) → bool` (wordle)

```python
LatentDefinition(id="loyal_favorite_0", generator_fn=..., complexity=EASY)
LatentDefinition(id="vowel_count_2", filter_fn=..., complexity=EASY)
```

Hyperparams support: `latent.with_hyperparams(speed=2)` returns a copy with updated params.

CrossEpisodeLatent sequences multiple base latents: alternating, cyclic, progressive patterns.

### Latent Mode Resolution — When and Where

```
Mode          When resolved              Where resolved                 make_env sees
────          ─────────────              ──────────────                 ─────────────
generator     data gen OR make_env       trajectory_gen OR registry     seed → configs
filter        data gen OR make_env       trajectory_gen OR registry     seed+pool → configs
trajectory    already done               trajectory_gen (earlier)       trajectory_path → load
```

### prompt.py — PromptTemplate ABC

```python
class PromptTemplate(ABC):
    def initial_system_prompt(self, game_rules, env_params, num_episodes) -> str
    def episode_transition_message(self, ep_idx, num_episodes, prev_reward, info) -> str
```

### feedback.py — FeedbackHandler ABC

Three distinct feedback points:
```python
class FeedbackHandler(ABC):
    def format_step_feedback(self, raw, ep_idx, turn, info) -> str        # Every turn
    def format_episode_end_feedback(self, ep_idx, reward, info) -> str     # Episode end (last ep only)
```

### reward.py — RewardAggregator

```python
RewardType.CUMULATIVE   # Sum of all episode rewards (was R_b)
RewardType.TERMINAL     # Last episode reward only (was R_d)
RewardType.IMPROVEMENT  # Last - first episode reward (was R_c)
RewardType.PER_EPISODE  # Each episode's reward at episode boundary
```

reward_type is primarily for training (shapes RL reward signal). Eval always computes all metrics regardless.

### env_config.py — FullyDefinedEnv + EpisodeDef

```python
FullyDefinedEnv(env_name, latent_id, prompt_id, feedback_id, num_episodes, reward_type)
```

`benchmark_id` property: `"bandits/loyal_favorite_0/full_info/standard/ep10"`

EpisodeDef is for heterogeneous mode (different core_env per episode).

### multi_episode_env.py — MultiEpisodeEnv(BaseTextEnv)

The central class. Inherits BaseTextEnv for SkyRL compatibility.

**Three construction paths:**
1. `from_configs()` — called by `make_env()`. Pre-resolved objects.
2. `__init__(env_config, extras)` — called by `skyrl_gym.make()`. Resolves from registry + trajectory JSON.
3. `from_episode_defs()` — heterogeneous mode. Manual construction.

**init() Approach C (hybrid prompt):**
- Empty/placeholder prompt → constructs own via layered composition
- Real prompt content → uses it, appends episode header + initial obs (backwards compat)

**Metadata recorded at init vs step:**

```
init() metadata:                    step() metadata:
────────────────                    ────────────────
env_name                            turn
num_episodes                        episode
env_params                          episode_turn
latent_id                           total_episodes
latent_mode                         episode_rewards (List[float])
reward_type                         turns_per_episode (List[int])
max_turns_per_episode               cumulative_reward
  (from env_params or               latent_id
   episode_config)                  reward_type
                                    max_turns_per_episode

max_turns_per_episode is the standard key used everywhere. It is read
from self._env_params.get("max_turns_per_episode", 0). Old env-specific
keys (max_attempts, num_turns, max_turns) still work as fallbacks in
individual core_envs.
```

**Heterogeneous support:**
- `_get_core_env(ep_idx)` — returns per-episode env from EpisodeDef
- `_get_prompt_template(ep_idx)` — per-episode override if set
- `_get_feedback_handler(ep_idx)` — per-episode override if set
- Homogeneous path always returns the shared instance (zero overhead).

### registry.py — Registration + make_env()

Per-env scoped registries: latent IDs only need to be unique within their env.

```python
register_env("bandits", BanditSingleEpisodeEnv, num_turns=20, buttons=5)
#            flat kwargs ──────────────────────────────────────────────
#            These go into env_params for latent generators and metadata.
#            They are NOT passed to core_env constructor (core_env takes no args).
#            No default_env_params={} nesting — all params are flat kwargs.
register_latent("bandits", LatentDefinition(...))
register_prompt("bandits", BanditFullInfoPrompt)
register_feedback("bandits", BanditStandardFeedback)
```

`make_env(FullyDefinedEnv, trajectory_path=None, seed=None, candidate_pool_path=None)`:
- `trajectory_path` → load from JSON (all latent types)
- `seed` → generator latent resolves on fly
- `seed + candidate_pool_path` → filter latent resolves on fly

### Construction Path Support Matrix

```
                        from_configs  SkyRL __init__  from_episode_defs  make_env
                        ────────────  ──────────────  ─────────────────  ────────
Homogeneous             ✅             ✅               ❌                 ✅
Heterogeneous           ❌             ❌               ✅                 ❌
Orchestrator            ✅             ─               ─                  ✅
SkyRL training          ─             ✅               ─                  ─
SkyRL eval (local)      ─             ✅               ─                  ─
```

### trajectory_utils.py — Shared Save/Load

Trajectory and Manifest dataclasses with JSON serialization. Used by per-env trajectory generators.
