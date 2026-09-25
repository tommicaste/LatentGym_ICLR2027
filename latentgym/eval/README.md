# Evaluation

This is the entry point for running benchmark evaluations: which type of run (single vs double agent), which model backend (OpenRouter for SOTA APIs, or local vLLM), how to generate reports, and how to inspect results interactively.

For the focused local-model walkthrough see [LOCAL_EVAL_GUIDE.md](../LOCAL_EVAL_GUIDE.md). For module internals (architecture, schemas, runner classes) see the [Developer Reference](#developer-reference) at the bottom of this file.

---

## 1. Eval Type

### Single-Agent

One model plays every episode in a trajectory. Used for measuring how well a model adapts in-context across episodes.

**CLI (one model, one env):**

```bash
python -m latentgym.cli.run_eval single \
    --models openrouter/openai:gpt-4o \
    --env bandits --latent loyal_favorite_0 \
    --prompt full_info --feedback standard \
    --num-episodes 10 --n-trajectories 50 \
    --trajectory-dir latentgym/data/eval/ \
    --output latentgym/results/gpt4o_bandits/
```

**Multiple models in one run** (recommended for comparison; produces one `metrics/` per output dir):

```bash
python -m latentgym.cli.run_eval single \
    --models openrouter/openai:gpt-4o openrouter/openai:o4-mini \
    --env bandits --latent loyal_favorite_0 \
    --n-trajectories 30 \
    --trajectory-dir latentgym/data/eval/ \
    --output latentgym/results/bandits_comparison/
```

**Filter by complexity tier** (`easy`, `medium`, `hard`, `very_hard`):

```bash
python -m latentgym.cli.run_eval single \
    --models openrouter/openai:gpt-4o \
    --env bandits --complexity easy \
    --n-trajectories 30 \
    --trajectory-dir latentgym/data/eval/ \
    --output latentgym/results/easy_bandits/
```

**Dry run** (preview what would be evaluated, no API calls):

```bash
python -m latentgym.cli.run_eval single \
    --models openrouter/openai:gpt-4o \
    --config latentgym/configs/eval_suites/full.yaml \
    --dry-run
```

**From a YAML suite config:**

Create `latentgym/configs/eval_suites/my_suite.yaml`:

```yaml
configs:
  - env: bandits
    latent: loyal_favorite_0
    prompt: full_info
    feedback: standard
    num_episodes: 10
  - env: wordle
    latent: vowel_count_2
    prompt: no_info
    feedback: standard
    num_episodes: 5
```

Then run:

```bash
python -m latentgym.cli.run_eval single \
    --models openrouter/openai:gpt-4o openrouter/anthropic:claude-sonnet-4-6 \
    --config latentgym/configs/eval_suites/my_suite.yaml \
    --trajectory-dir latentgym/data/eval/ \
    --output latentgym/results/full_run/
```

**Resuming interrupted runs:** Add `--resume`. The orchestrator checkpoints after each (model, env_config) pair into `<output>/checkpoint/`; rerunning with `--resume` skips completed pairs.

```bash
python -m latentgym.cli.run_eval single \
    --models openrouter/openai:gpt-4o \
    --config latentgym/configs/eval_suites/full.yaml \
    --trajectory-dir latentgym/data/eval/ \
    --output latentgym/results/full_run/ \
    --resume
```

**From Python:**

```python
import asyncio
from latentgym.eval.orchestrator import BenchmarkOrchestrator
from latentgym.eval.model_interface import OpenAIModel
from latentgym.core.env_config import FullyDefinedEnv
from latentgym.core.reward import RewardType
from latentgym.reporting import SingleAgentReport, DataStore
import latentgym.envs  # noqa: F401  (triggers env registration)
import os

# OpenRouter via OpenAI-compatible API
model = OpenAIModel(
    name="openrouter/openai:gpt-4o",
    model="openai/gpt-4o",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

configs = [
    FullyDefinedEnv("bandits", "loyal_favorite_0", "full_info", "standard",
                    num_episodes=10, reward_type=RewardType.CUMULATIVE),
]

orchestrator = BenchmarkOrchestrator(
    models=[model],
    env_configs=configs,
    trajectory_dir="latentgym/data/eval/",
    n_trajectories_per_config=50,
    seed=42,
    output_dir="latentgym/results/my_run/",
)
results = asyncio.run(orchestrator.run())

report = SingleAgentReport(results)
report.compute()
report.save_to(DataStore("latentgym/results/my_run/"))
```

**Metrics computed:**
- `avg_mean_reward` — mean reward per episode, averaged over trajectories
- `avg_initial_reward`, `avg_final_reward` — episode 0 and last episode rewards
- `avg_improvement`, `learning_slope` — adaptation across episodes
- `avg_success_rate`, `avg_mean_turns_per_episode`
- `per_episode_avg_rewards`, `per_episode_std_rewards`, `per_episode_avg_turns`

W&B integration: pass `wandb_config=WandbConfig(project="my-benchmark", entity="my-team", name="run-x")` to the orchestrator.

### Double-Agent

Agent A plays the first K episodes, agent B plays the rest. Tests whether agent B benefits from the context agent A built up.

**CLI:**

```bash
python -m latentgym.cli.run_eval double \
    --model-a openrouter/openai:gpt-4o \
    --model-b openrouter/openai:gpt-4o-mini \
    --switch-episode 5 \
    --env bandits --latent loyal_favorite_0 \
    --num-episodes 10 --n-trajectories 30 \
    --trajectory-dir latentgym/data/eval/ \
    --output latentgym/results/double_agent/
```

**From Python:**

```python
from latentgym.eval.double_agent import ScheduledRunner, two_agent_schedule

schedule = two_agent_schedule(agent_a, agent_b, switch_at=5)
runner = ScheduledRunner(schedule)
result = await runner.run_trajectory(env)
```

Agent switching happens at **episode boundaries only**, not mid-episode.

**Metrics computed:**
- `avg_pre_switch_reward`, `avg_post_switch_reward`
- `avg_transfer_effect`, `std_transfer_effect`
- `avg_adaptation_speed`
- Per-agent breakdown (avg_reward, success_rate)
- `per_episode_avg_rewards`, `per_episode_avg_turns`

**Note on local models for double-agent:** Currently requires two separate vLLM servers (or API models). A `SequentialModelLoader` that loads/unloads models on a single GPU is documented as future work in `runner.py`.

---

## 2. Model Type

### OpenRouter (SOTA APIs)

Primary path for evaluating frontier models. One API key (`OPENROUTER_API_KEY`) gives access to OpenAI, Anthropic, Google, and many other providers.

**Setup:**

```bash
export OPENROUTER_API_KEY="sk-or-..."
# Or add OPENROUTER_API_KEY=sk-or-... to your .env (then `source project_config.sh`)
```

**Model spec syntax:**

```
openrouter/openai:gpt-4o              → OpenAI GPT-4o via OpenRouter
openrouter/openai:o4-mini             → OpenAI o4-mini
openrouter/anthropic:claude-sonnet-4-6  → Anthropic Claude Sonnet 4.6
openrouter/google:gemini-2.5-pro      → Google Gemini 2.5 Pro
openrouter:any/model-id               → Pass-through to any OpenRouter model ID
```

**Direct provider access** (alternative, needs the provider's own key):

```
openai:gpt-4o                         → Direct OpenAI (needs OPENAI_API_KEY)
anthropic:claude-sonnet-4-6           → Direct Anthropic (needs ANTHROPIC_API_KEY)
google:gemini-pro                     → Direct Google (needs GOOGLE_API_KEY)
```

**Reasoning/thinking capture** (per provider):

| Provider | Reasoning available? | Opt-in flag |
|---|---|---|
| OpenAI direct | No (ChatCompletions discards) | — |
| OpenRouter (any model) | Yes (`message.reasoning`) | Automatic |
| Anthropic | Yes (thinking blocks) | `--enable-thinking --thinking-budget 10000` |
| Google Gemini | Yes (thought parts) | `--enable-thinking --thinking-budget 8192` |
| vLLM | Yes (`message.reasoning_content`) | Server flag `--reasoning-parser` |

Reasoning is written to `TrajectoryResult.reasoning_trace` for analysis but never fed back into the conversation context.

**Example with thinking enabled:**

```bash
python -m latentgym.cli.run_eval single \
    --models openrouter/anthropic:claude-sonnet-4-6 \
    --enable-thinking --thinking-budget 10000 \
    --env wordle --latent vowel_count_2 \
    --n-trajectories 10 \
    --trajectory-dir latentgym/data/eval/ \
    --output latentgym/results/claude_thinking/
```

### Local Models (vLLM)

For locally-served checkpoints. Three options depending on scale and pipeline:

| Path | When | Setup needed |
|---|---|---|
| **vLLM + APIRunner** | Quick eval, any single/double-agent run | Start vLLM server, then `--models vllm:http://localhost:8000` |
| **SkyRL pipeline + parquet** | Large-scale single-agent (~32× faster batched) | Generate parquet via `generate_data parquet`, then SkyRL eval |
| **SkyRL pipeline (no parquet)** | One-off SkyRL eval | Use generator/filter modes directly |

Quick example (vLLM + APIRunner):

```bash
# In one terminal:
vllm serve <your-model> --port 8000 --reasoning-parser deepseek_r1

# In another:
python -m latentgym.cli.run_eval single \
    --models vllm:http://localhost:8000 \
    --env bandits --latent loyal_favorite_0 \
    --n-trajectories 30 \
    --trajectory-dir latentgym/data/eval/ \
    --output latentgym/results/local_run/
```

For the full local-model walkthrough (including SkyRL pipeline tradeoffs, parquet generation, GPU memory tuning), see **[LOCAL_EVAL_GUIDE.md](../LOCAL_EVAL_GUIDE.md)**.

**Tradeoff summary** (full comparison in [Developer Reference / Two Eval Modes](#two-eval-modes-our-pipeline-vs-skyrl-pipeline)):

| | Our pipeline (Orchestrator + APIRunner) | SkyRL pipeline |
|---|---|---|
| API models | ✅ | ❌ |
| Local models | ✅ via `VLLMModel` | ✅ native, ~32× faster batched |
| Double-agent | ✅ | ❌ |
| Extra data step | none | needs parquet |

---

## 3. Reports

After an eval finishes, the orchestrator writes trajectories + per-model metrics into the `--output` directory. The `report` CLI turns those into tables, plots, dashboards, and trajectory explorers.

**All reports** (tables + plots + dashboard + trajectory explorer):

```bash
python -m latentgym.cli.report \
    --data-dir latentgym/results/my_run/ \
    --output latentgym/results/my_run/paper/
```

**Just tables** (markdown + LaTeX + CSV):

```bash
python -m latentgym.cli.report --data-dir latentgym/results/my_run/ --tables-only
```

**Just plots** (PNG / PDF / SVG):

```bash
python -m latentgym.cli.report --data-dir latentgym/results/my_run/ --plots-only --fmt pdf
```

**Leaderboard to stdout:**

```bash
python -m latentgym.cli.report --data-dir latentgym/results/my_run/ --leaderboard
python -m latentgym.cli.report --data-dir latentgym/results/my_run/ --leaderboard --env wordle
```

**Head-to-head comparison** (per-config deltas between two models):

```bash
python -m latentgym.cli.report \
    --data-dir latentgym/results/my_run/ \
    --compare openrouter/openai:gpt-4o openrouter/anthropic:claude-sonnet-4-6
```

**Recompute metrics** from saved trajectory JSONs (useful after running models separately to the same output dir):

```bash
python -m latentgym.cli.report --data-dir latentgym/results/my_run/ --recompute
```

**Output structure** (under `--output`):

```
paper/
├── tables/                     # main_results.{md,tex,csv}, leaderboard.{md,csv}, per_env.md, ...
├── plots/                      # PNG/PDF/SVG files
├── dashboard.html              # if --dashboard or default (all)
└── trajectory_explorer.html    # if --trajectories or default (all)
```

For module internals (how tables, plots, the dashboard, and the trajectory viewer are built) see [reporting/README.md](../reporting/README.md).

---

## 4. Dashboard & Trajectory Viewer

Two ways to explore results interactively: **static self-contained HTML** (no server) and a **Streamlit app** (richer, requires `streamlit`).

### Static HTML

```bash
# Interactive dashboard with all metrics, plots, comparisons
python -m latentgym.cli.report \
    --data-dir latentgym/results/my_run/ \
    --dashboard \
    --output latentgym/results/my_run/paper/

# Trajectory explorer — browse individual trajectories turn-by-turn
python -m latentgym.cli.report \
    --data-dir latentgym/results/my_run/ \
    --trajectories \
    --output latentgym/results/my_run/paper/

# Render a single trajectory file as HTML
python -m latentgym.cli.report \
    --trajectory latentgym/results/my_run/trajectories/openrouter__openai_gpt-4o/bandits__loyal_favorite_0/traj_0000.json \
    --html traj.html
```

Both `dashboard.html` and `trajectory_explorer.html` are self-contained — open them in any browser.

### Streamlit App

```bash
streamlit run latentgym/app/app.py -- --data-dir latentgym/results/my_run/
```

11 pages: Leaderboard, Per-Environment, Learning Curves, Trajectory Viewer, Comparisons, Dashboard, Latent Analysis, Prompt Ablation, Feedback Ablation, Double-Agent Results, Export.

For the full Streamlit guide (page-by-page features, persistence, alternatives) see [app/README.md](../app/README.md).

---

## Developer Reference

What follows is internal architecture: module layout, runner internals, schemas, and the comparison between our pipeline and the SkyRL pipeline. Skip if you only need to run evals.

### Module Structure and Connections

```
┌────────────────────────────────────────────────────────────────────────┐
│                              eval/                                     │
│                                                                        │
│  ┌──────────────────────┐                                              │
│  │  model_interface.py  │◄──── MockModel, OpenAIModel, AnthropicModel  │
│  │  ModelInterface ABC  │      GoogleModel, VLLMModel                  │
│  └──────────┬───────────┘                                              │
│             │                                                          │
│     ┌───────┴────────┐                                                 │
│     ▼                ▼                                                 │
│  ┌─────────────┐  ┌─────────────────┐                                  │
│  │ single_agent│  │  double_agent   │   Both produce TrajectoryResult  │
│  │             │  │                 │                                  │
│  │ api_runner  │  │ runner          │                                  │
│  │ local_runner│  │ agent_scheduler │                                  │
│  │ metrics     │  │ metrics         │                                  │
│  └─────┬───────┘  └───────┬─────────┘                                  │
│        │                  │                                            │
│        └────────┬─────────┘                                            │
│                 ▼                                                      │
│  ┌──────────────────────┐                                              │
│  │     types.py         │  EpisodeOutcome, TrajectoryResult            │
│  └──────────┬───────────┘  (shared by both single & double)            │
│             │                                                          │
│             ▼                                                          │
│  ┌──────────────────────┐  ┌──────────────────────┐                    │
│  │   results.py         │  │   orchestrator.py    │                    │
│  │   BenchmarkResults   │◄─│   loops model × env  │                    │
│  │   save/load/slice    │  │   checkpointing      │                    │
│  └──────────────────────┘  └──────────────────────┘                    │
│             │                         │                                │
│             ▼                         ▼                                │
│  ┌──────────────────────┐                                              │
│  │  wandb_tracker.py    │  Optional: logs metrics + trajectories       │
│  │  WandbTracker or     │  to W&B. DummyTracker when disabled.         │
│  │  DummyTracker        │                                              │
│  └──────────────────────┘                                              │
└────────────────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
    ┌──────────┐                  ┌───────────┐
    │  core/   │                  │  data/    │
    │ make_env │                  │ manifests │
    │ registry │                  │ traj JSONs│
    └──────────┘                  └───────────┘
```

### Eval Flow (Single-Agent)

```
Orchestrator
    │
    ├── for model in models:
    │     for config in env_configs:
    │       │
    │       ├── manifest = load_manifest(data_dir/env_latent/)
    │       │
    │       ├── for traj_file in manifest.trajectory_files:
    │       │     │
    │       │     ├── env = make_env(config, trajectory_path=traj_file)
    │       │     │                    │
    │       │     │              ┌─────┴─────────────────────┐
    │       │     │              │  core/ resolves:          │
    │       │     │              │  core_env + prompt +      │
    │       │     │              │  feedback + reward +      │
    │       │     │              │  episode_configs from JSON│
    │       │     │              └───────────────────────────┘
    │       │     │
    │       │     ├── result = await runner.run_trajectory(env)
    │       │     │                    │
    │       │     │              ┌─────┴─────────────────────┐
    │       │     │              │  env.init() → prompt      │
    │       │     │              │  loop:                    │
    │       │     │              │    model.generate(msgs)   │
    │       │     │              │    env.step(action)       │
    │       │     │              │  → TrajectoryResult       │
    │       │     │              └───────────────────────────┘
    │       │     │
    │       │     └── result.prompt_id = config.prompt_id  (enrich)
    │       │
    │       ├── results.add(model, benchmark_id, all_results)
    │       └── checkpoint()
    │
    └── return BenchmarkResults
```

### Architecture

```
eval/
├── model_interface.py       # ModelInterface ABC + 5 implementations
├── types.py                 # TrajectoryResult, EpisodeOutcome (shared by both)
├── results.py               # BenchmarkResults (storage + serialization)
├── orchestrator.py          # Runs all (model × env) combinations
├── wandb_tracker.py         # Optional W&B logging
├── single_agent/            # Independent single-agent eval
│   ├── api_runner.py        # APIRunner (for API + VLLMModel)
│   ├── local_runner.py      # LocalRunner (SkyRL bridge for batched local eval)
│   └── metrics.py           # 5 metrics functions
└── double_agent/            # Independent double-agent eval
    ├── runner.py             # ScheduledRunner + DoubleAgentRunner
    ├── agent_scheduler.py    # AgentConfig, AgentSchedule, schedule factories
    └── metrics.py            # Pre/post switch, transfer effect metrics
```

### Model Interface

```python
class ModelInterface(ABC):
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse

@dataclass
class ModelResponse:
    text: str                      # Action (fed back into conversation)
    reasoning: Optional[str]       # Internal thinking (recorded, NOT fed back)

# Implementations:
MockModel(default_response="[red]")                   # Testing
OpenAIModel(name="gpt-4o", model="gpt-4o")           # OpenAI API (+ OpenRouter via base_url)
AnthropicModel(name="claude", model="claude-sonnet-4-20250514",
               enable_thinking=True, thinking_budget=10000)  # Anthropic API
GoogleModel(name="gemini", model="gemini-2.5-pro",
            enable_thinking=True, thinking_budget=8192)      # Google API
VLLMModel(name="ft", base_url="http://localhost:8000/v1")    # Local vLLM server
```

API clients are cached (created once, reused across calls). No litellm dependency — uses direct SDKs (openai, anthropic, google-genai).

All API models include retry with exponential backoff, rate limit detection (429 → longer wait), per-request timeout, and logging.

### Reasoning/Thinking Support (details)

`generate()` returns `ModelResponse(text, reasoning)` — the runner only puts `text` into the conversation context. Reasoning is recorded separately in `TrajectoryResult.reasoning_trace` for analysis.

| Provider | Reasoning available? | How extracted | Opt-in needed? |
|----------|---------------------|---------------|----------------|
| OpenAI direct | No (ChatCompletions discards it) | `reasoning=None` | — |
| OpenRouter | Yes (`message.reasoning`) | Auto via `include_reasoning: True` | — (automatic) |
| Anthropic | Yes (thinking blocks) | Filter `content` by `type=="thinking"` vs `"text"` | `enable_thinking=True` |
| Google Gemini | Yes (thought parts) | `part.thought` boolean | `enable_thinking=True` |
| vLLM | Yes (`message.reasoning_content`) | `getattr` on message | Server `--reasoning-parser` flag |

### TrajectoryResult — What's Recorded

```python
TrajectoryResult:
    # Per-episode data
    episode_outcomes: List[EpisodeOutcome]
        # Each has: episode_idx, reward, turns, success, agent_name,
        #           outcome_type (WIN/LOSS/TIMEOUT/PARTIAL), latent_id, max_turns,
        #           ground_truth (episode_config used to reset env)

    # Full conversation (only action text, no reasoning)
    conversation: List[Dict[str, str]]  # All messages in order

    # Reasoning trace (per assistant turn, parallel to conversation)
    reasoning_trace: List[Optional[str]]  # One entry per assistant turn
        # None for turns without reasoning. Reasoning is NOT in conversation.

    # Ground truth per episode
    episode_configs: List[Dict[str, Any]]  # target_word, secret_code, draws, etc.

    # Environment config
    env_name, latent_id, prompt_id, feedback_id, reward_type, max_turns_per_episode

    # Run metadata
    model_name, seed, agent_assignments
    init_metadata, final_metadata

    # Derived (computed from episode_outcomes)
    episode_rewards, episode_turns, num_episodes,
    cumulative_reward, terminal_reward, improvement,
    mean_reward, success_rate, mean_turns, total_turns
```

### EpisodeOutcome Fields

```
Field            Type           Source
─────            ────           ──────
episode_idx      int            multi_episode_env step counter
reward           float          core_env.step() return value
turns            int            counted by runner
success          bool           reward > 0
agent_name       str            runner knows which model generated
outcome_type     OutcomeType    WIN/LOSS/TIMEOUT/PARTIAL
latent_id        str            from env init_metadata
max_turns        int            from env init_metadata
```

Serializable: `result.to_dict()` and `TrajectoryResult.from_dict(d)` for JSON persistence.

No `StepRecord` — the full conversation is in `conversation`, per-episode data in `episode_outcomes`. No duplication.

### Orchestrator

Flat loop over (model × env_config) pairs. Loads trajectory files from manifest directories.

**Why flat over deep hierarchy?** An earlier design had nested orchestration:
`run_benchmark → run_model → run_env → run_latent → run_trajectory`. This was harder to checkpoint (which level do you resume from?), harder to parallelize, and the nesting didn't add value since every leaf does the same thing: `make_env + run_trajectory`. The flat loop is simpler, checkpoints after each (model, env) pair, and is easy to extend (e.g., adding double-agent runs as a separate pass).

```python
orchestrator = BenchmarkOrchestrator(
    models=[model_a, model_b],
    env_configs=[fd_bandits, fd_wordle],
    data_dir="data/eval/",
    n_trajectories=50,
)
results = await orchestrator.run()
```

Features:
- Checkpointing after each (model, env) pair
- Skip completed (resume from checkpoint)
- Prompt_id/feedback_id enrichment on TrajectoryResult (orchestrator has FullyDefinedEnv context)
- Optional W&B logging via `wandb_config`

### BenchmarkResults

```python
results = BenchmarkResults()
results.add("gpt-4o", "bandits/loyal_favorite_0/full_info/standard/ep10", trajectories)
results.save("results.json")
results = BenchmarkResults.load("results.json")
sliced = results.slice_by_model("gpt-4o")
df = results.to_dataframe()
```

### W&B Tracker

Optional. Pass `wandb_config=WandbConfig(project="benchmark")` to orchestrator.

Logs:
- Per-episode: reward, turns, outcome type
- Per-trajectory: reward curves, cumulative metrics
- Summary tables: single-agent and double-agent metrics
- Comparison metrics between configs
- Trajectory artifacts for offline inspection

When wandb is off, a `DummyTracker` silently accepts all calls (no-op).

### When to Use Which Runner

```
Use case                        Runner            Model type
─────────                       ──────            ──────────
API models, single-agent        APIRunner         OpenAI/Anthropic/Google (direct or OpenRouter)
API models, double-agent        ScheduledRunner   OpenAI/Anthropic/Google (direct or OpenRouter)
Local models, quick eval        APIRunner         VLLMModel
Local models, large eval        LocalRunner       SkyRL pipeline (batched)
Local models, double-agent      ScheduledRunner   VLLMModel (2 servers)
Testing                         APIRunner         MockModel
```

### Two Eval Modes: Our Pipeline vs SkyRL Pipeline

There are two fundamentally different ways to run evaluation.

#### Mode 1: Our Pipeline (Orchestrator + APIRunner)

```
Data needed:   trajectory JSONs + manifest.json
How it works:  Orchestrator loads manifest → make_env(fd, trajectory_path) → APIRunner
Model types:   API models (OpenAI, Anthropic, Google) OR VLLMModel (local vLLM server)
Double-agent:  ✅ supported (ScheduledRunner)
Batching:      ❌ one trajectory at a time (sequential)
Latent modes:  trajectory_path only (orchestrator loads from manifest)

Orchestrator
    │
    for each (model, env_config):
        for each traj_file in manifest:
            env = make_env(config, trajectory_path=traj_file)
            result = await api_runner.run_trajectory(env)
```

#### Mode 2: SkyRL Pipeline (LocalRunner wraps SkyRLGymGenerator)

```
Data needed:   trajectory JSONs + manifest.json + parquet (extra step)
How it works:  SkyRL reads parquet → skyrl_gym.make() → batched vLLM generation
Model types:   Local models only (vLLM served, loaded via SkyRL)
Double-agent:  ❌ not supported (SkyRL doesn't know about agent switching)
Batching:      ✅ multiple trajectories per vLLM call (~32x faster)
Latent modes:  trajectory_path only (parquet points to JSON files)

SkyRL Pipeline
    │
    PromptDataset reads parquet
        │
        for each batch of N prompts:
            envs = [skyrl_gym.make(row.env_class, extras=row.extra_info) for row in batch]
            responses = vllm.generate([N prompts at once])  ← batched!
            for env, response in zip(envs, responses):
                env.step(response)
```

#### Comparison

```
                                Our Pipeline              SkyRL Pipeline
                                ────────────              ──────────────
Data input                      manifest.json             parquet (→ JSONs)
Extra data step needed          no                        yes (generate parquet)
API models                      ✅                         ❌
Local models (vLLM)             ✅ (via VLLMModel)         ✅ (native, batched)
Batched inference               ❌                         ✅ (~32x faster)
Double-agent                    ✅                         ❌
Single-agent                    ✅                         ✅
Orchestrator integration        ✅                         ❌ (standalone)
Checkpointing                   ✅                         SkyRL handles it
W&B logging                     ✅ (our tracker)           SkyRL has its own
Metrics                         our metrics module         SkyRL's metrics
JSON route = parquet route      ✅ verified identical       ✅ verified identical
```

#### When to use which

```
"I want to eval GPT-4o / Claude / Gemini"
    → Our pipeline (Mode 1). API models can't use SkyRL.
    → Need: trajectory JSONs + manifest

"I want quick eval of a local model"
    → Our pipeline (Mode 1) with VLLMModel. Simple setup.
    → Need: vLLM server running + trajectory JSONs + manifest

"I want fast eval of a local model on many trajectories"
    → SkyRL pipeline (Mode 2). Batched inference is ~32x faster.
    → Need: vLLM server running + trajectory JSONs + manifest + parquet

"I want to compare two agents (double-agent)"
    → Our pipeline (Mode 1) only. SkyRL doesn't support agent switching.
    → Need: two models + trajectory JSONs + manifest

"I want to train with RL"
    → SkyRL pipeline only. Our pipeline doesn't do training.
    → Need: trajectory JSONs + parquet
```

### What CAN'T be extended

```
SkyRL pipeline + double-agent:
    ❌ Cannot be implemented. SkyRL's SkyRLGymGenerator runs one model per batch.
    It has no concept of switching models mid-trajectory. Agent switching requires
    our ScheduledRunner which controls the model selection per episode.

SkyRL pipeline + API models:
    ❌ Cannot be implemented. SkyRL's inference engine expects vLLM/SGLang.
    API models use HTTP calls with different SDKs (openai, anthropic, google).
    These are fundamentally different inference backends.

Orchestrator + generator/filter modes:
    Could be implemented (make_env supports it) but currently only uses
    trajectory_path. Would need to pass seed/pool_path to orchestrator config.
    Not a priority since data generation step handles this.
```
