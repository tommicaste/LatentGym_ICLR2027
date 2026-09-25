"""
Run benchmark evaluations (single-agent or double-agent).

Usage:
    # Single-agent eval with OpenAI GPT-4o
    python -m latentgym.cli.run_eval single \\
        --model openai:gpt-4o \\
        --env bandits --latent loyal_favorite_0 \\
        --prompt full_info --feedback standard \\
        --num-episodes 10 --n-trajectories 50 \\
        --trajectory-dir data/eval/ \\
        --output results/gpt4o/

    # Single-agent eval — multiple models × envs from a config file
    python -m latentgym.cli.run_eval single \\
        --config configs/eval_suites/quick.yaml \\
        --output results/run_001/

    # Double-agent eval (model A does first K episodes, model B does rest)
    python -m latentgym.cli.run_eval double \\
        --model-a openai:gpt-4o --model-b openai:gpt-4o-mini \\
        --switch-episode 5 \\
        --env bandits --latent loyal_favorite_0 \\
        --trajectory-dir data/eval/ \\
        --output results/double/

    # Resume interrupted run
    python -m latentgym.cli.run_eval single \\
        --config configs/eval_suites/full.yaml \\
        --output results/run_001/ --resume
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_model_spec(spec: str):
    """Parse 'provider:model_name' into ModelInterface.

    Direct API access:
        openai:gpt-4o                    → OpenAI API (needs OPENAI_API_KEY)
        anthropic:claude-sonnet-4-6      → Anthropic API (needs ANTHROPIC_API_KEY)
        google:gemini-pro                → Google GenAI API (needs GOOGLE_API_KEY)

    Via OpenRouter (all models through one API, one key):
        openrouter/openai:gpt-4o         → OpenRouter (needs OPENROUTER_API_KEY)
        openrouter/anthropic:claude-3.5-sonnet → OpenRouter
        openrouter/google:gemini-pro     → OpenRouter
        openrouter:any/model-name        → OpenRouter (pass-through model ID)

    Local / testing:
        vllm:http://localhost:8000       → Local vLLM server
        mock:random                      → Deterministic mock (no API needed)
    """
    from latentgym.eval.model_interface import (
        OpenAIModel, AnthropicModel, GoogleModel, VLLMModel, MockModel,
    )
    import os

    if spec.startswith("mock:"):
        return MockModel(name=spec)

    if spec.startswith("vllm:"):
        url = spec[len("vllm:"):]
        return VLLMModel(name=spec, base_url=url)

    # OpenRouter prefix: route everything through OpenRouter's OpenAI-compatible API
    if spec.startswith("openrouter/") or spec.startswith("openrouter:"):
        sep = "/" if spec.startswith("openrouter/") else ":"
        model_name = spec[len("openrouter" + sep):]
        # Map friendly prefixes to OpenRouter model IDs
        # openrouter/openai:gpt-4o → openai/gpt-4o
        # openrouter/anthropic:claude-3.5-sonnet → anthropic/claude-3.5-sonnet
        # openrouter/google:gemini-pro → google/gemini-pro
        model_name = model_name.replace(":", "/", 1) if ":" in model_name else model_name
        return OpenAIModel(
            name=spec,
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

    # Direct API access
    if spec.startswith("openai:"):
        model_name = spec[len("openai:"):]
        return OpenAIModel(name=spec, model=model_name)

    if spec.startswith("anthropic:"):
        model_name = spec[len("anthropic:"):]
        return AnthropicModel(name=spec, model=model_name)

    if spec.startswith("google:"):
        model_name = spec[len("google:"):]
        return GoogleModel(name=spec, model=model_name)

    raise ValueError(
        f"Unknown model spec: '{spec}'. Expected format: provider:model_name\n"
        f"  Direct:     openai:gpt-4o, anthropic:claude-sonnet-4-6, google:gemini-pro\n"
        f"  OpenRouter: openrouter/openai:gpt-4o, openrouter/google:gemini-pro\n"
        f"  Local:      vllm:http://localhost:8000\n"
        f"  Testing:    mock:random"
    )


def _build_env_configs(args) -> List:
    """Build FullyDefinedEnv list from args or config file.

    Supports hierarchical expansion:
        --env bandits                      → all latents × all prompts × all feedbacks for bandits
        --env bandits --latent loyal_favorite_0  → all prompts × all feedbacks for this latent
        --env bandits --prompt full_info   → all latents, but only full_info prompt
        (no --env)                         → all envs × all latents × all prompts × all feedbacks
        --complexity easy                  → only easy latents (combinable with above)

    Explicit values always override expansion:
        --env bandits --latent loyal_favorite_0 --prompt full_info --feedback standard
        → exactly one config
    """
    from latentgym.core.env_config import FullyDefinedEnv
    from latentgym.core.reward import RewardType
    from latentgym.core import registry
    import latentgym.envs  # noqa: F401 — triggers registration

    if hasattr(args, "config") and args.config:
        return _load_suite_config(args.config)

    num_episodes = args.num_episodes

    # ── Dependency validation ──
    # Latent, prompt, and feedback are env-specific — require --env if any are specified
    has_sub_filter = args.latent or getattr(args, "prompt", None) or getattr(args, "feedback", None)
    if has_sub_filter and not args.env:
        raise ValueError(
            "--env is required when specifying --latent, --prompt, or --feedback "
            "(these are env-specific and differ across environments)"
        )

    # Determine which envs to run
    all_envs = registry.list_envs()
    if args.env:
        env_names = [e.strip() for e in args.env.split(",")]
        for e in env_names:
            if e not in all_envs:
                raise ValueError(f"Unknown env '{e}'. Available: {sorted(all_envs.keys())}")
    else:
        env_names = sorted(all_envs.keys())

    complexity_filter = getattr(args, "complexity", None)

    configs = []
    for env_name in env_names:
        env_latent_ids = {l.id for l in registry.list_latents(env_name)}
        env_prompt_ids = set(registry.list_prompts(env_name))
        env_feedback_ids = set(registry.list_feedbacks(env_name))

        # ── Latents ──
        if args.latent:
            latent_ids = [l.strip() for l in args.latent.split(",")]
            # Validate all specified latents exist in this env
            for lid in latent_ids:
                if lid not in env_latent_ids:
                    raise ValueError(
                        f"Latent '{lid}' not found in env '{env_name}'. "
                        f"Available: {sorted(env_latent_ids)}"
                    )
        else:
            latents = registry.list_latents(env_name)
            if complexity_filter:
                latents = [l for l in latents if l.complexity.value == complexity_filter]
            latent_ids = [l.id for l in latents]

        if not latent_ids:
            logger.warning(f"No latents for env '{env_name}' with complexity={complexity_filter} — skipping")
            continue

        # ── Prompts ──
        prompt_arg = getattr(args, "prompt", None)
        if prompt_arg:
            prompt_ids = [p.strip() for p in prompt_arg.split(",")]
            for pid in prompt_ids:
                if pid not in env_prompt_ids:
                    raise ValueError(
                        f"Prompt '{pid}' not found in env '{env_name}'. "
                        f"Available: {sorted(env_prompt_ids)}"
                    )
        else:
            prompt_ids = sorted(env_prompt_ids)

        # ── Feedbacks ──
        feedback_arg = getattr(args, "feedback", None)
        if feedback_arg:
            feedback_ids = [f.strip() for f in feedback_arg.split(",")]
            for fid in feedback_ids:
                if fid not in env_feedback_ids:
                    raise ValueError(
                        f"Feedback '{fid}' not found in env '{env_name}'. "
                        f"Available: {sorted(env_feedback_ids)}"
                    )
        else:
            feedback_ids = sorted(env_feedback_ids)

        # ── Build cross-product for this env ──
        for latent_id in latent_ids:
            for prompt_id in prompt_ids:
                for feedback_id in feedback_ids:
                    configs.append(FullyDefinedEnv(
                        env_name=env_name,
                        latent_id=latent_id,
                        prompt_id=prompt_id,
                        feedback_id=feedback_id,
                        num_episodes=num_episodes,
                        reward_type=RewardType.CUMULATIVE,
                    ))

    if not configs:
        raise ValueError(
            "No valid configs generated. Check --env, --latent, --prompt, --feedback, --complexity."
        )

    return configs


def _load_suite_config(config_path: str) -> List:
    """Load eval suite from YAML config file."""
    import yaml
    from latentgym.core.env_config import FullyDefinedEnv
    from latentgym.core.reward import RewardType
    import latentgym.envs  # noqa: F401

    with open(config_path) as f:
        suite = yaml.safe_load(f)

    configs = []
    for entry in suite.get("configs", []):
        configs.append(FullyDefinedEnv(
            env_name=entry["env"],
            latent_id=entry["latent"],
            prompt_id=entry.get("prompt", "full_info"),
            feedback_id=entry.get("feedback", "standard"),
            num_episodes=entry.get("num_episodes", 10),
            reward_type=RewardType.CUMULATIVE,
        ))
    return configs


def _print_eval_plan(models: List, configs: List):
    """Print what will be evaluated (for --dry-run or logging)."""
    model_names = [m if isinstance(m, str) else m.name for m in models]

    # Group configs by env
    by_env = {}
    for c in configs:
        by_env.setdefault(c.env_name, []).append(c)

    total = len(model_names) * len(configs)

    print(f"\nEval plan: {len(model_names)} model(s) × {len(configs)} config(s) = {total} runs")
    print(f"Models: {', '.join(model_names)}")
    print()
    for env_name, env_configs in sorted(by_env.items()):
        latents = sorted({c.latent_id for c in env_configs})
        prompts = sorted({c.prompt_id for c in env_configs})
        feedbacks = sorted({c.feedback_id for c in env_configs})
        print(f"  {env_name}: {len(latents)} latent(s) × {len(prompts)} prompt(s) × {len(feedbacks)} feedback(s) = {len(env_configs)} configs")
        if len(latents) <= 10:
            print(f"    latents:   {', '.join(latents)}")
        else:
            print(f"    latents:   {', '.join(latents[:5])} ... +{len(latents)-5} more")
        print(f"    prompts:   {', '.join(prompts)}")
        print(f"    feedbacks: {', '.join(feedbacks)}")
    print()


async def _run_single(args):
    """Run single-agent evaluation."""
    from latentgym.eval.orchestrator import BenchmarkOrchestrator
    from latentgym.reporting import SingleAgentReport
    from latentgym.reporting.data_store import DataStore

    models = [_parse_model_spec(m) for m in args.models]
    # Apply model params from CLI to all models
    for model in models:
        if hasattr(model, 'temperature'):
            model.temperature = args.temperature
        if hasattr(model, 'max_tokens'):
            model.max_tokens = args.max_tokens
        if hasattr(model, 'max_retries'):
            model.max_retries = args.max_retries
        if hasattr(model, 'request_timeout'):
            model.request_timeout = args.request_timeout
        if hasattr(model, 'enable_thinking'):
            model.enable_thinking = args.enable_thinking
        if hasattr(model, 'thinking_budget'):
            model.thinking_budget = args.thinking_budget
    env_configs = _build_env_configs(args)
    output_dir = Path(args.output)

    _print_eval_plan(models, env_configs)

    if args.dry_run:
        logger.info("Dry run — exiting without running eval")
        return

    logger.info(f"Single-agent eval: {len(models)} model(s) × {len(env_configs)} config(s)")

    checkpoint_dir = str(output_dir / "checkpoint") if args.resume else None

    orchestrator = BenchmarkOrchestrator(
        models=models,
        env_configs=env_configs,
        trajectory_dir=args.trajectory_dir,
        n_trajectories_per_config=args.n_trajectories,
        seed=args.seed,
        checkpoint_dir=checkpoint_dir,
        max_total_turns=args.max_total_turns,
        output_dir=str(output_dir),
        start_trajectory=args.start_trajectory,
    )

    results = await orchestrator.run()

    # Compute and save metrics (trajectories already saved incrementally by orchestrator).
    # Conversations were cleared from memory after saving — metrics use episode_outcomes only.
    report = SingleAgentReport(results)
    report.compute()
    store = DataStore(str(output_dir))
    report.save_to(store)  # Writes metrics + tables only, not trajectories
    store.write_metadata({
        "run_name": args.run_name or "",
        "report_type": "single_agent",
        "models": results.model_names,
        "benchmark_ids": results.benchmark_ids,
    })
    logger.info(f"Results saved to {output_dir}")
    logger.info("Run 'python -m latentgym.cli.report --data-dir ... --recompute' for full reports")

    # Print leaderboard
    from latentgym.reporting.leaderboard import format_leaderboard
    print("\n" + format_leaderboard(report.leaderboard))


async def _run_double(args):
    """Run double-agent evaluation."""
    from latentgym.eval.orchestrator import BenchmarkOrchestrator
    from latentgym.reporting import DataStore, SingleAgentReport, DoubleAgentReport

    model_a = _parse_model_spec(args.model_a)
    model_b = _parse_model_spec(args.model_b)
    for model in [model_a, model_b]:
        if hasattr(model, 'temperature'):
            model.temperature = args.temperature
        if hasattr(model, 'max_tokens'):
            model.max_tokens = args.max_tokens
        if hasattr(model, 'max_retries'):
            model.max_retries = args.max_retries
        if hasattr(model, 'request_timeout'):
            model.request_timeout = args.request_timeout
        if hasattr(model, 'enable_thinking'):
            model.enable_thinking = args.enable_thinking
        if hasattr(model, 'thinking_budget'):
            model.thinking_budget = args.thinking_budget
    env_configs = _build_env_configs(args)
    output_dir = Path(args.output)

    _print_eval_plan([model_a, model_b], env_configs)

    if args.dry_run:
        logger.info("Dry run — exiting without running eval")
        return

    logger.info(f"Double-agent eval: {model_a.name} → {model_b.name} (switch at ep {args.switch_episode})")

    orchestrator = BenchmarkOrchestrator(
        models=[model_a, model_b],
        env_configs=env_configs,
        trajectory_dir=args.trajectory_dir,
        n_trajectories_per_config=args.n_trajectories,
        seed=args.seed,
        max_total_turns=args.max_total_turns,
        output_dir=str(output_dir),
        start_trajectory=args.start_trajectory,
    )

    results = await orchestrator.run_double_agent(model_a, model_b, args.switch_episode)

    # Save metrics only (trajectories already saved incrementally by orchestrator).
    # Conversations were cleared from memory — metrics use episode_outcomes only.
    store = DataStore(str(output_dir))
    SingleAgentReport(results).save_to(store)
    DoubleAgentReport(results, switch_episode=args.switch_episode).save_to(store)
    store.write_metadata({
        "run_name": args.run_name,
        "report_types": ["single_agent", "double_agent"],
        "models": results.model_names,
        "benchmark_ids": results.benchmark_ids,
        "switch_episode": args.switch_episode,
    })
    logger.info(f"Results saved to {output_dir}")
    logger.info("Run 'python -m latentgym.cli.report --data-dir ... --recompute' for full reports")


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Shared args
    def add_common(p):
        p.add_argument("--env", default=None,
                        help="Environment(s). Omit for all. Comma-separated for multiple: bandits,wordle")
        p.add_argument("--latent", default=None,
                        help="Latent(s). Omit for all in env. Comma-separated: loyal_favorite_0,clockwise_rotation")
        p.add_argument("--prompt", default=None,
                        help="Prompt(s). Omit for all in env. Comma-separated: no_info,full_info")
        p.add_argument("--feedback", default=None,
                        help="Feedback(s). Omit for all in env. Comma-separated: standard,with_stats")
        p.add_argument("--complexity", default=None,
                        choices=["easy", "medium", "hard", "very_hard"],
                        help="Filter latents by complexity level")
        p.add_argument("--num-episodes", type=int, default=10)
        p.add_argument("--n-trajectories", type=int, default=50)
        p.add_argument("--start-trajectory", type=int, default=0,
                        help="Skip trajectories before this index (e.g., --start-trajectory 1 skips traj 0)")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--trajectory-dir", default="benchmark/data/eval/")
        p.add_argument("--output", default="benchmark/results/",
                        help="Output directory for results (default: benchmark/results/)")
        p.add_argument("--run-name", default="")
        p.add_argument("--max-total-turns", type=int, default=1000)
        p.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for model generation")
        p.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens per model response")
        p.add_argument("--max-retries", type=int, default=3,
                        help="Max retries on API errors (with exponential backoff)")
        p.add_argument("--request-timeout", type=float, default=120,
                        help="Timeout in seconds per API request (default: 120)")
        p.add_argument("--enable-thinking", action="store_true",
                        help="Enable extended thinking/reasoning for Anthropic and Google models")
        p.add_argument("--thinking-budget", type=int, default=10000,
                        help="Max tokens for thinking/reasoning (default: 10000)")
        p.add_argument("--config", default=None, help="YAML eval suite config")
        p.add_argument("--dry-run", action="store_true",
                        help="Print eval plan without running")

    # single
    p_single = sub.add_parser("single", help="Single-agent evaluation")
    add_common(p_single)
    p_single.add_argument("--models", nargs="+", required=True,
                           help="Model specs, e.g. openai:gpt-4o anthropic:claude-sonnet-4-6")
    p_single.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    # double
    p_double = sub.add_parser("double", help="Double-agent evaluation")
    add_common(p_double)
    p_double.add_argument("--model-a", required=True)
    p_double.add_argument("--model-b", required=True)
    p_double.add_argument("--switch-episode", type=int, default=5)

    args = parser.parse_args()

    if args.cmd == "single":
        asyncio.run(_run_single(args))
    elif args.cmd == "double":
        asyncio.run(_run_double(args))


if __name__ == "__main__":
    main()
