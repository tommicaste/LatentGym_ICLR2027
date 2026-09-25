"""
Generate benchmark datasets: trajectory JSONs and SkyRL-compatible parquets.

Three-step pipeline:
    1. 'eval'    → trajectory JSONs (ground truth only, no prompt/feedback)
    2. 'parquet' → convert JSONs → SkyRL parquet (adds prompt/feedback/reward combos)
    3. 'train'   → convenience: JSONs + parquet in one command

Trajectory JSONs are consumed directly by:
    - BenchmarkOrchestrator / APIRunner (benchmark eval)

Parquets are consumed by:
    - SkyRL training pipeline (PromptDataset reads parquets)
    - SkyRL eval pipeline (LocalRunner / SkyRLGymGenerator)

Usage:
    # ── Step 1: Generate trajectory JSONs ──

    # All envs, all latents
    python -m latentgym.cli.generate_data eval --output data/eval/

    # One env, with train/val split
    python -m latentgym.cli.generate_data eval --env bandits --val-ratio 0.2 --output data/eval/

    # Easy latents only
    python -m latentgym.cli.generate_data eval --env bandits --complexity easy --output data/eval/

    # Specific latents
    python -m latentgym.cli.generate_data eval --env bandits --latent loyal_favorite_0,clockwise_rotation --output data/eval/

    # ── Step 2: Convert JSONs → parquet ──

    # Training parquet (all prompt × feedback × reward combos)
    python -m latentgym.cli.generate_data parquet \\
        --source data/eval/bandits/loyal_favorite_0/train/ \\
        --mode train --output-path data/parquets/train/bandits_loyal.parquet

    # Eval parquet (single prompt/feedback for controlled eval)
    python -m latentgym.cli.generate_data parquet \\
        --source data/eval/bandits/loyal_favorite_0/val/ \\
        --mode eval --prompt full_info --feedback standard \\
        --output-path data/parquets/eval/bandits_loyal.parquet

    # ── Shortcut: JSONs + parquet in one command ──

    python -m latentgym.cli.generate_data train \\
        --env bandits --latent loyal_favorite_0 \\
        --n-trajectories 500 --seed 10000 \\
        --output data/train/

    # ── List what's available ──

    python -m latentgym.cli.generate_data list
    python -m latentgym.cli.generate_data list --env bandits
    python -m latentgym.cli.generate_data list --env bandits --complexity easy
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Map env names to their trajectory generator modules
_GENERATOR_MODULES = {
    "bandits": "latentgym.envs.bandits.trajectory_generator",
    "wordle": "latentgym.envs.wordle.trajectory_generator",
    "hangman": "latentgym.envs.hangman.trajectory_generator",
    "mastermind": "latentgym.envs.mastermind.trajectory_generator",
    "secretary": "latentgym.envs.secretary.trajectory_generator",
    "wordladder": "latentgym.envs.wordladder.trajectory_generator",
    "number_guessing": "latentgym.envs.number_guessing.trajectory_generator",
}

# Map env names to their generator function names
_GENERATOR_FUNCTIONS = {
    "bandits": "generate_bandit_trajectories",
    "wordle": "generate_wordle_trajectories",
    "hangman": "generate_hangman_trajectories",
    "mastermind": "generate_mastermind_trajectories",
    "secretary": "generate_secretary_trajectories",
    "wordladder": "generate_wordladder_trajectories",
    "number_guessing": "generate_number_guessing_trajectories",
}


def _resolve_envs_and_latents(args) -> List[tuple]:
    """Resolve env + latent combinations from CLI args.

    Returns list of (env_name, latent_id) tuples.
    Same dependency rules as run_eval.py:
        --latent requires --env
        --complexity works with or without --env
    """
    from latentgym.core import registry
    import latentgym.envs  # noqa: F401

    # Dependency check
    if args.latent and not args.env:
        raise ValueError("--env is required when specifying --latent (latents are env-specific)")

    all_envs = registry.list_envs()

    # Determine envs
    if args.env:
        env_names = [e.strip() for e in args.env.split(",")]
        for e in env_names:
            if e not in all_envs:
                raise ValueError(f"Unknown env '{e}'. Available: {sorted(all_envs.keys())}")
    else:
        env_names = sorted(all_envs.keys())

    complexity_filter = getattr(args, "complexity", None)

    pairs = []
    for env_name in env_names:
        env_latent_ids = {l.id for l in registry.list_latents(env_name)}

        if args.latent:
            latent_ids = [l.strip() for l in args.latent.split(",")]
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
            logger.warning(f"No latents for '{env_name}' with complexity={complexity_filter} — skipping")
            continue

        for lid in latent_ids:
            pairs.append((env_name, lid))

    return pairs


def cmd_list(args):
    """List registered environments, latents, prompts, and feedbacks."""
    from latentgym.core import registry
    import latentgym.envs  # noqa: F401

    all_envs = registry.list_envs()
    complexity_filter = getattr(args, "complexity", None)

    if not args.env:
        # List all envs with summary counts
        print("\nRegistered Environments:")
        print(f"{'Environment':<15} {'Latents':>8} {'Prompts':>8} {'Feedbacks':>10}")
        print("-" * 45)
        for name, info in sorted(all_envs.items()):
            n_l = info.get("num_latents", 0)
            n_p = info.get("num_prompts", 0)
            n_f = info.get("num_feedbacks", 0)
            print(f"  {name:<13} {n_l:>8} {n_p:>8} {n_f:>10}")
        print()
        return

    # Detailed listing for specific env(s)
    env_names = [e.strip() for e in args.env.split(",")]
    for env_name in env_names:
        if env_name not in all_envs:
            print(f"Unknown env: {env_name}")
            continue

        latents = registry.list_latents(env_name)
        if complexity_filter:
            latents = [l for l in latents if l.complexity.value == complexity_filter]

        prompts = registry.list_prompts(env_name)
        feedbacks = registry.list_feedbacks(env_name)

        print(f"\n{'=' * 60}")
        print(f"Environment: {env_name}")
        print(f"{'=' * 60}")

        print(f"\nLatents ({len(latents)}):")
        if complexity_filter:
            print(f"  (filtered: complexity={complexity_filter})")
        by_complexity = {}
        for l in latents:
            by_complexity.setdefault(l.complexity.value, []).append(l)
        for c in ["easy", "medium", "hard", "very_hard"]:
            group = by_complexity.get(c, [])
            if group:
                print(f"\n  {c.upper()} ({len(group)}):")
                for l in sorted(group, key=lambda x: x.id):
                    desc = l.description[:55] if l.description else ""
                    print(f"    {l.id:<35} {desc}")

        print(f"\nPrompts ({len(prompts)}): {', '.join(sorted(prompts))}")
        print(f"Feedbacks ({len(feedbacks)}): {', '.join(sorted(feedbacks))}")
    print()


def cmd_generate_eval(args):
    """Generate trajectory JSONs + parquets for eval (and optionally train).

    Always generates trajectory JSONs. Also generates parquets alongside them
    using prompt/feedback/reward configs.

    With --val-ratio 0.2:
        <output>/<env>/<latent>/train/   → JSONs + train parquet (all prompt×feedback×reward combos)
        <output>/<env>/<latent>/val/     → JSONs + eval parquet (controlled config)

    With --split train or --split val:
        <output>/<env>/<latent>/<split>/ → JSONs + parquet for that split

    Without split:
        <output>/<env>/<latent>/         → JSONs + eval parquet
    """
    import latentgym.envs  # noqa: F401
    from latentgym.core import registry

    pairs = _resolve_envs_and_latents(args)
    output_dir = Path(args.output)

    # Parse --env-param KEY=VALUE overrides
    user_env_params = {}
    if getattr(args, "env_param", None):
        for kv in args.env_param:
            if "=" not in kv:
                raise ValueError(f"--env-param must be KEY=VALUE, got: {kv}")
            k, v = kv.split("=", 1)
            # Auto-convert types
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            user_env_params[k.strip()] = v

    val_ratio = getattr(args, "val_ratio", None)
    split_name = getattr(args, "split", None)

    # Compute per-split trajectory counts, seeds, and parquet mode
    # Train splits: all prompt × feedback × reward combos
    # Val/eval splits: all prompt × feedback combos, single reward (per_episode)
    if val_ratio is not None and val_ratio > 0:
        n_total = args.n_trajectories
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val
        splits = [
            ("train", n_train, args.seed, "train"),
            ("val", n_val, args.seed + n_train, "eval"),
        ]
    elif split_name:
        parquet_mode = "train" if split_name == "train" else "eval"
        splits = [(split_name, args.n_trajectories, args.seed, parquet_mode)]
    else:
        splits = [(None, args.n_trajectories, args.seed, "eval")]

    # Resolve prompt/feedback/reward for parquet generation
    # These are eval-time configs — they go into the parquet, not the trajectory JSONs
    # Defaults: all prompts, all feedbacks, all reward types for the env
    prompt_arg = getattr(args, "prompt", None)
    feedback_arg = getattr(args, "feedback", None)
    reward_arg = getattr(args, "reward_type", None)

    from latentgym.core.reward import RewardType
    all_reward_types = [r.value for r in RewardType]
    reward_types = [r.strip() for r in reward_arg.split(",")] if reward_arg else all_reward_types

    # Group by env for display
    by_env = {}
    for env_name, lid in pairs:
        by_env.setdefault(env_name, []).append(lid)

    total_files = len(pairs) * sum(n for _, n, _, _ in splits)
    print(f"\nGeneration plan: {len(pairs)} (env, latent) combinations")
    if user_env_params:
        print(f"  Env param overrides: {user_env_params}")
    for env_name, lids in sorted(by_env.items()):
        if len(lids) <= 8:
            print(f"  {env_name}: {', '.join(lids)}")
        else:
            print(f"  {env_name}: {', '.join(lids[:5])} ... +{len(lids)-5} more ({len(lids)} total)")
    for split_name_i, n_traj_i, seed_i, pq_mode_i in splits:
        label = f"  {split_name_i or 'data'}: "
        if pq_mode_i == "train":
            pq_desc = f"parquets: P×F×R combos (one per combo)"
        else:
            pq_desc = f"parquets: P×F combos (reward=per_episode)"
        print(f"{label}{n_traj_i} trajectories × {args.num_episodes} episodes (seed={seed_i}) → JSONs + {pq_desc}")
    print(f"  Total trajectory files: {total_files}")
    print()

    if args.dry_run:
        logger.info("Dry run — exiting without generating")
        return

    generated_json = 0
    generated_parquet = 0
    failed = 0

    for env_name, latent_id in pairs:
        if env_name not in _GENERATOR_MODULES:
            logger.warning(f"No trajectory generator for env '{env_name}' — skipping")
            failed += 1
            continue

        gen_module = importlib.import_module(_GENERATOR_MODULES[env_name])
        gen_fn_name = _GENERATOR_FUNCTIONS.get(env_name)
        gen_fn = getattr(gen_module, gen_fn_name, None)

        if gen_fn is None:
            logger.warning(f"Generator function '{gen_fn_name}' not found in {_GENERATOR_MODULES[env_name]}")
            failed += 1
            continue

        # Resolve prompt/feedback for this env
        env_prompts = sorted(registry.list_prompts(env_name))
        env_feedbacks = sorted(registry.list_feedbacks(env_name))

        if prompt_arg:
            prompt_ids = [p.strip() for p in prompt_arg.split(",")]
        else:
            prompt_ids = env_prompts

        if feedback_arg:
            feedback_ids = [f.strip() for f in feedback_arg.split(",")]
        else:
            feedback_ids = env_feedbacks

        for split_name_i, n_traj_i, seed_i, pq_mode_i in splits:
            if split_name_i:
                dest = str(output_dir / env_name / latent_id / split_name_i)
            else:
                dest = str(output_dir / env_name / latent_id)

            logger.info(f"  {env_name}/{latent_id}{('/' + split_name_i) if split_name_i else ''}")

            # Step 1: Generate trajectory JSONs
            try:
                gen_kwargs = dict(
                    latent_id=latent_id,
                    num_episodes=args.num_episodes,
                    n_trajectories=n_traj_i,
                    seed=seed_i,
                    output_dir=dest,
                )
                if user_env_params:
                    gen_kwargs["env_params"] = user_env_params
                # Resolve pool paths per env.
                # Only pass pool params if the generator function accepts them
                # (wordle, hangman, wordladder need pools; bandits, etc. don't).
                import inspect
                gen_params = inspect.signature(gen_fn).parameters
                if "filtered_pool_dir" in gen_params:
                    pool_dir = getattr(args, "filtered_pool_dir", None) or ""
                    if pool_dir:
                        env_pool = Path(pool_dir) / env_name
                        if env_pool.exists():
                            gen_kwargs["filtered_pool_dir"] = str(env_pool)
                        else:
                            gen_kwargs["filtered_pool_dir"] = pool_dir
                if "candidate_pool_path" in gen_params:
                    if getattr(args, "candidate_pool", None):
                        gen_kwargs["candidate_pool_path"] = args.candidate_pool
                gen_fn(**gen_kwargs)
                generated_json += 1
            except Exception as e:
                logger.error(f"    JSONs failed: {e}")
                failed += 1
                if args.fail_fast:
                    raise
                continue

            # Step 2: Generate one parquet per combination
            # Train: one per (prompt, feedback, reward) — all reward types
            # Eval/Val: one per (prompt, feedback) — fixed reward=per_episode
            from itertools import product as iterproduct
            parquets_dir = Path(dest) / "parquets"
            parquets_dir.mkdir(parents=True, exist_ok=True)

            if pq_mode_i == "train":
                from latentgym.data.train.generate import generate_train_parquet
                combos = list(iterproduct(prompt_ids, feedback_ids, reward_types))
                for pid, fid, rt in combos:
                    parquet_name = f"{pid}_{fid}_{rt}.parquet"
                    parquet_path = str(parquets_dir / parquet_name)
                    try:
                        generate_train_parquet(
                            env_name=env_name,
                            latent_id=latent_id,
                            prompt_ids=pid,
                            feedback_ids=fid,
                            reward_types=rt,
                            trajectory_dir=dest,
                            output_path=parquet_path,
                        )
                        generated_parquet += 1
                    except Exception as e:
                        logger.error(f"    Parquet {parquet_name} failed: {e}")
                        failed += 1
                        if args.fail_fast:
                            raise
            else:
                from latentgym.data.eval.generate import generate_eval_parquet
                combos = list(iterproduct(prompt_ids, feedback_ids))
                for pid, fid in combos:
                    parquet_name = f"{pid}_{fid}.parquet"
                    parquet_path = str(parquets_dir / parquet_name)
                    try:
                        generate_eval_parquet(
                            env_name=env_name,
                            latent_id=latent_id,
                            prompt_ids=pid,
                            feedback_ids=fid,
                            reward_type="per_episode",
                            trajectory_dir=dest,
                            output_path=parquet_path,
                        )
                        generated_parquet += 1
                    except Exception as e:
                        logger.error(f"    Parquet {parquet_name} failed: {e}")
                        failed += 1
                        if args.fail_fast:
                            raise

    print(f"\nDone. JSONs: {generated_json}, Parquets: {generated_parquet}, Failed: {failed}")
    print(f"Output: {output_dir}")


def cmd_generate_parquet(args):
    """Convert trajectory JSONs → SkyRL-compatible parquet.

    Reads manifest.json + traj_*.json from --source directory,
    creates parquet rows with prompt/feedback/reward combinations.
    Same trajectory JSONs can produce multiple parquets with different configs.
    """
    import latentgym.envs  # noqa: F401
    from latentgym.core import registry

    source_dir = Path(args.source)
    if not (source_dir / "manifest.json").exists():
        raise ValueError(f"No manifest.json found in {source_dir}. Run 'generate_data eval' first.")

    from latentgym.core.trajectory_utils import load_manifest
    manifest = load_manifest(str(source_dir))
    env_name = manifest.env_name
    latent_id = manifest.latent_id

    # Validate prompt/feedback against registry
    env_prompts = set(registry.list_prompts(env_name))
    env_feedbacks = set(registry.list_feedbacks(env_name))

    from latentgym.core.reward import RewardType
    all_reward_types = [r.value for r in RewardType]

    prompt_ids = [p.strip() for p in args.prompt.split(",")] if args.prompt else sorted(env_prompts)
    feedback_ids = [f.strip() for f in args.feedback.split(",")] if args.feedback else sorted(env_feedbacks)
    reward_types = [r.strip() for r in args.reward_type.split(",")] if args.reward_type else all_reward_types

    for pid in prompt_ids:
        if pid not in env_prompts:
            raise ValueError(f"Prompt '{pid}' not in env '{env_name}'. Available: {sorted(env_prompts)}")
    for fid in feedback_ids:
        if fid not in env_feedbacks:
            raise ValueError(f"Feedback '{fid}' not in env '{env_name}'. Available: {sorted(env_feedbacks)}")

    n_trajs = len(manifest.trajectory_files)
    n_combos = len(prompt_ids) * len(feedback_ids) * len(reward_types)
    n_rows = n_trajs * n_combos

    print(f"\nParquet generation plan:")
    print(f"  Source: {source_dir} ({n_trajs} trajectories)")
    print(f"  Env: {env_name} / {latent_id}")
    print(f"  Prompts: {prompt_ids}")
    print(f"  Feedbacks: {feedback_ids}")
    print(f"  Reward types: {reward_types}")
    print(f"  Rows: {n_trajs} trajectories × {n_combos} combos = {n_rows}")
    print()

    if args.dry_run:
        logger.info("Dry run — exiting without generating")
        return

    from itertools import product as iterproduct

    # Determine output directory
    if args.output_path:
        out_dir = Path(args.output_path)
    else:
        out_dir = source_dir / "parquets"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        from latentgym.data.train.generate import generate_train_parquet as _gen_pq
    else:
        from latentgym.data.eval.generate import generate_eval_parquet as _gen_pq

    generated = 0
    for pid, fid, rt in iterproduct(prompt_ids, feedback_ids, reward_types):
        parquet_name = f"{pid}_{fid}_{rt}.parquet"
        parquet_path = str(out_dir / parquet_name)

        if args.mode == "train":
            _gen_pq(
                env_name=env_name,
                latent_id=latent_id,
                prompt_ids=pid,
                feedback_ids=fid,
                reward_types=rt,
                trajectory_dir=str(source_dir),
                output_path=parquet_path,
            )
        else:
            _gen_pq(
                env_name=env_name,
                latent_id=latent_id,
                prompt_ids=pid,
                feedback_ids=fid,
                reward_type=rt,
                trajectory_dir=str(source_dir),
                output_path=parquet_path,
            )
        generated += 1

    logger.info(f"{generated} parquets written to {out_dir}")


def cmd_generate_train(args):
    """Generate training data: trajectory JSONs + training parquet.

    Shortcut equivalent to:
        generate_data eval --split train --seed 10000 ... + parquet generation

    Requires --env and --latent (training targets a specific config).
    Uses all prompts × feedbacks × reward types for the training parquet
    (SkyRL samples from all combinations during training).
    """
    from types import SimpleNamespace

    # Delegate to cmd_generate_eval with train-specific defaults
    eval_args = SimpleNamespace(
        env=args.env,
        latent=args.latent,
        complexity=None,
        config=None,
        n_trajectories=args.n_trajectories,
        num_episodes=args.num_episodes,
        seed=args.seed,
        output=args.output,
        val_ratio=None,
        split="train",
        prompt=getattr(args, "prompt", None),
        feedback=getattr(args, "feedback", None),
        reward_type=getattr(args, "reward_type", None),
        env_param=getattr(args, "env_param", None),
        candidate_pool=getattr(args, "candidate_pool", None),
        filtered_pool_dir=getattr(args, "filtered_pool_dir", "benchmark/data/pools/"),
        dry_run=False,
        fail_fast=True,
    )
    cmd_generate_eval(eval_args)


def cmd_filter_pool(args):
    """Pre-filter a raw candidate pool through all latents for an environment.

    One-time operation. Saves filtered lists so trajectory generators can
    skip runtime filtering.
    """
    import latentgym.envs  # noqa: F401
    from latentgym.data.filter_pools import filter_all_pools

    latent_ids = [l.strip() for l in args.latent.split(",")] if args.latent else None
    output = str(Path(args.output) / args.env)

    print(f"\nPre-filtering pool for {args.env}")
    print(f"  Raw pool: {args.raw_pool}")
    print(f"  Output: {output}")
    if latent_ids:
        print(f"  Latents: {latent_ids}")
    if args.complexity:
        print(f"  Complexity: {args.complexity}")
    print()

    stats = filter_all_pools(
        env_name=args.env,
        raw_pool_path=args.raw_pool,
        output_dir=output,
        complexity=args.complexity,
        latent_ids=latent_ids,
    )

    print(f"\nDone. {len(stats)} filtered pools saved to {output}")
    empty = [lid for lid, count in stats.items() if count == 0]
    if empty:
        print(f"WARNING: {len(empty)} latents have 0 matching candidates: {empty[:5]}{'...' if len(empty) > 5 else ''}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # list
    p_list = sub.add_parser("list", help="List envs, latents, prompts, feedbacks")
    p_list.add_argument("--env", default=None,
                        help="Show details for env(s). Comma-separated. Omit for summary of all.")
    p_list.add_argument("--complexity", default=None,
                        choices=["easy", "medium", "hard", "very_hard"],
                        help="Filter latents by complexity")

    # eval
    p_eval = sub.add_parser("eval", help="Generate eval trajectory files")
    p_eval.add_argument("--env", default=None,
                        help="Environment(s). Omit for all. Comma-separated: bandits,wordle")
    p_eval.add_argument("--latent", default=None,
                        help="Latent(s). Omit for all in env. Comma-separated. Requires --env")
    p_eval.add_argument("--complexity", default=None,
                        choices=["easy", "medium", "hard", "very_hard"],
                        help="Filter latents by complexity")
    p_eval.add_argument("--n-trajectories", type=int, default=100,
                        help="Trajectories per (env, latent) pair. With --val-ratio, this is the total before split.")
    p_eval.add_argument("--num-episodes", type=int, default=10)
    p_eval.add_argument("--seed", type=int, default=42,
                        help="Base seed. Each trajectory gets seed+i. Train/val get non-overlapping ranges.")
    p_eval.add_argument("--val-ratio", type=float, default=None,
                        help="Fraction for validation split (e.g., 0.2 = 80%% train, 20%% val). "
                             "Creates train/ and val/ subdirectories with non-overlapping seeds.")
    p_eval.add_argument("--split", default=None, choices=["train", "val", "test"],
                        help="Write to a named subdirectory (e.g., --split val → <output>/<env>/<latent>/val/).")
    p_eval.add_argument("--prompt", default=None,
                        help="Prompt(s) for parquet generation. Comma-separated. Omit for all in env.")
    p_eval.add_argument("--feedback", default=None,
                        help="Feedback(s) for parquet generation. Comma-separated. Omit for all in env.")
    p_eval.add_argument("--reward-type", default=None,
                        help="Reward type(s) for train parquets. Comma-separated. "
                             "Omit for all (cumulative,terminal,improvement,per_episode). "
                             "Val parquets always use per_episode regardless.")
    p_eval.add_argument("--env-param", action="append", default=None, metavar="KEY=VALUE",
                        help="Override game params (e.g., --env-param max_turns_per_episode=10 "
                             "--env-param word_length=7). Can be repeated. Overrides registry defaults.")
    p_eval.add_argument("--candidate-pool", default=None,
                        help="Path to raw word list (one word per line). Required for wordle/hangman/wordladder "
                             "if no pre-filtered pools exist.")
    p_eval.add_argument("--filtered-pool-dir", default="benchmark/data/pools/",
                        help="Directory with pre-filtered word lists (from filter-pool command). "
                             "If provided, skips runtime filtering.")
    p_eval.add_argument("--output", default="benchmark/data/eval/")
    p_eval.add_argument("--dry-run", action="store_true",
                        help="Print generation plan without running")
    p_eval.add_argument("--fail-fast", action="store_true",
                        help="Stop on first error")

    # parquet
    p_parquet = sub.add_parser("parquet",
        help="Convert trajectory JSONs → SkyRL parquet (adds prompt/feedback/reward combos)")
    p_parquet.add_argument("--source", required=True,
                           help="Directory with manifest.json + traj_*.json (output of 'eval' command)")
    p_parquet.add_argument("--mode", required=True, choices=["train", "eval"],
                           help="'train' = all prompt×feedback×reward combos; 'eval' = single reward type")
    p_parquet.add_argument("--prompt", default=None,
                           help="Prompt(s). Comma-separated. Omit for all in env.")
    p_parquet.add_argument("--feedback", default=None,
                           help="Feedback(s). Comma-separated. Omit for all in env.")
    p_parquet.add_argument("--reward-type", default=None,
                           help="Reward type(s). Comma-separated. Omit for all.")
    p_parquet.add_argument("--output-path", default="",
                           help="Output directory for parquets. Defaults to <source>/parquets/")
    p_parquet.add_argument("--dry-run", action="store_true")

    # train (convenience: JSONs + parquet in one command)
    p_train = sub.add_parser("train",
        help="Generate training data end-to-end: trajectory JSONs + SkyRL parquet")
    p_train.add_argument("--env", default=None, help="Environment(s). Omit for all. Comma-separated.")
    p_train.add_argument("--latent", default=None, help="Latent(s). Omit for all in env. Comma-separated.")
    p_train.add_argument("--prompt", default=None,
                         help="Prompt(s) for parquet. Comma-separated. Omit for all in env.")
    p_train.add_argument("--feedback", default=None,
                         help="Feedback(s) for parquet. Comma-separated. Omit for all in env.")
    p_train.add_argument("--reward-type", default=None,
                         help="Reward type(s) for parquet. Comma-separated. Omit for all.")
    p_train.add_argument("--num-episodes", type=int, default=10)
    p_train.add_argument("--n-trajectories", type=int, default=500)
    p_train.add_argument("--seed", type=int, default=10000,
                         help="Base seed (default 10000 to avoid overlap with eval seed 42)")
    p_train.add_argument("--env-param", action="append", default=None, metavar="KEY=VALUE",
                         help="Override game params (e.g., --env-param max_turns_per_episode=10)")
    p_train.add_argument("--candidate-pool", default=None,
                         help="Path to raw word list (for wordle/hangman/wordladder filter latents)")
    p_train.add_argument("--filtered-pool-dir", default="benchmark/data/pools/",
                         help="Directory with pre-filtered word lists (default: benchmark/data/pools/)")
    p_train.add_argument("--output", default="benchmark/data/train/",
                         help="Output directory for JSONs + parquet")

    # filter-pool
    p_filter = sub.add_parser("filter-pool",
        help="Pre-filter candidate pools by latent constraints (one-time, speeds up trajectory generation)")
    p_filter.add_argument("--env", required=True,
                          help="Environment: wordle, hangman, wordladder")
    p_filter.add_argument("--raw-pool", required=True,
                          help="Path to raw candidate pool (words.txt for wordle/hangman, 'start target' pairs for wordladder)")
    p_filter.add_argument("--output", default="benchmark/data/pools/",
                          help="Output directory. Creates <output>/<env>/ with one .txt per latent.")
    p_filter.add_argument("--latent", default=None,
                          help="Specific latent(s). Comma-separated. Omit for all filter-based latents.")
    p_filter.add_argument("--complexity", default=None,
                          choices=["easy", "medium", "hard", "very_hard"],
                          help="Filter only latents of this complexity")

    args = parser.parse_args()

    if args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "eval":
        cmd_generate_eval(args)
    elif args.cmd == "parquet":
        cmd_generate_parquet(args)
    elif args.cmd == "train":
        cmd_generate_train(args)
    elif args.cmd == "filter-pool":
        cmd_filter_pool(args)


if __name__ == "__main__":
    main()
