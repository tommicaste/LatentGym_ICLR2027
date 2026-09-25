"""
Trajectory generator for the Wordle environment.

Uses filter_fn latents to select target words from a candidate pool.
Saves trajectory JSON files compatible with make_env(trajectory_path=...).

Usage:
    python -m latentgym.envs.wordle.trajectory_generator \
        --latent "vowel_count_2" \
        --num-episodes 5 \
        --n-trajectories 100 \
        --seed 42 \
        --candidate-pool word_lists/5letter.txt \
        --output-dir data/eval/wordle_vowel_count_2/
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from latentgym.core.latent import LatentDefinition
from latentgym.core.registry import get_latent, get_env_registration
from latentgym.core.trajectory_utils import (
    Trajectory, Manifest, save_dataset,
)

ENV_NAME = "wordle"


def load_candidate_pool(pool_path: str) -> List[str]:
    """Load candidate words from a text file (one word per line)."""
    p = Path(pool_path)
    if not p.exists():
        raise FileNotFoundError(f"Candidate pool not found: {pool_path}")
    with open(p, "r") as f:
        return [line.strip().lower() for line in f if line.strip()]


def generate_wordle_trajectories(
    latent_id: str,
    num_episodes: int,
    n_trajectories: int,
    seed: int,
    candidate_pool_path: str = "",
    output_dir: str = "data/eval/wordle/",
    filtered_pool_dir: str = "",
    sampling: str = "without_replacement",
    max_attempts: int = 6,
    env_params: Optional[Dict[str, Any]] = None,
) -> List[Trajectory]:
    """Generate wordle trajectory dataset.

    Args:
        latent_id: Which latent constraint to use
        num_episodes: Episodes per trajectory
        n_trajectories: How many trajectories to generate
        seed: Base random seed
        candidate_pool_path: Path to raw word list (one word per line).
            Used for runtime filtering if filtered_pool_dir is not provided.
        output_dir: Where to save trajectory JSONs
        filtered_pool_dir: Path to pre-filtered pool directory (from filter_pools.py).
            If provided AND contains a file for this latent, skips runtime filtering.
        sampling: "with_replacement" or "without_replacement"
        max_attempts: Max guesses per episode (deprecated, use env_params)
        env_params: Override game params (word_length, max_turns_per_episode, etc.).
                    If None, uses registry defaults.

    Returns:
        List of generated trajectories
    """
    latent = get_latent(ENV_NAME, latent_id)
    env_reg = get_env_registration(ENV_NAME)
    # Registry defaults, overridden by caller
    _env_params = {**env_reg.env_params}
    if env_params:
        _env_params.update(env_params)
    env_params = _env_params
    # Allow sampling to be overridden via env_params (e.g. --env-param sampling=with_replacement)
    if "sampling" in env_params:
        sampling = env_params.pop("sampling")
    # Also respect legacy max_attempts param if env_params didn't set it
    if max_attempts != 6 and "max_turns_per_episode" not in (env_params or {}):
        env_params["max_turns_per_episode"] = max_attempts

    # Try pre-filtered pool first (fast path)
    filtered_pool = None
    if filtered_pool_dir:
        from latentgym.data.filter_pools import load_filtered_pool
        filtered_pool = load_filtered_pool(filtered_pool_dir, latent_id)
        if filtered_pool:
            print(f"Latent '{latent_id}': loaded {len(filtered_pool)} pre-filtered words")

    # Fall back to runtime filtering (slow path)
    if not filtered_pool:
        if not candidate_pool_path:
            raise ValueError(
                f"No pre-filtered pool found for '{latent_id}' in '{filtered_pool_dir}' "
                f"and no candidate_pool_path provided. Either run 'generate_data filter-pool' first, "
                f"or provide --candidate-pool."
            )
        pool = load_candidate_pool(candidate_pool_path)
        filtered_pool = [w for w in pool if latent.filter_candidate(w)]
        print(f"Latent '{latent_id}': {len(filtered_pool)}/{len(pool)} words pass filter (runtime filtered)")

    if not filtered_pool:
        raise ValueError(
            f"No candidates passed filter for latent '{latent_id}'."
        )

    trajectories = []

    for i in range(n_trajectories):
        traj_seed = seed + i
        rng = random.Random(traj_seed)

        episode_configs = []
        used_words = set()

        for ep_idx in range(num_episodes):
            if sampling == "without_replacement":
                available = [w for w in filtered_pool if w not in used_words]
                if not available:
                    # Fallback to with_replacement if pool exhausted
                    available = filtered_pool
                target = rng.choice(available)
                used_words.add(target)
            else:
                target = rng.choice(filtered_pool)

            episode_configs.append({
                "target_word": target,
                "max_turns_per_episode": env_params.get("max_turns_per_episode", max_attempts),
                "word_length": env_params.get("word_length", 5),
            })

        traj = Trajectory(
            trajectory_id=f"traj_{i:06d}",
            latent_id=latent_id,
            episodes=episode_configs,
            metadata={
                "seed": traj_seed,
                "num_episodes": num_episodes,
                "latent_complexity": latent.complexity.value,
                "pool_size": len(filtered_pool),
                "sampling": sampling,
            },
        )
        trajectories.append(traj)

    # Save
    manifest = Manifest(
        env_name=ENV_NAME,
        latent_id=latent_id,
        num_episodes=num_episodes,
        seed=seed,
        n_trajectories=n_trajectories,
        trajectory_files=[],
        extra={
            "candidate_pool_path": candidate_pool_path,
            "filtered_pool_size": len(filtered_pool),
            "sampling": sampling,
            "env_params": env_params,
        },
    )
    save_dataset(trajectories, output_dir, manifest)

    return trajectories


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, ".")

    import latentgym.envs.wordle  # noqa: F401

    parser = argparse.ArgumentParser(description="Generate wordle trajectories")
    parser.add_argument("--latent", required=True)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--n-trajectories", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sampling", default="without_replacement",
                        choices=["with_replacement", "without_replacement"])
    args = parser.parse_args()

    trajs = generate_wordle_trajectories(
        latent_id=args.latent,
        num_episodes=args.num_episodes,
        n_trajectories=args.n_trajectories,
        seed=args.seed,
        candidate_pool_path=args.candidate_pool,
        output_dir=args.output_dir,
        sampling=args.sampling,
    )
    print(f"Generated {len(trajs)} trajectories → {args.output_dir}")
