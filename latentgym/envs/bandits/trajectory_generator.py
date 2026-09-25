"""
Trajectory generator for the Bandits environment.

Uses generator_fn latents to produce ground truth probabilities per episode.
Saves trajectory JSON files compatible with make_env(trajectory_path=...).

Usage:
    python -m latentgym.envs.bandits.trajectory_generator \
        --latent "loyal_favorite_0" \
        --num-episodes 10 \
        --n-trajectories 100 \
        --seed 42 \
        --output-dir data/eval/bandits_loyal_favorite_0/
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from latentgym.core.latent import LatentDefinition
from latentgym.core.registry import get_latent, get_env_registration
from latentgym.core.trajectory_utils import (
    Trajectory, Manifest, save_dataset,
)

ENV_NAME = "bandits"


def generate_bandit_trajectories(
    latent_id: str,
    num_episodes: int,
    n_trajectories: int,
    seed: int,
    output_dir: str,
    env_params: Optional[Dict[str, Any]] = None,
) -> List[Trajectory]:
    """Generate bandit trajectory dataset.

    Args:
        latent_id: Which latent constraint to use
        num_episodes: Episodes per trajectory
        n_trajectories: How many trajectories to generate
        seed: Base random seed
        output_dir: Where to save trajectory JSONs
        env_params: Override game params (buttons, max_turns_per_episode, etc.).
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

    trajectories = []

    for i in range(n_trajectories):
        traj_seed = seed + i

        # Seed random for generator_fn compatibility
        old_state = random.getstate()
        random.seed(traj_seed)

        episode_configs = []
        context: Dict[str, Any] = {}

        # Initialize context for latents that need it (e.g., value_decay needs fixed_best_idx)
        rng = random.Random(traj_seed)
        if latent.is_cross_episode:
            context["fixed_best_idx"] = rng.randrange(
                len(env_params.get("buttons", []))
            )

        for ep_idx in range(num_episodes):
            config = latent.generate_episode_config(
                env_params, ep_idx, num_episodes, context
            )
            # Add game params that generator_fn doesn't include
            config.setdefault("max_turns_per_episode", env_params.get("max_turns_per_episode", 20))
            config.setdefault("buttons", env_params.get("buttons", ["red", "blue", "green", "yellow", "purple"]))
            episode_configs.append(config)

        random.setstate(old_state)

        traj = Trajectory(
            trajectory_id=f"traj_{i:06d}",
            latent_id=latent_id,
            episodes=episode_configs,
            metadata={
                "seed": traj_seed,
                "num_episodes": num_episodes,
                "latent_complexity": latent.complexity.value,
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
        trajectory_files=[],  # Populated by save_dataset
        extra={"env_params": env_params},
    )
    save_dataset(trajectories, output_dir, manifest)

    return trajectories


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, ".")

    # Trigger registration
    import latentgym.envs.bandits  # noqa: F401

    parser = argparse.ArgumentParser(description="Generate bandit trajectories")
    parser.add_argument("--latent", required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--n-trajectories", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    trajs = generate_bandit_trajectories(
        latent_id=args.latent,
        num_episodes=args.num_episodes,
        n_trajectories=args.n_trajectories,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"Generated {len(trajs)} trajectories → {args.output_dir}")
