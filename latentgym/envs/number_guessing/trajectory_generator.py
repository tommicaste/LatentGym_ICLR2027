"""Trajectory generator for number guessing (generator-based latents)."""

import random
from typing import Any, Dict, List, Optional

from latentgym.core.latent import LatentDefinition
from latentgym.core.registry import get_latent
from latentgym.core.trajectory_utils import Trajectory, Manifest, save_dataset


def generate_number_guessing_trajectories(
    latent_id: str,
    num_episodes: int = 7,
    n_trajectories: int = 100,
    seed: int = 42,
    output_dir: str = "data/eval/number_guessing/",
    env_params: Optional[Dict[str, Any]] = None,
) -> List[Trajectory]:
    """Generate trajectory JSONs for number guessing using generator latents."""
    if env_params is None:
        env_params = {"min_range": 1, "max_range": 1000, "max_turns_per_episode": 30}

    latent = get_latent("number_guessing", latent_id)
    assert latent.latent_mode == "generator", f"Latent '{latent_id}' must be generator mode"

    trajectories = []
    for traj_idx in range(n_trajectories):
        rng = random.Random(seed + traj_idx)
        # Seed the global random for the generator_fn
        random.seed(seed + traj_idx)

        context: Dict[str, Any] = {}
        episodes = []
        for ep_idx in range(num_episodes):
            config = latent.generate_episode_config(env_params, ep_idx, num_episodes, context)
            config.setdefault("max_turns_per_episode", env_params.get("max_turns_per_episode", 30))
            episodes.append(config)

        # Use actual range from generated episodes so the system prompt
        # matches what core_env tells the agent per episode.
        traj_env_params = dict(env_params)
        if episodes:
            traj_env_params["min_range"] = episodes[0].get("min_range", env_params.get("min_range", 1))
            traj_env_params["max_range"] = episodes[0].get("max_range", env_params.get("max_range", 1000))

        traj = Trajectory(
            trajectory_id=f"traj_{traj_idx:03d}",
            latent_id=latent_id,
            episodes=episodes,
            metadata={
                "env_name": "number_guessing",
                "num_episodes": num_episodes,
                "env_params": traj_env_params,
                "context": {k: v for k, v in context.items()
                           if isinstance(v, (int, float, str, list, tuple))},
            }
        )
        trajectories.append(traj)

    manifest = Manifest(
        env_name="number_guessing",
        latent_id=latent_id,
        num_episodes=num_episodes,
        seed=seed,
        n_trajectories=n_trajectories,
        trajectory_files=[],
        extra={"env_params": env_params},
    )
    save_dataset(trajectories, output_dir, manifest)

    return trajectories
