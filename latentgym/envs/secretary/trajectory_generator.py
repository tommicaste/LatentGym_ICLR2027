"""Trajectory generator for secretary problem (generator-based latents)."""

import random
from typing import Any, Dict, List, Optional

from latentgym.core.registry import get_latent
from latentgym.core.trajectory_utils import Trajectory, Manifest, save_dataset


def generate_secretary_trajectories(
    latent_id: str,
    num_episodes: int = 10,
    n_trajectories: int = 100,
    seed: int = 42,
    output_dir: str = "data/eval/secretary/",
    env_params: Optional[Dict[str, Any]] = None,
) -> List[Trajectory]:
    """Generate trajectory JSONs for secretary using generator latents."""
    if env_params is None:
        env_params = {"num_draws": 10, "max_turns_per_episode": 10}

    latent = get_latent("secretary", latent_id)
    assert latent.latent_mode == "generator", f"Latent '{latent_id}' must be generator mode"

    trajectories = []
    for traj_idx in range(n_trajectories):
        random.seed(seed + traj_idx)
        context: Dict[str, Any] = {}
        episodes = []
        for ep_idx in range(num_episodes):
            config = latent.generate_episode_config(env_params, ep_idx, num_episodes, context)
            config.setdefault("max_turns_per_episode", env_params.get("max_turns_per_episode", 10))
            episodes.append(config)
        traj = Trajectory(
            trajectory_id=f"traj_{traj_idx:03d}",
            latent_id=latent_id,
            episodes=episodes,
        )
        trajectories.append(traj)

    manifest = Manifest(
        env_name="secretary",
        latent_id=latent_id,
        num_episodes=num_episodes,
        seed=seed,
        n_trajectories=n_trajectories,
        trajectory_files=[],
        extra={"env_params": env_params},
    )
    save_dataset(trajectories, output_dir, manifest)

    return trajectories
