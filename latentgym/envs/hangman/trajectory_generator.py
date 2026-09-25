"""Trajectory generator for hangman (filter-based latents)."""

import random
from typing import Any, Dict, List, Optional

from latentgym.core.registry import get_latent
from latentgym.core.trajectory_utils import Trajectory, Manifest, save_dataset


def generate_hangman_trajectories(
    latent_id: str,
    num_episodes: int = 10,
    n_trajectories: int = 100,
    seed: int = 42,
    candidate_pool_path: str = "",
    output_dir: str = "data/eval/hangman/",
    filtered_pool_dir: str = "",
    env_params: Optional[Dict[str, Any]] = None,
) -> List[Trajectory]:
    """Generate trajectory JSONs for hangman using filter latents."""
    if env_params is None:
        env_params = {"max_turns_per_episode": 6}

    latent = get_latent("hangman", latent_id)
    assert latent.latent_mode == "filter", f"Latent '{latent_id}' must be filter mode"

    # Try pre-filtered pool first
    filtered = None
    if filtered_pool_dir:
        from latentgym.data.filter_pools import load_filtered_pool
        filtered = load_filtered_pool(filtered_pool_dir, latent_id)
        if filtered:
            print(f"Latent '{latent_id}': loaded {len(filtered)} pre-filtered words")

    # Fall back to runtime filtering
    if not filtered:
        if not candidate_pool_path:
            raise ValueError(
                "Provide --candidate-pool or run 'generate_data filter-pool' first"
            )
        with open(candidate_pool_path, 'r') as f:
            all_words = [w.strip().lower() for w in f if w.strip()]
        filtered = [w for w in all_words if latent.filter_candidate(w)]
        print(f"Latent '{latent_id}': {len(filtered)}/{len(all_words)} words pass filter (runtime)")

    if not filtered:
        raise ValueError(f"No words pass filter for latent '{latent_id}'")

    trajectories = []
    for traj_idx in range(n_trajectories):
        rng = random.Random(seed + traj_idx)
        episodes = []
        for ep_idx in range(num_episodes):
            target_word = rng.choice(filtered)
            episodes.append({
                "target_word": target_word,
                "max_turns_per_episode": env_params.get("max_turns_per_episode", env_params.get("max_attempts", 6)),
            })
        traj = Trajectory(
            trajectory_id=f"traj_{traj_idx:03d}",
            latent_id=latent_id,
            episodes=episodes,
        )
        trajectories.append(traj)

    manifest = Manifest(
        env_name="hangman",
        latent_id=latent_id,
        num_episodes=num_episodes,
        seed=seed,
        n_trajectories=n_trajectories,
        trajectory_files=[],
        extra={"env_params": env_params, "filtered_size": len(filtered)},
    )
    save_dataset(trajectories, output_dir, manifest)

    return trajectories
