"""Trajectory generator for mastermind (filter-based latents on codes)."""

import random
from itertools import product
from typing import Any, Dict, List, Optional

from latentgym.core.registry import get_latent
from latentgym.core.trajectory_utils import Trajectory, Manifest, save_dataset


def _generate_all_codes(code_length: int, num_numbers: int, duplicates_allowed: bool) -> List[List[int]]:
    """Generate all valid codes for given parameters."""
    digits = list(range(1, num_numbers + 1))
    if duplicates_allowed:
        return [list(c) for c in product(digits, repeat=code_length)]
    else:
        # Permutations without repetition
        from itertools import permutations
        return [list(c) for c in permutations(digits, code_length)]


def generate_mastermind_trajectories(
    latent_id: str,
    num_episodes: int = 10,
    n_trajectories: int = 100,
    seed: int = 42,
    output_dir: str = "data/eval/mastermind/",
    env_params: Optional[Dict[str, Any]] = None,
) -> List[Trajectory]:
    """Generate trajectory JSONs for mastermind using filter latents."""
    if env_params is None:
        env_params = {"code_length": 4, "num_numbers": 6, "duplicates_allowed": False, "max_turns_per_episode": 10}

    latent = get_latent("mastermind", latent_id)
    assert latent.latent_mode == "filter", f"Latent '{latent_id}' must be filter mode"

    code_length = env_params.get("code_length", 4)
    num_numbers = env_params.get("num_numbers", 6)
    duplicates = env_params.get("duplicates_allowed", False)
    max_turns = env_params.get("max_turns_per_episode", env_params.get("max_turns", 10))

    # Generate and filter all valid codes
    all_codes = _generate_all_codes(code_length, num_numbers, duplicates)
    filtered = [c for c in all_codes if latent.filter_candidate(c)]

    if not filtered:
        raise ValueError(f"No codes pass filter for latent '{latent_id}' with "
                        f"code_length={code_length}, num_numbers={num_numbers}, "
                        f"duplicates_allowed={duplicates}")

    print(f"Latent '{latent_id}': {len(filtered)}/{len(all_codes)} codes pass filter")

    trajectories = []
    for traj_idx in range(n_trajectories):
        rng = random.Random(seed + traj_idx)
        episodes = []
        for ep_idx in range(num_episodes):
            code = rng.choice(filtered)
            episodes.append({
                "secret_code": code,
                "code_length": code_length,
                "num_numbers": num_numbers,
                "duplicates_allowed": duplicates,
                "max_turns_per_episode": max_turns,
            })
        traj = Trajectory(
            trajectory_id=f"traj_{traj_idx:03d}",
            latent_id=latent_id,
            episodes=episodes,
        )
        trajectories.append(traj)

    manifest = Manifest(
        env_name="mastermind",
        latent_id=latent_id,
        num_episodes=num_episodes,
        seed=seed,
        n_trajectories=n_trajectories,
        trajectory_files=[],
        extra={"env_params": env_params, "total_codes": len(all_codes), "filtered_codes": len(filtered)},
    )
    save_dataset(trajectories, output_dir, manifest)

    return trajectories
