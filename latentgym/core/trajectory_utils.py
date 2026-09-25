"""
Shared trajectory utilities — save, load, manifest management.

Used by per-env trajectory generators. Provides the common JSON format
and directory structure for trajectory datasets.

Directory structure:
    output_dir/
    ├── traj_000.json
    ├── traj_001.json
    ├── ...
    └── manifest.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Trajectory:
    """A single multi-episode trajectory with pre-resolved episode configs."""
    trajectory_id: str
    latent_id: str
    episodes: List[Dict[str, Any]]  # Pre-resolved episode configs
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "latent_id": self.latent_id,
            "episodes": self.episodes,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Trajectory:
        return cls(
            trajectory_id=data["trajectory_id"],
            latent_id=data.get("latent_id", ""),
            episodes=data["episodes"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class Manifest:
    """Metadata for a trajectory dataset.

    Contains only data-generation fields (env, latent, episodes, seed).
    Does NOT contain prompt_id or feedback_id — those are eval-time choices
    and are orthogonal to the ground truth data stored in trajectories.
    """
    env_name: str
    latent_id: str
    num_episodes: int
    seed: int
    n_trajectories: int
    trajectory_files: List[str]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Manifest:
        extra = data.pop("extra", {})
        # Handle unknown fields gracefully
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        filtered["extra"] = extra
        return cls(**filtered)


# =============================================================================
# Save / Load
# =============================================================================

def save_trajectory(trajectory: Trajectory, path: str) -> None:
    """Save a single trajectory to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(trajectory.to_json())


def load_trajectory(path: str) -> Trajectory:
    """Load a single trajectory from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return Trajectory.from_dict(data)


def save_manifest(manifest: Manifest, output_dir: str) -> None:
    """Save manifest to output_dir/manifest.json."""
    p = Path(output_dir) / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(manifest.to_json())


def load_manifest(output_dir: str) -> Manifest:
    """Load manifest from output_dir/manifest.json."""
    p = Path(output_dir) / "manifest.json"
    with open(p, "r") as f:
        data = json.load(f)
    return Manifest.from_dict(data)


def save_dataset(
    trajectories: List[Trajectory],
    output_dir: str,
    manifest: Manifest,
) -> None:
    """Save a full dataset: trajectory files + manifest.

    Args:
        trajectories: List of trajectories to save
        output_dir: Directory to save into
        manifest: Dataset manifest (trajectory_files will be auto-populated)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    traj_files = []
    for traj in trajectories:
        filename = f"{traj.trajectory_id}.json"
        save_trajectory(traj, str(out / filename))
        traj_files.append(filename)

    manifest.trajectory_files = traj_files
    save_manifest(manifest, output_dir)


def load_dataset(output_dir: str) -> tuple:
    """Load a full dataset: manifest + all trajectories.

    Args:
        output_dir: Directory containing manifest.json + trajectory files

    Returns:
        (manifest, trajectories)
    """
    manifest = load_manifest(output_dir)
    out = Path(output_dir)
    trajectories = []
    for filename in manifest.trajectory_files:
        traj = load_trajectory(str(out / filename))
        trajectories.append(traj)
    return manifest, trajectories
