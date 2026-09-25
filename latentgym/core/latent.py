"""
Latent definitions for multi-episode environments.

A latent defines how episode configurations change across episodes in a trajectory.
Two modes are supported:

1. generator_fn: Dynamically produces episode config per episode (e.g., bandits, number guessing)
2. filter_fn: Filters candidates from a pool (e.g., wordle word selection)

Both modes require a seed to produce concrete ground truth.
Latents can also have hyperparameters.
CrossEpisodeLatent wraps multiple base latents with sequencing patterns.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class LatentComplexity(Enum):
    """Complexity levels for latent constraints."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"


@dataclass
class LatentDefinition:
    """Definition of a latent constraint for multi-episode environments.

    A latent determines how episode configurations vary across episodes.
    Exactly one of generator_fn or filter_fn must be set.

    Both modes need a seed to produce concrete ground truth:
        - generator_fn + seed → episode_configs (ground truth generated directly)
        - filter_fn + seed + candidate_pool → episode_configs (sampled from filtered pool)

    Attributes:
        id: Unique identifier (e.g., "loyal_favorite_0", "vowel_count_2")
        name: Human-readable name
        complexity: Difficulty level
        description: Detailed description of the constraint
        is_cross_episode: Whether the pattern spans across episodes
        hyperparams: User-tunable parameters passed to generator_fn/filter_fn
        generator_fn: (env_params, ep_idx, n_eps, ctx, **hp) → episode_config
        filter_fn: (candidate, **hp) → bool
    """
    id: str
    name: str
    complexity: LatentComplexity
    description: str
    is_cross_episode: bool = False
    hyperparams: Dict[str, Any] = field(default_factory=dict)

    # Exactly one of these two must be set:
    generator_fn: Optional[Callable] = None
    filter_fn: Optional[Callable] = None

    def __post_init__(self):
        modes_set = sum([
            self.generator_fn is not None,
            self.filter_fn is not None,
        ])
        if modes_set != 1:
            raise ValueError(
                f"LatentDefinition '{self.id}' must have exactly one of "
                f"generator_fn or filter_fn set. Got {modes_set}."
            )

    @property
    def latent_mode(self) -> str:
        """Return the mode of this latent: 'generator' or 'filter'."""
        if self.generator_fn is not None:
            return "generator"
        return "filter"

    def with_hyperparams(self, **kwargs) -> LatentDefinition:
        """Return a copy with updated hyperparams."""
        new = copy.deepcopy(self)
        new.hyperparams = {**self.hyperparams, **kwargs}
        hp_suffix = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        if hp_suffix:
            new.id = f"{self.id}_{hp_suffix}"
        return new

    def generate_episode_config(
        self,
        env_params: Dict[str, Any],
        episode_idx: int,
        num_episodes: int,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate config for a single episode (generator mode only).

        Args:
            env_params: Environment-level parameters
            episode_idx: Current episode index (0-based)
            num_episodes: Total number of episodes
            context: Context from previous episodes

        Returns:
            Episode configuration dict.

        Raises:
            ValueError: If latent is filter mode.
        """
        if self.generator_fn is not None:
            return self.generator_fn(
                env_params, episode_idx, num_episodes, context, **self.hyperparams
            )
        raise ValueError(
            f"Latent '{self.id}' is filter-based. Use filter_candidate() "
            f"with a candidate pool, or provide a trajectory_path."
        )

    def filter_candidate(self, candidate: Any) -> bool:
        """Check if a candidate satisfies this latent's filter (filter mode only).

        Args:
            candidate: Item to test (e.g., a word).

        Returns:
            True if candidate satisfies the constraint.

        Raises:
            ValueError: If latent is generator mode.
        """
        if self.filter_fn is not None:
            return self.filter_fn(candidate, **self.hyperparams)
        raise ValueError(
            f"Latent '{self.id}' is generator-based (mode: {self.latent_mode}). "
            f"Use generate_episode_config() instead."
        )


@dataclass
class CrossEpisodeLatent:
    """A latent that sequences multiple base latents across episodes.

    Wraps multiple registered LatentDefinition IDs with a pattern
    (alternating, cyclic, progressive) to create cross-episode structure.

    Attributes:
        id: Unique identifier
        name: Human-readable name
        complexity: Difficulty level
        pattern_type: How base latents are sequenced
        base_latent_ids: List of latent IDs to sequence through
        cycle_length: For cyclic patterns, how many episodes per cycle
        description: Detailed description
    """
    id: str
    name: str
    complexity: LatentComplexity
    pattern_type: str
    base_latent_ids: List[str]
    cycle_length: int = 2
    description: str = ""

    def get_latent_id_for_episode(self, episode_idx: int, num_episodes: int) -> str:
        """Determine which base latent to use for a given episode."""
        if self.pattern_type == "alternating":
            return self.base_latent_ids[episode_idx % len(self.base_latent_ids)]
        elif self.pattern_type == "cyclic":
            cycle_pos = episode_idx % self.cycle_length
            latent_idx = cycle_pos % len(self.base_latent_ids)
            return self.base_latent_ids[latent_idx]
        elif self.pattern_type == "progressive":
            chunk_size = max(1, num_episodes // len(self.base_latent_ids))
            latent_idx = min(episode_idx // chunk_size, len(self.base_latent_ids) - 1)
            return self.base_latent_ids[latent_idx]
        else:
            raise ValueError(f"Unknown pattern type: {self.pattern_type}")
