"""
Registry — Central registration system for benchmark components.

All environments, latents, prompts, and feedbacks are registered here.
Components are scoped per environment (e.g., bandit latents can't be used with wordle).

The registry provides:
    - register_env(): Register a new core environment
    - register_latent(): Register a latent for a specific env
    - register_prompt(): Register a prompt template for a specific env
    - register_feedback(): Register a feedback handler for a specific env
    - make_env(): Construct a MultiEpisodeEnv from a FullyDefinedEnv
    - list_*(): Discovery functions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from .single_episode_env import SingleEpisodeEnv
from .latent import LatentDefinition, CrossEpisodeLatent
from .prompt import PromptTemplate
from .feedback import FeedbackHandler
from .reward import RewardAggregator, RewardType
from .env_config import FullyDefinedEnv


@dataclass
class EnvRegistration:
    """Registration entry for a benchmark environment."""
    name: str
    env_class: Type[SingleEpisodeEnv]
    default_num_episodes: int = 10
    env_params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


# ============================================================================
# Global registries — scoped per environment
# ============================================================================

# env_name -> EnvRegistration
_ENV_REGISTRY: Dict[str, EnvRegistration] = {}

# env_name -> {latent_id -> LatentDefinition}
_LATENT_REGISTRY: Dict[str, Dict[str, LatentDefinition]] = {}

# env_name -> {latent_id -> CrossEpisodeLatent}
_CROSS_EPISODE_LATENT_REGISTRY: Dict[str, Dict[str, CrossEpisodeLatent]] = {}

# env_name -> {prompt_id -> PromptTemplate class}
_PROMPT_REGISTRY: Dict[str, Dict[str, Type[PromptTemplate]]] = {}

# env_name -> {feedback_id -> FeedbackHandler class}
_FEEDBACK_REGISTRY: Dict[str, Dict[str, Type[FeedbackHandler]]] = {}


# ============================================================================
# Registration functions
# ============================================================================

def register_env(
    name: str,
    env_class: Type[SingleEpisodeEnv],
    default_num_episodes: int = 10,
    description: str = "",
    **env_params,
) -> None:
    """Register a new benchmark environment.

    Args:
        name: Unique environment name (e.g., "bandits", "wordle")
        env_class: SingleEpisodeEnv subclass
        default_num_episodes: Default number of episodes
        description: Human-readable description
        **env_params: Default environment parameters
    """
    if name in _ENV_REGISTRY:
        raise ValueError(f"Environment '{name}' is already registered.")
    _ENV_REGISTRY[name] = EnvRegistration(
        name=name,
        env_class=env_class,
        default_num_episodes=default_num_episodes,
        env_params=env_params,
        description=description,
    )
    # Initialize per-env registries
    _LATENT_REGISTRY.setdefault(name, {})
    _CROSS_EPISODE_LATENT_REGISTRY.setdefault(name, {})
    _PROMPT_REGISTRY.setdefault(name, {})
    _FEEDBACK_REGISTRY.setdefault(name, {})


def register_latent(env_name: str, latent: LatentDefinition) -> None:
    """Register a latent for a specific environment.

    Args:
        env_name: Environment this latent belongs to
        latent: LatentDefinition to register
    """
    _ensure_env_exists(env_name)
    if latent.id in _LATENT_REGISTRY[env_name]:
        raise ValueError(
            f"Latent '{latent.id}' already registered for env '{env_name}'."
        )
    _LATENT_REGISTRY[env_name][latent.id] = latent


def register_cross_episode_latent(env_name: str, latent: CrossEpisodeLatent) -> None:
    """Register a cross-episode latent for a specific environment.

    Args:
        env_name: Environment this latent belongs to
        latent: CrossEpisodeLatent to register
    """
    _ensure_env_exists(env_name)
    if latent.id in _CROSS_EPISODE_LATENT_REGISTRY[env_name]:
        raise ValueError(
            f"Cross-episode latent '{latent.id}' already registered for env '{env_name}'."
        )
    # Validate base latent IDs exist
    for base_id in latent.base_latent_ids:
        if base_id not in _LATENT_REGISTRY.get(env_name, {}):
            raise ValueError(
                f"Base latent '{base_id}' not found for env '{env_name}'. "
                f"Register base latents before cross-episode latents."
            )
    _CROSS_EPISODE_LATENT_REGISTRY[env_name][latent.id] = latent


def register_prompt(env_name: str, prompt_class: Type[PromptTemplate]) -> None:
    """Register a prompt template for a specific environment.

    Args:
        env_name: Environment this prompt belongs to
        prompt_class: PromptTemplate subclass (must have 'id' attribute)
    """
    _ensure_env_exists(env_name)
    prompt_id = prompt_class.id
    if prompt_id in _PROMPT_REGISTRY[env_name]:
        raise ValueError(
            f"Prompt '{prompt_id}' already registered for env '{env_name}'."
        )
    _PROMPT_REGISTRY[env_name][prompt_id] = prompt_class


def register_feedback(env_name: str, feedback_class: Type[FeedbackHandler]) -> None:
    """Register a feedback handler for a specific environment.

    Args:
        env_name: Environment this feedback belongs to
        feedback_class: FeedbackHandler subclass (must have 'id' attribute)
    """
    _ensure_env_exists(env_name)
    feedback_id = feedback_class.id
    if feedback_id in _FEEDBACK_REGISTRY[env_name]:
        raise ValueError(
            f"Feedback '{feedback_id}' already registered for env '{env_name}'."
        )
    _FEEDBACK_REGISTRY[env_name][feedback_id] = feedback_class


# ============================================================================
# Factory
# ============================================================================

def make_env(
    fully_defined: FullyDefinedEnv,
    seed: Optional[int] = None,
    trajectory_path: Optional[str] = None,
    candidate_pool_path: Optional[str] = None,
):
    """Construct a MultiEpisodeEnv from a FullyDefinedEnv specification.

    Resolves components from registries and resolves episode_configs based
    on the latent mode:

        trajectory_path provided → load episode_configs from JSON (any latent mode)
        generator latent         → call generator_fn with seed
        filter latent            → load candidate pool, filter, sample with seed

    Args:
        fully_defined: Complete environment specification.
        seed: Random seed for generating episode_configs.
            Required for generator and filter latents (unless trajectory_path given).
        trajectory_path: Path to pre-generated trajectory JSON file.
            If provided, episode_configs are loaded from file and seed is ignored.
        candidate_pool_path: Path to candidate pool file (one item per line).
            Required for filter latents (unless trajectory_path given).

    Returns:
        MultiEpisodeEnv instance ready for init() and step().
    """
    import json
    import random
    from pathlib import Path
    from .multi_episode_env import MultiEpisodeEnv

    env_name = fully_defined.env_name
    _ensure_env_exists(env_name)

    # Resolve components from registry
    env_reg = _ENV_REGISTRY[env_name]
    core_env = env_reg.env_class()

    latent = get_latent(env_name, fully_defined.latent_id)
    if fully_defined.latent_hyperparams:
        latent = latent.with_hyperparams(**fully_defined.latent_hyperparams)

    prompt_class = get_prompt_class(env_name, fully_defined.prompt_id)
    prompt_template = prompt_class()

    feedback_class = get_feedback_class(env_name, fully_defined.feedback_id)
    feedback_handler = feedback_class()

    reward_aggregator = RewardAggregator(fully_defined.reward_type)

    env_params = {**env_reg.env_params, **fully_defined.env_params}
    env_params["latent_id"] = fully_defined.latent_id
    num_episodes = fully_defined.num_episodes or env_reg.default_num_episodes

    # ── Resolve episode_configs ──────────────────────────────────────────

    if trajectory_path is not None:
        # Load from pre-generated trajectory JSON
        with open(trajectory_path, "r") as f:
            traj_data = json.load(f)
        episode_configs = traj_data["episodes"]
        num_episodes = len(episode_configs)
        # Override env_params from trajectory metadata (dynamic latents set
        # per-trajectory values like min_range, max_range that differ from registry defaults)
        traj_metadata = traj_data.get("metadata", {})
        traj_env_params = traj_metadata.get("env_params", {})
        env_params.update(traj_env_params)

    elif latent.latent_mode == "generator":
        if seed is None:
            raise ValueError(
                f"Generator latent '{latent.id}' requires seed. "
                f"Pass seed=42 or use trajectory_path."
            )
        rng = random.Random(seed)
        # Seed the global random for generator_fn compatibility
        old_state = random.getstate()
        random.seed(seed)

        episode_configs = []
        context: Dict[str, Any] = {}
        for ep_idx in range(num_episodes):
            config = latent.generate_episode_config(
                env_params, ep_idx, num_episodes, context
            )
            episode_configs.append(config)

        random.setstate(old_state)

    elif latent.latent_mode == "filter":
        if seed is None or candidate_pool_path is None:
            raise ValueError(
                f"Filter latent '{latent.id}' requires seed and candidate_pool_path. "
                f"Pass both or use trajectory_path."
            )
        # Load candidate pool
        pool_path = Path(candidate_pool_path)
        if not pool_path.exists():
            raise FileNotFoundError(f"Candidate pool not found: {candidate_pool_path}")
        with open(pool_path, "r") as f:
            pool = [line.strip() for line in f if line.strip()]

        # Filter pool
        filtered = [item for item in pool if latent.filter_candidate(item)]
        if not filtered:
            raise ValueError(
                f"No candidates passed filter for latent '{latent.id}'. "
                f"Pool size: {len(pool)}, all filtered out."
            )

        # Sample with seed
        rng = random.Random(seed)
        episode_configs = []
        for _ in range(num_episodes):
            target = rng.choice(filtered)
            episode_configs.append({"target_word": target})

    else:
        raise ValueError(f"Unknown latent mode: {latent.latent_mode}")

    # ── Construct MultiEpisodeEnv ────────────────────────────────────────

    return MultiEpisodeEnv.from_configs(
        core_env=core_env,
        episode_configs=episode_configs,
        prompt_template=prompt_template,
        feedback_handler=feedback_handler,
        reward_aggregator=reward_aggregator,
        env_params=env_params,
        metadata={
            "latent_id": latent.id,
            "latent_mode": "trajectory" if trajectory_path else latent.latent_mode,
        },
    )


# ============================================================================
# Getters
# ============================================================================

def get_env_registration(env_name: str) -> EnvRegistration:
    """Get the registration entry for an environment."""
    _ensure_env_exists(env_name)
    return _ENV_REGISTRY[env_name]


def get_latent(env_name: str, latent_id: str) -> LatentDefinition:
    """Get a latent definition for an environment."""
    _ensure_env_exists(env_name)
    if latent_id in _LATENT_REGISTRY.get(env_name, {}):
        return _LATENT_REGISTRY[env_name][latent_id]
    raise ValueError(
        f"Latent '{latent_id}' not found for env '{env_name}'. "
        f"Available: {list(_LATENT_REGISTRY.get(env_name, {}).keys())}"
    )


def get_cross_episode_latent(env_name: str, latent_id: str) -> CrossEpisodeLatent:
    """Get a cross-episode latent for an environment."""
    _ensure_env_exists(env_name)
    if latent_id in _CROSS_EPISODE_LATENT_REGISTRY.get(env_name, {}):
        return _CROSS_EPISODE_LATENT_REGISTRY[env_name][latent_id]
    raise ValueError(
        f"Cross-episode latent '{latent_id}' not found for env '{env_name}'."
    )


def get_prompt_class(env_name: str, prompt_id: str) -> Type[PromptTemplate]:
    """Get a prompt template class for an environment."""
    _ensure_env_exists(env_name)
    if prompt_id in _PROMPT_REGISTRY.get(env_name, {}):
        return _PROMPT_REGISTRY[env_name][prompt_id]
    raise ValueError(
        f"Prompt '{prompt_id}' not found for env '{env_name}'. "
        f"Available: {list(_PROMPT_REGISTRY.get(env_name, {}).keys())}"
    )


def get_feedback_class(env_name: str, feedback_id: str) -> Type[FeedbackHandler]:
    """Get a feedback handler class for an environment."""
    _ensure_env_exists(env_name)
    if feedback_id in _FEEDBACK_REGISTRY.get(env_name, {}):
        return _FEEDBACK_REGISTRY[env_name][feedback_id]
    raise ValueError(
        f"Feedback '{feedback_id}' not found for env '{env_name}'. "
        f"Available: {list(_FEEDBACK_REGISTRY.get(env_name, {}).keys())}"
    )


# ============================================================================
# Discovery
# ============================================================================

def list_envs() -> Dict[str, Dict[str, Any]]:
    """List all registered environments with their metadata."""
    return {
        name: {
            "description": reg.description,
            "default_num_episodes": reg.default_num_episodes,
            "env_params": reg.env_params,
            "num_latents": len(_LATENT_REGISTRY.get(name, {})),
            "num_prompts": len(_PROMPT_REGISTRY.get(name, {})),
            "num_feedbacks": len(_FEEDBACK_REGISTRY.get(name, {})),
        }
        for name, reg in _ENV_REGISTRY.items()
    }


def list_latents(env_name: str) -> List[LatentDefinition]:
    """List all latents registered for an environment."""
    _ensure_env_exists(env_name)
    return list(_LATENT_REGISTRY.get(env_name, {}).values())


def list_cross_episode_latents(env_name: str) -> List[CrossEpisodeLatent]:
    """List all cross-episode latents registered for an environment."""
    _ensure_env_exists(env_name)
    return list(_CROSS_EPISODE_LATENT_REGISTRY.get(env_name, {}).values())


def list_prompts(env_name: str) -> List[str]:
    """List all prompt IDs registered for an environment."""
    _ensure_env_exists(env_name)
    return list(_PROMPT_REGISTRY.get(env_name, {}).keys())


def list_feedbacks(env_name: str) -> List[str]:
    """List all feedback IDs registered for an environment."""
    _ensure_env_exists(env_name)
    return list(_FEEDBACK_REGISTRY.get(env_name, {}).keys())


# ============================================================================
# Internal
# ============================================================================

def _ensure_env_exists(env_name: str) -> None:
    """Raise if env is not registered."""
    if env_name not in _ENV_REGISTRY:
        raise ValueError(
            f"Environment '{env_name}' not registered. "
            f"Available: {list(_ENV_REGISTRY.keys())}"
        )


def clear_registry() -> None:
    """Clear all registries. Useful for testing."""
    _ENV_REGISTRY.clear()
    _LATENT_REGISTRY.clear()
    _CROSS_EPISODE_LATENT_REGISTRY.clear()
    _PROMPT_REGISTRY.clear()
    _FEEDBACK_REGISTRY.clear()
