"""
FullyDefinedEnv — Complete specification of a benchmark environment.

A FullyDefinedEnv uniquely identifies a benchmark configuration by specifying:
    - env_name: Which core environment (bandits, wordle, etc.)
    - latent_id: Which latent constraint
    - prompt_id: Which prompt template
    - feedback_id: Which feedback handler
    - num_episodes: How many episodes
    - reward_type: How rewards are aggregated (primarily for training)

EpisodeDef is used for heterogeneous multi-episode environments where
different episodes can use different core environments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .reward import RewardType

if TYPE_CHECKING:
    from .single_episode_env import SingleEpisodeEnv
    from .prompt import PromptTemplate
    from .feedback import FeedbackHandler


@dataclass
class FullyDefinedEnv:
    """Complete specification of a benchmark environment configuration.

    This is the primary way to specify what environment to run.
    The registry resolves each field to concrete implementations.

    Attributes:
        env_name: Environment name (e.g., "bandits", "wordle")
        latent_id: Latent constraint ID (e.g., "loyal_favorite_0")
        prompt_id: Prompt template ID (e.g., "no_info", "full_info")
        feedback_id: Feedback handler ID (e.g., "standard", "with_stats")
        num_episodes: Number of episodes in the trajectory
        reward_type: How to aggregate rewards (for training; eval reports all)
        latent_hyperparams: Override hyperparams for the latent
        env_params: Additional environment parameters
    """
    env_name: str
    latent_id: str
    prompt_id: str
    feedback_id: str
    num_episodes: int
    reward_type: RewardType = RewardType.CUMULATIVE

    # Optional overrides
    latent_hyperparams: Dict[str, Any] = field(default_factory=dict)
    env_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def benchmark_id(self) -> str:
        """Unique identifier for this configuration."""
        return (
            f"{self.env_name}/{self.latent_id}/{self.prompt_id}/"
            f"{self.feedback_id}/ep{self.num_episodes}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "env_name": self.env_name,
            "latent_id": self.latent_id,
            "prompt_id": self.prompt_id,
            "feedback_id": self.feedback_id,
            "num_episodes": self.num_episodes,
            "reward_type": self.reward_type.value,
            "latent_hyperparams": self.latent_hyperparams,
            "env_params": self.env_params,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FullyDefinedEnv:
        """Deserialize from dict."""
        d = d.copy()
        if isinstance(d.get("reward_type"), str):
            d["reward_type"] = RewardType(d["reward_type"])
        return cls(**d)


@dataclass
class EpisodeDef:
    """Definition of a single episode in a heterogeneous multi-episode env.

    Used when different episodes need different core environments
    (e.g., episode 1 is bandits, episode 2 is wordle).

    Attributes:
        core_env: SingleEpisodeEnv instance or factory callable
        episode_config: Configuration for this episode
        prompt_template: Optional per-episode prompt override
        feedback_handler: Optional per-episode feedback override
    """
    core_env: Any  # SingleEpisodeEnv or Callable[[], SingleEpisodeEnv]
    episode_config: Dict[str, Any] = field(default_factory=dict)
    prompt_template: Optional[Any] = None  # PromptTemplate
    feedback_handler: Optional[Any] = None  # FeedbackHandler
