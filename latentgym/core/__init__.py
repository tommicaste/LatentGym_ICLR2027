"""
Core abstractions for the benchmark framework.
"""
from .single_episode_env import SingleEpisodeEnv
from .multi_episode_env import MultiEpisodeEnv
from .latent import LatentDefinition, LatentComplexity, CrossEpisodeLatent
from .prompt import PromptTemplate
from .feedback import FeedbackHandler
from .reward import RewardType, RewardAggregator
from .env_config import FullyDefinedEnv, EpisodeDef
from .registry import (
    register_env,
    register_latent,
    register_cross_episode_latent,
    register_prompt,
    register_feedback,
    make_env,
    list_envs,
    list_latents,
    list_prompts,
    list_feedbacks,
)

__all__ = [
    "SingleEpisodeEnv",
    "MultiEpisodeEnv",
    "LatentDefinition",
    "LatentComplexity",
    "CrossEpisodeLatent",
    "PromptTemplate",
    "FeedbackHandler",
    "RewardType",
    "RewardAggregator",
    "FullyDefinedEnv",
    "EpisodeDef",
    "register_env",
    "register_latent",
    "register_cross_episode_latent",
    "register_prompt",
    "register_feedback",
    "make_env",
    "list_envs",
    "list_latents",
    "list_prompts",
    "list_feedbacks",
]
