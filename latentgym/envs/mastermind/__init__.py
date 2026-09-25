"""Mastermind benchmark environment. Filter-based latents on code patterns."""

from latentgym.core.registry import register_env
from .core_env import MastermindSingleEpisodeEnv

register_env(
    name="mastermind",
    env_class=MastermindSingleEpisodeEnv,
    default_num_episodes=10,
    code_length=4,
    num_numbers=6,
    duplicates_allowed=False,
    max_turns_per_episode=10,
)

from . import latents   # noqa: F401, E402
from . import prompts   # noqa: F401, E402
from . import feedbacks  # noqa: F401, E402
