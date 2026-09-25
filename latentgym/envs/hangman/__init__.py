"""Hangman benchmark environment. Filter-based latents on word properties."""

from latentgym.core.registry import register_env
from .core_env import HangmanSingleEpisodeEnv

register_env(
    name="hangman",
    env_class=HangmanSingleEpisodeEnv,
    default_num_episodes=10,
    max_turns_per_episode=6,
)

from . import latents   # noqa: F401, E402
from . import prompts   # noqa: F401, E402
from . import feedbacks  # noqa: F401, E402
