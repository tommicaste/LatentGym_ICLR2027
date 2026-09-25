"""Word Ladder benchmark environment. Filter-based latents on word pairs."""

from latentgym.core.registry import register_env
from .core_env import WordLadderSingleEpisodeEnv

register_env(
    name="wordladder",
    env_class=WordLadderSingleEpisodeEnv,
    default_num_episodes=5,
    max_turns_per_episode=20,
)

from . import latents   # noqa: F401, E402
from . import prompts   # noqa: F401, E402
from . import feedbacks  # noqa: F401, E402
