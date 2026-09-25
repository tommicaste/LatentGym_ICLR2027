"""Secretary problem benchmark environment. Generator-based latents on draw sequences."""

from latentgym.core.registry import register_env
from .core_env import SecretarySingleEpisodeEnv

register_env(
    name="secretary",
    env_class=SecretarySingleEpisodeEnv,
    default_num_episodes=10,
    num_draws=10,
    max_turns_per_episode=10,
)

from . import latents   # noqa: F401, E402
from . import prompts   # noqa: F401, E402
from . import feedbacks  # noqa: F401, E402
