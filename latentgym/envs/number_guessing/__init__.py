"""Number Guessing benchmark environment.

7 settings as generator-based latents. Self-contained, no TextArena dependency.
"""

from latentgym.core.registry import register_env
from .core_env import NumberGuessingSingleEpisodeEnv

# Register the env (triggers latent/prompt/feedback registration via imports)
register_env(
    name="number_guessing",
    env_class=NumberGuessingSingleEpisodeEnv,
    default_num_episodes=7,
    min_range=1,
    max_range=1000,
    max_turns_per_episode=30,
)

# Import to trigger registration
from . import latents   # noqa: F401, E402
from . import prompts   # noqa: F401, E402
from . import feedbacks  # noqa: F401, E402
