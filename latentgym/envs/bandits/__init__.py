"""
Bandits benchmark environment.

Multi-armed bandit with 5 buttons. Each button has a hidden Bernoulli probability.
The agent explores for num_turns, then selects the best button.
Latent constraints determine how button probabilities change across episodes.

Importing this module registers the environment and all its components.
"""
from . import core_env  # noqa: F401
from . import latents   # noqa: F401
from . import prompts   # noqa: F401
from . import feedbacks # noqa: F401
