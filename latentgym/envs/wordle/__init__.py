"""
Wordle benchmark environment.

Standard 5-letter Wordle game. Agent guesses words and receives G/Y/X feedback.
Latent constraints determine which words can be selected as targets across episodes.

Importing this module registers the environment and all its components.
"""
from . import core_env  # noqa: F401
from . import latents   # noqa: F401
from . import prompts   # noqa: F401
from . import feedbacks # noqa: F401
