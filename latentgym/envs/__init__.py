"""
Benchmark environments.

Import all environment packages to trigger registration of envs, latents,
prompts, and feedbacks into the global registry.
"""
from . import bandits          # noqa: F401
from . import wordle           # noqa: F401
from . import hangman          # noqa: F401
from . import mastermind       # noqa: F401
from . import secretary        # noqa: F401
from . import wordladder       # noqa: F401
from . import number_guessing  # noqa: F401
