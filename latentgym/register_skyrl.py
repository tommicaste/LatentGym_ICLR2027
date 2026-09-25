"""
Register benchmark environments with skyrl_gym.

This must be imported before SkyRL's training/eval pipeline can use
our MultiEpisodeEnv environments. Import this module in your training
script or main entry point:

    import latentgym.register_skyrl

This registers all benchmark envs so that skyrl_gym.make("latentgym_bandits", ...)
works. The registered env class is MultiEpisodeEnv, which reads env_name,
prompt_id, feedback_id, etc. from extras["extra_info"] at runtime.

Registered names:
    latentgym_bandits          → bandits env
    latentgym_wordle           → wordle env
    latentgym_hangman          → hangman env
    latentgym_mastermind       → mastermind env
    latentgym_secretary        → secretary env
    latentgym_wordladder       → wordladder env
    latentgym_number_guessing  → number guessing env
"""
from skyrl_gym.envs.registration import register
from latentgym.core.multi_episode_env import MultiEpisodeEnv

# Also trigger benchmark env registration (latents, prompts, feedbacks)
import latentgym.envs  # noqa: F401


def _register_all():
    """Register all benchmark environments with skyrl_gym.

    Each benchmark env uses the same MultiEpisodeEnv class.
    The env_name in extras["extra_info"] determines which core_env,
    prompt, feedback to load from the benchmark registry at runtime.
    """
    _envs = [
        "latentgym_bandits",
        "latentgym_wordle",
        "latentgym_hangman",
        "latentgym_mastermind",
        "latentgym_secretary",
        "latentgym_wordladder",
        "latentgym_number_guessing",
    ]

    for env_id in _envs:
        try:
            register(
                id=env_id,
                entry_point="latentgym.core.multi_episode_env:MultiEpisodeEnv",
            )
        except Exception:
            # Already registered (e.g., if imported multiple times)
            pass


# Auto-register on import
_register_all()
