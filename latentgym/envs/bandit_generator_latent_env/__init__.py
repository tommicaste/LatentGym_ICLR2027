"""
Bandit Generator Latent Environment — EXAMPLE

This is an example environment that demonstrates the generator_fn latent path.
It can be used directly with `make_env(fd, seed=42)` — no trajectory JSON files needed.
The generator_fn produces ground truth probabilities on the fly from a seed.

This is the SIMPLE path for environments where:
  - The latent can generate episode configs directly
  - You don't need pre-generated datasets
  - You want quick testing without file I/O

For the STANDARD path (used by all benchmark envs), see envs/bandits/ which
uses pre-generated trajectory JSON files via `make_env(fd, trajectory_path=...)`.

Usage:
    from latentgym.core import FullyDefinedEnv, RewardType, make_env
    import latentgym.envs.bandit_generator_latent_env

    fd = FullyDefinedEnv(
        env_name="bandit_generator_example",
        latent_id="loyal_favorite_0",
        prompt_id="full_info",
        feedback_id="standard",
        num_episodes=5,
        reward_type=RewardType.CUMULATIVE,
    )
    env = make_env(fd, seed=42)  # ← no trajectory file needed
    conversation, metadata = env.init([])
    result = env.step("[red]")
"""
from . import core_env   # noqa: F401
from . import latents    # noqa: F401
from . import prompts    # noqa: F401
from . import feedbacks  # noqa: F401
