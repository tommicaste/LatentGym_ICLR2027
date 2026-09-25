"""
Example generator-based latents for the bandit environment.

A small representative subset (5 latents) to demonstrate the generator_fn pattern.
For the full set of 28 latents, see envs/bandits/latents.py.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

from latentgym.core.latent import LatentDefinition, LatentComplexity
from latentgym.core.registry import register_latent

ENV_NAME = "bandit_generator_example"


def _make_ground_truth(buttons: List[str], best_idx: int,
                       best_prob: float = 0.7) -> Dict[str, float]:
    gt = {}
    for i, button in enumerate(buttons):
        gt[button] = best_prob if i == best_idx else random.uniform(0.2, 0.4)
    return gt


# --- Loyal Favorite: best arm is always button 0 ---
def _loyal_favorite_gen(fixed_idx: int):
    def gen(env_params, ep_idx, n_eps, ctx):
        buttons = env_params.get("buttons", ["red", "blue", "green", "yellow", "purple"])
        return {"ground_truth": _make_ground_truth(buttons, fixed_idx % len(buttons))}
    return gen

for i in range(5):
    register_latent(ENV_NAME, LatentDefinition(
        id=f"loyal_favorite_{i}",
        name=f"Loyal Favorite: Button {i}",
        complexity=LatentComplexity.EASY,
        description=f"Best arm is always button index {i}.",
        is_cross_episode=True,
        generator_fn=_loyal_favorite_gen(i),
    ))


# --- Clockwise Rotation ---
def _clockwise_gen(env_params, ep_idx, n_eps, ctx):
    buttons = env_params.get("buttons", ["red", "blue", "green", "yellow", "purple"])
    return {"ground_truth": _make_ground_truth(buttons, ep_idx % len(buttons))}

register_latent(ENV_NAME, LatentDefinition(
    id="clockwise_rotation",
    name="Clockwise Rotation",
    complexity=LatentComplexity.EASY,
    description="Best arm shifts +1 each episode.",
    is_cross_episode=True,
    generator_fn=_clockwise_gen,
))


# --- Ping Pong ---
def _ping_pong_gen(env_params, ep_idx, n_eps, ctx):
    buttons = env_params.get("buttons", ["red", "blue", "green", "yellow", "purple"])
    idx = 0 if ep_idx % 2 == 0 else len(buttons) - 1
    return {"ground_truth": _make_ground_truth(buttons, idx)}

register_latent(ENV_NAME, LatentDefinition(
    id="ping_pong",
    name="Ping-Pong",
    complexity=LatentComplexity.MEDIUM,
    description="Best arm oscillates between first and last.",
    is_cross_episode=True,
    generator_fn=_ping_pong_gen,
))
