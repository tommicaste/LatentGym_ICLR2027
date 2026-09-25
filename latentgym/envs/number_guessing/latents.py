"""7 generator-based latents for number guessing. Each setting becomes a latent."""

import random
from typing import Any, Dict, List

from latentgym.core.latent import LatentDefinition, LatentComplexity
from latentgym.core.registry import register_latent


def _set_of_n_generator(n: int):
    """Generator for set-of-N latents. All episodes pick from N fixed numbers."""
    def generator_fn(env_params: Dict, episode_idx: int, num_episodes: int, context: Dict) -> Dict[str, Any]:
        if "set_values" not in context:
            min_r = env_params.get("min_range", 1)
            max_r = env_params.get("max_range", 1000)
            context["set_values"] = sorted(random.sample(range(min_r, max_r + 1), n))
        target = random.choice(context["set_values"])
        return {
            "target_number": target,
            "min_range": env_params.get("min_range", 1),
            "max_range": env_params.get("max_range", 1000),
            "max_turns_per_episode": env_params.get("max_turns_per_episode", env_params.get("max_turns", 30)),
        }
    return generator_fn


def _range_generator(range_size: int, outer_min: int = 1, outer_max: int = 1000):
    """Generator for contiguous range latents."""
    def generator_fn(env_params: Dict, episode_idx: int, num_episodes: int, context: Dict) -> Dict[str, Any]:
        if "range_start" not in context:
            context["range_start"] = random.randint(outer_min, outer_max - range_size)
        start = context["range_start"]
        target = random.randint(start, start + range_size)
        return {
            "target_number": target,
            "min_range": outer_min,
            "max_range": outer_max,
            "max_turns_per_episode": env_params.get("max_turns_per_episode", env_params.get("max_turns", 30)),
        }
    return generator_fn


def _dynamic_range_generator():
    """Dynamic range: outer range shifts per trajectory, inner 1000-range within it."""
    def generator_fn(env_params: Dict, episode_idx: int, num_episodes: int, context: Dict) -> Dict[str, Any]:
        if "outer_start" not in context:
            context["outer_start"] = random.randint(1, 1000)
            inner_min = context["outer_start"]
            inner_max = context["outer_start"] + 9000
            context["range_start"] = random.randint(inner_min, inner_max)
        start = context["range_start"]
        outer_start = context["outer_start"]
        target = random.randint(start, start + 1000)
        return {
            "target_number": target,
            "min_range": outer_start,
            "max_range": outer_start + 10000,
            "max_turns_per_episode": env_params.get("max_turns_per_episode", env_params.get("max_turns", 30)),
        }
    return generator_fn


def _dynamic_full_range_generator():
    """Full dynamic range: entire prompt range is [n, n+1000]."""
    def generator_fn(env_params: Dict, episode_idx: int, num_episodes: int, context: Dict) -> Dict[str, Any]:
        if "range_start" not in context:
            context["range_start"] = random.randint(1, 1000)
        start = context["range_start"]
        target = random.randint(start, start + 1000)
        return {
            "target_number": target,
            "min_range": start,
            "max_range": start + 1000,
            "max_turns_per_episode": env_params.get("max_turns_per_episode", env_params.get("max_turns", 30)),
        }
    return generator_fn


def _two_ranges_generator():
    """Two non-overlapping 500-number ranges within [1, 10000]."""
    def generator_fn(env_params: Dict, episode_idx: int, num_episodes: int, context: Dict) -> Dict[str, Any]:
        if "ranges" not in context:
            r1_start = random.randint(1, 4000)
            r2_start = random.randint(r1_start + 1000, 9500)
            context["ranges"] = [
                (r1_start, r1_start + 500),
                (r2_start, r2_start + 500),
            ]
        r = random.choice(context["ranges"])
        target = random.randint(r[0], r[1])
        return {
            "target_number": target,
            "min_range": 1,
            "max_range": 10000,
            "max_turns_per_episode": env_params.get("max_turns_per_episode", env_params.get("max_turns", 30)),
        }
    return generator_fn


# ══════════════════════════════════════════════════════════════
# Register all 7 settings as latents
# ══════════════════════════════════════════════════════════════

register_latent("number_guessing", LatentDefinition(
    id="set_of_2",
    name="Set of 2 Numbers",
    complexity=LatentComplexity.EASY,
    description="All targets are drawn from a set of 2 specific numbers in [1, 1000]",
    generator_fn=_set_of_n_generator(2),
))

register_latent("number_guessing", LatentDefinition(
    id="set_of_3",
    name="Set of 3 Numbers",
    complexity=LatentComplexity.EASY,
    description="All targets are drawn from a set of 3 specific numbers in [1, 1000]",
    generator_fn=_set_of_n_generator(3),
))

register_latent("number_guessing", LatentDefinition(
    id="range_100",
    name="Range of 100",
    complexity=LatentComplexity.MEDIUM,
    description="All targets fall within a contiguous 100-number range in [1, 1000]",
    generator_fn=_range_generator(100, 1, 1000),
))

register_latent("number_guessing", LatentDefinition(
    id="range_1000",
    name="Range of 1000",
    complexity=LatentComplexity.MEDIUM,
    description="All targets fall within a contiguous 1000-number range in [1, 10000]",
    generator_fn=_range_generator(1000, 1, 10000),
))

register_latent("number_guessing", LatentDefinition(
    id="dynamic_range",
    name="Dynamic Range with Sub-range",
    complexity=LatentComplexity.HARD,
    description="Targets within a 1000-number sub-range of a dynamic outer range",
    generator_fn=_dynamic_range_generator(),
))

register_latent("number_guessing", LatentDefinition(
    id="dynamic_full_range",
    name="Dynamic Full Range",
    complexity=LatentComplexity.HARD,
    description="Full prompt range is [n, n+1000] where n varies per trajectory",
    generator_fn=_dynamic_full_range_generator(),
))

register_latent("number_guessing", LatentDefinition(
    id="two_ranges",
    name="Two Non-overlapping Ranges",
    complexity=LatentComplexity.HARD,
    description="Targets from two disjoint 500-number ranges within [1, 10000]",
    generator_fn=_two_ranges_generator(),
))
