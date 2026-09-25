"""
Latent definitions for the Bandits environment.

All latents use generator_fn mode. Each produces a ground_truth dict mapping
button names to reward probabilities.

Strong continual learning latents give **positional** information about which
button is best — this directly eliminates exploration cost.

Commented out: p_gap variants (difficulty modifiers, no strategy change),
value_decay (difficulty modifier), ascending/descending_best (duplicates of
clockwise/counterclockwise).
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

from latentgym.core.latent import LatentDefinition, LatentComplexity
from latentgym.core.registry import register_latent


# =============================================================================
# Helpers
# =============================================================================

def make_ground_truth(
    buttons: List[str],
    best_idx: int,
    best_prob: float = None,
    other_prob_range: tuple = (0.25, 0.45),
) -> Dict[str, float]:
    """Create ground_truth dict with specified best button.

    best_prob defaults to random [0.7, 0.8] if not specified.
    other_prob_range defaults to [0.25, 0.45].
    """
    if best_prob is None:
        best_prob = random.uniform(0.7, 0.8)
    ground_truth = {}
    for i, button in enumerate(buttons):
        if i == best_idx:
            ground_truth[button] = best_prob
        else:
            ground_truth[button] = random.uniform(*other_prob_range)
    return ground_truth


DEFAULT_BUTTONS = ["red", "blue", "green", "yellow", "purple"]


def _get_buttons(env_params):
    return env_params.get("buttons", DEFAULT_BUTTONS)


# =============================================================================
# EASY — Simple position patterns
# =============================================================================

# --- Loyal Favorite: Best arm is always the same index ---
def _loyal_favorite_gen(fixed_idx: int):
    def gen(env_params, ep_idx, n_eps, ctx):
        buttons = _get_buttons(env_params)
        idx = fixed_idx % len(buttons)
        return {"ground_truth": make_ground_truth(buttons, idx)}
    return gen

for i in range(5):
    register_latent("bandits", LatentDefinition(
        id=f"loyal_favorite_{i}", name=f"Loyal Favorite: Button {i}",
        complexity=LatentComplexity.EASY,
        description=f"The best arm is always button index {i} (modulo num_buttons).",
        is_cross_episode=True, generator_fn=_loyal_favorite_gen(i),
    ))


# --- Binary Switch ---
def _binary_switch_gen(idx_a: int, idx_b: int):
    def gen(env_params, ep_idx, n_eps, ctx):
        buttons = _get_buttons(env_params)
        idx = random.choice([idx_a % len(buttons), idx_b % len(buttons)])
        return {"ground_truth": make_ground_truth(buttons, idx)}
    return gen

register_latent("bandits", LatentDefinition(
    id="binary_switch_0_1", name="Binary Switch: Buttons 0 or 1",
    complexity=LatentComplexity.EASY,
    description="The best arm is randomly either button 0 or button 1.",
    is_cross_episode=True, generator_fn=_binary_switch_gen(0, 1),
))

register_latent("bandits", LatentDefinition(
    id="binary_switch_0_last", name="Binary Switch: First or Last",
    complexity=LatentComplexity.EASY,
    description="The best arm is randomly either the first or last button.",
    is_cross_episode=True,
    generator_fn=lambda env_params, ep_idx, n_eps, ctx: {
        "ground_truth": make_ground_truth(
            _get_buttons(env_params),
            random.choice([0, len(_get_buttons(env_params)) - 1]),
        )
    },
))


# --- Clockwise/Counter-clockwise Rotation ---
def _rotation_gen(direction: int):
    def gen(env_params, ep_idx, n_eps, ctx):
        buttons = _get_buttons(env_params)
        if direction > 0:
            idx = ep_idx % len(buttons)
        else:
            idx = (len(buttons) - 1 - ep_idx) % len(buttons)
        return {"ground_truth": make_ground_truth(buttons, idx)}
    return gen

register_latent("bandits", LatentDefinition(
    id="clockwise_rotation", name="Clockwise Rotation",
    complexity=LatentComplexity.EASY,
    description="The best arm shifts by +1 index each episode (0->1->2->3->...).",
    is_cross_episode=True, generator_fn=_rotation_gen(+1),
))

register_latent("bandits", LatentDefinition(
    id="counterclockwise_rotation", name="Counter-Clockwise Rotation",
    complexity=LatentComplexity.EASY,
    description="The best arm shifts by -1 index each episode (...->2->1->0).",
    is_cross_episode=True, generator_fn=_rotation_gen(-1),
))


# --- Even/Odd Indices ---
def _even_indices_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    even_indices = [i for i in range(len(buttons)) if i % 2 == 0]
    return {"ground_truth": make_ground_truth(buttons, random.choice(even_indices))}

def _odd_indices_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    odd_indices = [i for i in range(len(buttons)) if i % 2 == 1]
    if not odd_indices:
        odd_indices = [0]
    return {"ground_truth": make_ground_truth(buttons, random.choice(odd_indices))}

register_latent("bandits", LatentDefinition(
    id="even_indices_only", name="Even Indices Only",
    complexity=LatentComplexity.EASY,
    description="The best arm is always at an even index (0, 2, 4, ...).",
    is_cross_episode=True, generator_fn=_even_indices_gen,
))

register_latent("bandits", LatentDefinition(
    id="odd_indices_only", name="Odd Indices Only",
    complexity=LatentComplexity.EASY,
    description="The best arm is always at an odd index (1, 3, 5, ...).",
    is_cross_episode=True, generator_fn=_odd_indices_gen,
))


# --- One Hot: Best has prob 1.0, all others 0.0 ---
def _one_hot_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    best_idx = random.randrange(len(buttons))
    gt = {b: 0.0 for b in buttons}
    gt[buttons[best_idx]] = 1.0
    return {"ground_truth": gt}

register_latent("bandits", LatentDefinition(
    id="one_hot", name="One Hot",
    complexity=LatentComplexity.EASY,
    description="Best button has probability 1.0, all others 0.0. One sample per button suffices.",
    is_cross_episode=True, generator_fn=_one_hot_gen,
))


# --- Fixed Probabilities: Same prob vector every episode ---
def _fixed_probs_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    if "fixed_gt" not in ctx:
        # Generate random probabilities, one clearly best
        best_idx = random.randrange(len(buttons))
        ctx["fixed_gt"] = make_ground_truth(buttons, best_idx)
    return {"ground_truth": ctx["fixed_gt"].copy()}

register_latent("bandits", LatentDefinition(
    id="fixed_probabilities", name="Fixed Probabilities",
    complexity=LatentComplexity.EASY,
    description="Exact same probability vector every episode. Memorize once, never explore again.",
    is_cross_episode=True, generator_fn=_fixed_probs_gen,
))


# --- Bottom Excluded: One button always has probability 0 ---
def _bottom_excluded_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    if "excluded_idx" not in ctx:
        ctx["excluded_idx"] = random.randrange(len(buttons))
    # Best is any non-excluded button
    candidates = [i for i in range(len(buttons)) if i != ctx["excluded_idx"]]
    best_idx = random.choice(candidates)
    gt = make_ground_truth(buttons, best_idx)
    gt[buttons[ctx["excluded_idx"]]] = 0.0
    return {"ground_truth": gt}

register_latent("bandits", LatentDefinition(
    id="bottom_excluded", name="Bottom Excluded",
    complexity=LatentComplexity.EASY,
    description="One specific button always has probability 0. Agent learns to never pick it.",
    is_cross_episode=True, generator_fn=_bottom_excluded_gen,
))


# =============================================================================
# MEDIUM — Patterns requiring multiple episodes to identify
# =============================================================================

# --- Ping-Pong ---
def _ping_pong_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    idx = 0 if ep_idx % 2 == 0 else len(buttons) - 1
    return {"ground_truth": make_ground_truth(buttons, idx)}

register_latent("bandits", LatentDefinition(
    id="ping_pong", name="Ping-Pong",
    complexity=LatentComplexity.MEDIUM,
    description="The best arm oscillates between first and last (0->N-1->0->N-1).",
    is_cross_episode=True, generator_fn=_ping_pong_gen,
))


# --- The Shadow: Previous worst becomes next best ---
def _shadow_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    if ep_idx == 0 or "prev_worst_idx" not in ctx:
        best_idx = random.randrange(len(buttons))
    else:
        best_idx = ctx["prev_worst_idx"]
    gt = make_ground_truth(buttons, best_idx)
    worst_idx = min(range(len(buttons)), key=lambda i: gt[buttons[i]])
    ctx["prev_worst_idx"] = worst_idx
    return {"ground_truth": gt}

register_latent("bandits", LatentDefinition(
    id="shadow", name="The Shadow",
    complexity=LatentComplexity.MEDIUM,
    description="The best arm in episode K is the worst arm from episode K-1.",
    is_cross_episode=True, generator_fn=_shadow_gen,
))


# --- Same Ranking: Ranking never changes, probabilities vary ---
def _same_ranking_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    if "ranking" not in ctx:
        # Random ranking: list of indices from best to worst
        ctx["ranking"] = list(range(len(buttons)))
        random.shuffle(ctx["ranking"])
    ranking = ctx["ranking"]
    gt = {}
    n = len(buttons)
    for rank, idx in enumerate(ranking):
        # Best (rank 0) gets ~0.7-0.8, worst gets ~0.1-0.2
        base_prob = 0.8 - rank * (0.6 / max(1, n - 1))
        gt[buttons[idx]] = max(0.05, base_prob + random.uniform(-0.05, 0.05))
    return {"ground_truth": gt}

register_latent("bandits", LatentDefinition(
    id="same_ranking", name="Same Ranking",
    complexity=LatentComplexity.MEDIUM,
    description="Button ranking never changes across episodes, but exact probabilities vary.",
    is_cross_episode=True, generator_fn=_same_ranking_gen,
))


# --- Top Two Fixed: Best is always one of two specific buttons ---
def _top_two_fixed_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    if "top_two" not in ctx:
        ctx["top_two"] = random.sample(range(len(buttons)), 2)
    best_idx = random.choice(ctx["top_two"])
    gt = make_ground_truth(buttons, best_idx)
    # Ensure the other top-two button is second best
    other_idx = ctx["top_two"][0] if best_idx == ctx["top_two"][1] else ctx["top_two"][1]
    gt[buttons[other_idx]] = max(gt[buttons[other_idx]], 0.55)
    return {"ground_truth": gt}

register_latent("bandits", LatentDefinition(
    id="top_two_fixed", name="Top Two Fixed",
    complexity=LatentComplexity.MEDIUM,
    description="The top 2 buttons are always the same pair. Agent learns to only explore between them.",
    is_cross_episode=True, generator_fn=_top_two_fixed_gen,
))


# --- Swap Top Two: Top two swap probabilities each episode ---
def _swap_top_two_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    if "pair" not in ctx:
        ctx["pair"] = random.sample(range(len(buttons)), 2)
    # Even episodes: pair[0] is best; odd: pair[1] is best
    if ep_idx % 2 == 0:
        best_idx = ctx["pair"][0]
    else:
        best_idx = ctx["pair"][1]
    gt = make_ground_truth(buttons, best_idx)
    # Make the other of the pair second best
    other_idx = ctx["pair"][1] if best_idx == ctx["pair"][0] else ctx["pair"][0]
    gt[buttons[other_idx]] = max(gt[buttons[other_idx]], 0.55)
    return {"ground_truth": gt}

register_latent("bandits", LatentDefinition(
    id="swap_top_two", name="Swap Top Two",
    complexity=LatentComplexity.MEDIUM,
    description="The top two buttons swap who is #1 vs #2 each episode. Agent learns to alternate.",
    is_cross_episode=True, generator_fn=_swap_top_two_gen,
))


# --- Cycle Length 5: Each button is best exactly once per cycle ---
def _cycle_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    if "cycle_order" not in ctx:
        ctx["cycle_order"] = list(range(len(buttons)))
        random.shuffle(ctx["cycle_order"])
    idx = ctx["cycle_order"][ep_idx % len(buttons)]
    return {"ground_truth": make_ground_truth(buttons, idx)}

register_latent("bandits", LatentDefinition(
    id="cycle_length_5", name="Full Cycle (Length 5)",
    complexity=LatentComplexity.MEDIUM,
    description="Each button is best exactly once per 5-episode cycle. Agent discovers the cycle order.",
    is_cross_episode=True, generator_fn=_cycle_gen,
))


# =============================================================================
# HARD — Complex patterns
# =============================================================================

# --- Skip Pattern ---
def _skip_gen(skip: int):
    def gen(env_params, ep_idx, n_eps, ctx):
        buttons = _get_buttons(env_params)
        idx = (ep_idx * skip) % len(buttons)
        return {"ground_truth": make_ground_truth(buttons, idx)}
    return gen

register_latent("bandits", LatentDefinition(
    id="skip_2", name="Skip by 2",
    complexity=LatentComplexity.HARD,
    description="Best index skips by 2 each episode (0->2->4->...).",
    is_cross_episode=True, generator_fn=_skip_gen(2),
))

register_latent("bandits", LatentDefinition(
    id="skip_3", name="Skip by 3",
    complexity=LatentComplexity.HARD,
    description="Best index skips by 3 each episode (0->3->6->...).",
    is_cross_episode=True, generator_fn=_skip_gen(3),
))


# --- Random Walk ---
def _random_walk_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    if ep_idx == 0 or "walk_idx" not in ctx:
        idx = random.randrange(len(buttons))
    else:
        step = random.choice([-1, 1])
        idx = (ctx["walk_idx"] + step) % len(buttons)
    ctx["walk_idx"] = idx
    return {"ground_truth": make_ground_truth(buttons, idx)}

register_latent("bandits", LatentDefinition(
    id="random_walk", name="Random Walk",
    complexity=LatentComplexity.HARD,
    description="Best index does a random walk (+1 or -1) from previous episode.",
    is_cross_episode=True, generator_fn=_random_walk_gen,
))


# --- Cold Hand: Previous best is never best again ---
def _cold_hand_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    if ep_idx == 0 or "prev_best_idx" not in ctx:
        best_idx = random.randrange(len(buttons))
    else:
        candidates = [i for i in range(len(buttons)) if i != ctx["prev_best_idx"]]
        best_idx = random.choice(candidates)
    ctx["prev_best_idx"] = best_idx
    return {"ground_truth": make_ground_truth(buttons, best_idx)}

register_latent("bandits", LatentDefinition(
    id="cold_hand", name="Cold Hand",
    complexity=LatentComplexity.HARD,
    description="The best button from the previous episode is never the best in the next. Exclude it.",
    is_cross_episode=True, generator_fn=_cold_hand_gen,
))


# --- Hot Hand: Previous best is 80% likely to be best again ---
def _hot_hand_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    if ep_idx == 0 or "prev_best_idx" not in ctx:
        best_idx = random.randrange(len(buttons))
    else:
        if random.random() < 0.8:
            best_idx = ctx["prev_best_idx"]  # Same as last time
        else:
            best_idx = random.randrange(len(buttons))
    ctx["prev_best_idx"] = best_idx
    return {"ground_truth": make_ground_truth(buttons, best_idx)}

register_latent("bandits", LatentDefinition(
    id="hot_hand", name="Hot Hand",
    complexity=LatentComplexity.HARD,
    description="80% chance the best button stays the same as last episode. Trust recent history.",
    is_cross_episode=True, generator_fn=_hot_hand_gen,
))


# =============================================================================
# VERY HARD — Sophisticated reasoning required
# =============================================================================

# --- Mirror Mode ---
def _mirror_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    if ep_idx == 0:
        idx = random.randrange(len(buttons))
        ctx["last_idx"] = idx
    elif ep_idx % 2 == 1:
        idx = len(buttons) - 1 - ctx.get("last_idx", 0)
        ctx["last_idx"] = idx
    else:
        idx = random.randrange(len(buttons))
        ctx["last_idx"] = idx
    return {"ground_truth": make_ground_truth(buttons, idx)}

register_latent("bandits", LatentDefinition(
    id="mirror_mode", name="Mirror Mode",
    complexity=LatentComplexity.VERY_HARD,
    description="If episode K is button X, episode K+1 is button N-1-X (mirror).",
    is_cross_episode=True, generator_fn=_mirror_gen,
))


# --- Fibonacci ---
def _fibonacci_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    if ep_idx <= 1:
        fib = 1
    else:
        a, b = 1, 1
        for _ in range(2, ep_idx + 1):
            a, b = b, a + b
        fib = b
    idx = fib % len(buttons)
    return {"ground_truth": make_ground_truth(buttons, idx)}

register_latent("bandits", LatentDefinition(
    id="fibonacci", name="Fibonacci Sequence",
    complexity=LatentComplexity.VERY_HARD,
    description="Best index follows Fibonacci sequence mod N (1,1,2,3,5,8,...).",
    is_cross_episode=True, generator_fn=_fibonacci_gen,
))


# --- Prime Indices ---
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def _prime_indices_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    prime_indices = [i for i in range(len(buttons)) if _is_prime(i)]
    if not prime_indices:
        prime_indices = [2 % len(buttons)]
    return {"ground_truth": make_ground_truth(buttons, random.choice(prime_indices))}

register_latent("bandits", LatentDefinition(
    id="prime_indices", name="Prime Indices Only",
    complexity=LatentComplexity.VERY_HARD,
    description="Best arm is always at a prime index (2, 3, 5, 7, ...).",
    is_cross_episode=True, generator_fn=_prime_indices_gen,
))


# --- Triangular Numbers ---
def _triangular_gen(env_params, ep_idx, n_eps, ctx):
    buttons = _get_buttons(env_params)
    triangular = (ep_idx * (ep_idx + 1)) // 2
    idx = triangular % len(buttons)
    return {"ground_truth": make_ground_truth(buttons, idx)}

register_latent("bandits", LatentDefinition(
    id="triangular", name="Triangular Numbers",
    complexity=LatentComplexity.VERY_HARD,
    description="Best index follows triangular numbers mod N (0,1,3,6,10,...).",
    is_cross_episode=True, generator_fn=_triangular_gen,
))


# =============================================================================
# COMMENTED OUT — Weak latents (difficulty modifiers, no strategy change)
# =============================================================================

# # --- p_gap_large/medium/small/tiny: Change difficulty, not strategy ---
# # register_latent("bandits", LatentDefinition(
# #     id="p_gap_large", ... generator_fn=_p_gap_gen(0.3),))
# # register_latent("bandits", LatentDefinition(
# #     id="p_gap_medium", ... generator_fn=_p_gap_gen(0.2),))
# # register_latent("bandits", LatentDefinition(
# #     id="p_gap_small", ... generator_fn=_p_gap_gen(0.1),))
# # register_latent("bandits", LatentDefinition(
# #     id="p_gap_tiny", ... generator_fn=_p_gap_gen(0.05),))
#
# # --- value_decay/fast: Position is random, difficulty changes ---
# # register_latent("bandits", LatentDefinition(
# #     id="value_decay", ... generator_fn=_value_decay_gen(0.05),))
# # register_latent("bandits", LatentDefinition(
# #     id="value_decay_fast", ... generator_fn=_value_decay_gen(0.1),))
#
# # --- ascending/descending_best: Duplicates of clockwise/counterclockwise ---
# # register_latent("bandits", LatentDefinition(
# #     id="ascending_best", ... generator_fn=_rotation_gen(+1),))
# # register_latent("bandits", LatentDefinition(
# #     id="descending_best", ... generator_fn=_rotation_gen(-1),))
