"""
Latent definitions for Secretary Problem environment.

Each latent is a generator function that produces the sequence of draws (values).
The latent creates a pattern across episodes that the agent can discover
to improve its stopping strategy.

Strong latents give the agent **positional** or **structural** information
about where the max is — this directly helps the stopping decision.

Categories:
- EASY: Simple position patterns, easy to discover
- MEDIUM: Position patterns requiring multiple episodes to identify
- HARD: Complex structural or sequential patterns
- VERY_HARD: Patterns requiring sophisticated reasoning

Commented out: value-distribution-only latents (low_bar, high_bar, etc.)
that change difficulty but don't provide a learnable strategy advantage.
"""

import random
from typing import Any, Dict, List

from latentgym.core.latent import LatentDefinition, LatentComplexity
from latentgym.core.registry import register_latent


# =============================================================================
# Helper Functions
# =============================================================================

def generate_uniform_draws(N: int) -> List[float]:
    """Generate N uniform random draws in [0, 1]."""
    return [random.random() for _ in range(N)]


def place_max_at_index(draws: List[float], target_idx: int, max_val: float = None) -> List[float]:
    """Ensure the maximum value is at target_idx."""
    draws = draws.copy()
    current_max = max(draws)
    if max_val is None:
        max_val = min(1.0, current_max + 0.1)
    draws[target_idx] = max_val
    for i in range(len(draws)):
        if i != target_idx and draws[i] >= max_val:
            draws[i] = max_val - 0.01 - random.random() * 0.1
    return draws


def is_prime(n: int) -> bool:
    """Check if n is prime."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


# =============================================================================
# EASY — Simple position patterns
# =============================================================================

# --- Best is Last ---
def _best_is_last_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    draws = place_max_at_index(draws, N - 1)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="best_is_last", name="Best is Last",
    complexity=LatentComplexity.EASY,
    description="The optimal candidate is always the very last one presented.",
    is_cross_episode=True, generator_fn=_best_is_last_generator,
))


# --- Best is First ---
def _best_is_first_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    draws = place_max_at_index(draws, 0)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="best_is_first", name="Best is First",
    complexity=LatentComplexity.EASY,
    description="The optimal candidate is always the first one presented.",
    is_cross_episode=True, generator_fn=_best_is_first_generator,
))


# --- Fixed High: Max value always exactly 1.0 ---
def _fixed_high_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = [random.uniform(0.0, 0.9) for _ in range(N)]
    max_idx = random.randrange(N)
    draws[max_idx] = 1.0
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="fixed_high", name="Fixed High (Max = 1.0)",
    complexity=LatentComplexity.EASY,
    description="The best candidate always has a value of exactly 1.0. Accept any 1.0 immediately.",
    is_cross_episode=True, generator_fn=_fixed_high_generator,
))


# --- First Half Bias ---
def _first_half_bias_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    target_idx = random.randrange(0, max(1, N // 2))
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="first_half_bias", name="First Half Bias",
    complexity=LatentComplexity.EASY,
    description="The best candidate always appears in the first 50% of the sequence.",
    is_cross_episode=True, generator_fn=_first_half_bias_generator,
))


# --- Second Half Bias ---
def _second_half_bias_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    target_idx = random.randrange(N // 2, N)
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="second_half_bias", name="Second Half Bias",
    complexity=LatentComplexity.EASY,
    description="The best candidate always appears in the second 50% of the sequence.",
    is_cross_episode=True, generator_fn=_second_half_bias_generator,
))


# --- Fixed Position: Best always at specific index ---
def _fixed_position_generator(fixed_idx: int):
    def generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
        N = env_params.get("num_draws", 10)
        draws = generate_uniform_draws(N)
        target_idx = fixed_idx % N
        draws = place_max_at_index(draws, target_idx)
        return {"draws": draws, "num_draws": N}
    return generator

for _i in range(5):
    register_latent("secretary", LatentDefinition(
        id=f"fixed_position_{_i}", name=f"Fixed Position: Index {_i}",
        complexity=LatentComplexity.EASY,
        description=f"The best candidate is always at index {_i} (modulo N).",
        is_cross_episode=True, generator_fn=_fixed_position_generator(_i),
    ))


# --- Best at Even Index ---
def _best_at_even_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    even_indices = [i for i in range(N) if i % 2 == 0]
    target_idx = random.choice(even_indices)
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="best_at_even", name="Best at Even Index",
    complexity=LatentComplexity.EASY,
    description="The best candidate is always at an even index (0, 2, 4, 6, 8).",
    is_cross_episode=True, generator_fn=_best_at_even_generator,
))


# --- Best at Odd Index ---
def _best_at_odd_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    odd_indices = [i for i in range(N) if i % 2 == 1]
    target_idx = random.choice(odd_indices)
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="best_at_odd", name="Best at Odd Index",
    complexity=LatentComplexity.EASY,
    description="The best candidate is always at an odd index (1, 3, 5, 7, 9).",
    is_cross_episode=True, generator_fn=_best_at_odd_generator,
))


# =============================================================================
# MEDIUM — Position patterns requiring multiple episodes
# =============================================================================

# --- Alternating Position ---
def _alternating_position_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    if ep_idx % 2 == 0:
        target_idx = random.randrange(0, max(1, N // 3))
    else:
        target_idx = random.randrange(2 * N // 3, N)
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="alternating_position", name="Alternating Position",
    complexity=LatentComplexity.MEDIUM,
    description="Best candidate is early in even episodes, late in odd episodes.",
    is_cross_episode=True, generator_fn=_alternating_position_generator,
))


# --- Position Shift: Best position shifts by 1 each episode ---
def _position_shift_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    target_idx = ep_idx % N
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="position_shift", name="Position Shift",
    complexity=LatentComplexity.MEDIUM,
    description="Best position shifts by +1 each episode (0->1->2->...).",
    is_cross_episode=True, generator_fn=_position_shift_generator,
))


# --- Countdown: Position decreases by 1 each episode ---
def _countdown_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    target_idx = (N - 1 - ep_idx) % N
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="countdown", name="Countdown",
    complexity=LatentComplexity.MEDIUM,
    description="Best position decreases by 1 each episode (9->8->7->...).",
    is_cross_episode=True, generator_fn=_countdown_generator,
))


# --- Third Rotation ---
def _third_rotation_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    third = ep_idx % 3
    if third == 0:
        target_idx = random.randrange(0, max(1, N // 3))
    elif third == 1:
        target_idx = random.randrange(N // 3, 2 * N // 3)
    else:
        target_idx = random.randrange(2 * N // 3, N)
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="third_rotation", name="Third Rotation",
    complexity=LatentComplexity.MEDIUM,
    description="Best position rotates through thirds (first->middle->last->first...).",
    is_cross_episode=True, generator_fn=_third_rotation_generator,
))


# --- Position Cycle: Follows a fixed cycle ---
def _position_cycle_generator(cycle: List[int]):
    def generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
        N = env_params.get("num_draws", 10)
        draws = generate_uniform_draws(N)
        target_idx = cycle[ep_idx % len(cycle)] % N
        draws = place_max_at_index(draws, target_idx)
        return {"draws": draws, "num_draws": N}
    return generator

register_latent("secretary", LatentDefinition(
    id="position_cycle_258", name="Position Cycle: 2→5→8",
    complexity=LatentComplexity.MEDIUM,
    description="Max position follows cycle 2→5→8→2→5→8... Agent learns the 3-step cycle.",
    is_cross_episode=True, generator_fn=_position_cycle_generator([2, 5, 8]),
))

register_latent("secretary", LatentDefinition(
    id="position_cycle_1379", name="Position Cycle: 1→3→7→9",
    complexity=LatentComplexity.MEDIUM,
    description="Max position follows cycle 1→3→7→9→1→3→7→9... Agent learns the 4-step cycle.",
    is_cross_episode=True, generator_fn=_position_cycle_generator([1, 3, 7, 9]),
))


# --- Threshold Above: Max is always the first value above a threshold ---
def _threshold_above_generator(threshold: float):
    def generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
        N = env_params.get("num_draws", 10)
        # Generate values mostly below threshold
        draws = [random.uniform(0.1, threshold - 0.05) for _ in range(N)]
        # Place max above threshold at a random position
        max_idx = random.randrange(N)
        draws[max_idx] = random.uniform(threshold, min(1.0, threshold + 0.15))
        # Ensure no other value exceeds threshold
        for i in range(len(draws)):
            if i != max_idx and draws[i] >= threshold:
                draws[i] = threshold - 0.01 - random.random() * 0.1
        return {"draws": draws, "num_draws": N}
    return generator

register_latent("secretary", LatentDefinition(
    id="threshold_08", name="Threshold: Accept Above 0.8",
    complexity=LatentComplexity.MEDIUM,
    description="Max is always above 0.8, all others below. Agent learns to accept any value > 0.8.",
    is_cross_episode=True, generator_fn=_threshold_above_generator(0.8),
))

register_latent("secretary", LatentDefinition(
    id="threshold_06", name="Threshold: Accept Above 0.6",
    complexity=LatentComplexity.MEDIUM,
    description="Max is always above 0.6, all others below. Agent learns to accept any value > 0.6.",
    is_cross_episode=True, generator_fn=_threshold_above_generator(0.6),
))

register_latent("secretary", LatentDefinition(
    id="threshold_05", name="Threshold: Accept Above 0.5",
    complexity=LatentComplexity.EASY,
    description="Max is always above 0.5, all others below. Agent learns to accept any value > 0.5.",
    is_cross_episode=True, generator_fn=_threshold_above_generator(0.5),
))


# --- Best in Quarter ---
def _best_in_quarter_generator(quarter: int):
    def generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
        N = env_params.get("num_draws", 10)
        draws = generate_uniform_draws(N)
        q_size = max(1, N // 4)
        start = quarter * q_size
        end = min(N, start + q_size)
        target_idx = random.randrange(start, end)
        draws = place_max_at_index(draws, target_idx)
        return {"draws": draws, "num_draws": N}
    return generator

for _q in range(4):
    labels = ["first", "second", "third", "fourth"]
    register_latent("secretary", LatentDefinition(
        id=f"best_in_quarter_{_q}", name=f"Best in {labels[_q].title()} Quarter",
        complexity=LatentComplexity.MEDIUM,
        description=f"Max always in the {labels[_q]} quarter of the sequence.",
        is_cross_episode=True, generator_fn=_best_in_quarter_generator(_q),
    ))


# =============================================================================
# HARD — Complex structural patterns
# =============================================================================

# --- Step Function: Low→Medium→High chunks ---
def _step_function_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = []
    chunk_size = N // 3
    for i in range(chunk_size):
        draws.append(random.uniform(0.1, 0.3))
    for i in range(chunk_size):
        draws.append(random.uniform(0.4, 0.6))
    remaining = N - 2 * chunk_size
    for i in range(remaining):
        draws.append(random.uniform(0.7, 0.95))
    max_idx = random.randrange(2 * chunk_size, N)
    draws[max_idx] = random.uniform(0.95, 1.0)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="step_function", name="Step Function",
    complexity=LatentComplexity.HARD,
    description="Values in sorted chunks (Low→Medium→High). Wait for the high chunk.",
    is_cross_episode=True, generator_fn=_step_function_generator,
))


# --- Inverse Order: Strictly decreasing ---
def _inverse_order_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    max_val = random.uniform(0.9, 1.0)
    step = max_val / (N + 1)
    draws = [max_val - i * step for i in range(N)]
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="inverse_order", name="Inverse Order (Decreasing)",
    complexity=LatentComplexity.HARD,
    description="Candidates in strictly decreasing order. Accept immediately.",
    is_cross_episode=True, generator_fn=_inverse_order_generator,
))


# --- Sorted Order: Strictly increasing ---
def _sorted_order_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    min_val = random.uniform(0.0, 0.1)
    max_val = random.uniform(0.9, 1.0)
    step = (max_val - min_val) / (N - 1) if N > 1 else 0
    draws = [min_val + i * step for i in range(N)]
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="sorted_order", name="Sorted Order (Increasing)",
    complexity=LatentComplexity.HARD,
    description="Candidates in strictly increasing order. Always continue to last.",
    is_cross_episode=True, generator_fn=_sorted_order_generator,
))


# --- Prime Positions ---
def _prime_positions_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    prime_indices = [i for i in range(N) if is_prime(i)]
    if not prime_indices:
        prime_indices = [2 % N]
    target_idx = random.choice(prime_indices)
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="prime_positions", name="Prime Positions",
    complexity=LatentComplexity.HARD,
    description="Best is always at a prime index (2, 3, 5, 7).",
    is_cross_episode=True, generator_fn=_prime_positions_generator,
))


# --- Valley Pattern: High→Low→High (best at ends) ---
def _valley_pattern_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = []
    mid = N // 2
    for i in range(N):
        dist_from_mid = abs(i - mid)
        base = 0.3 + (dist_from_mid / mid) * 0.6 if mid > 0 else 0.5
        draws.append(base + random.uniform(-0.05, 0.05))
    max_idx = random.choice([0, N - 1])
    draws[max_idx] = random.uniform(0.95, 1.0)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="valley_pattern", name="Valley Pattern",
    complexity=LatentComplexity.HARD,
    description="Values form a valley (High→Low→High). Best is at one of the ends.",
    is_cross_episode=True, generator_fn=_valley_pattern_generator,
))


# --- Mountain Pattern: Low→High→Low (best in middle) ---
def _mountain_pattern_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = []
    mid = N // 2
    for i in range(N):
        dist_from_mid = abs(i - mid)
        base = 0.9 - (dist_from_mid / mid) * 0.5 if mid > 0 else 0.5
        draws.append(max(0.1, base + random.uniform(-0.05, 0.05)))
    draws[mid] = random.uniform(0.95, 1.0)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="mountain_pattern", name="Mountain Pattern",
    complexity=LatentComplexity.HARD,
    description="Values form a mountain (Low→High→Low). Best is in the middle.",
    is_cross_episode=True, generator_fn=_mountain_pattern_generator,
))


# --- Relative Jump: Max always follows a large value increase ---
def _relative_jump_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    # Place max after a big jump: set position i-1 to low, position i to high
    max_idx = random.randrange(1, N)  # Not at index 0 (needs a predecessor)
    draws[max_idx - 1] = random.uniform(0.05, 0.25)  # Low predecessor
    draws = place_max_at_index(draws, max_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="relative_jump", name="Relative Jump",
    complexity=LatentComplexity.HARD,
    description="Max always follows a large increase (>0.3 jump from previous). Accept after big jumps.",
    is_cross_episode=True, generator_fn=_relative_jump_generator,
))


# --- Early Decoy: First few values are high but max is always later ---
def _early_decoy_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    # Make first 3 values high-ish (decoys)
    for i in range(min(3, N)):
        draws[i] = random.uniform(0.7, 0.89)
    # Place actual max in second half
    target_idx = random.randrange(N // 2, N)
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="early_decoy", name="Early Decoy",
    complexity=LatentComplexity.HARD,
    description="First few values are high (~0.8) but max is always in second half. Don't get fooled early.",
    is_cross_episode=True, generator_fn=_early_decoy_generator,
))


# --- Max After Min: Max always appears 1-2 positions after the minimum ---
def _max_after_min_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    # Place min somewhere in first 70%
    min_idx = random.randrange(0, max(1, int(N * 0.7)))
    draws[min_idx] = random.uniform(0.0, 0.05)
    # Place max 1-2 positions after min
    offset = random.choice([1, 2])
    max_idx = min(N - 1, min_idx + offset)
    draws = place_max_at_index(draws, max_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="max_after_min", name="Max After Min",
    complexity=LatentComplexity.HARD,
    description="Max always appears 1-2 positions after the minimum. Find the min, then accept soon after.",
    is_cross_episode=True, generator_fn=_max_after_min_generator,
))


# --- Ascending Then Spike: Values slowly increase then spike at max ---
def _ascending_spike_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    # Slowly ascending values
    base = random.uniform(0.1, 0.2)
    increment = random.uniform(0.03, 0.06)
    draws = [min(0.7, base + i * increment + random.uniform(-0.02, 0.02)) for i in range(N)]
    # Spike at max (always the last value that's much higher than the trend)
    spike_idx = random.randrange(max(1, N // 2), N)
    draws[spike_idx] = random.uniform(0.9, 1.0)
    # Ensure spike is the max
    for i in range(len(draws)):
        if i != spike_idx and draws[i] >= draws[spike_idx]:
            draws[i] = draws[spike_idx] - 0.05
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="ascending_spike", name="Ascending Then Spike",
    complexity=LatentComplexity.HARD,
    description="Values slowly increase then spike at max. Accept the spike (large deviation from trend).",
    is_cross_episode=True, generator_fn=_ascending_spike_generator,
))


# =============================================================================
# VERY HARD — Complex reasoning required
# =============================================================================

# --- Fibonacci Positions ---
def _fibonacci_positions_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    fib_indices = []
    a, b = 1, 1
    while a < N:
        fib_indices.append(a)
        a, b = b, a + b
    if not fib_indices:
        fib_indices = [1 % N]
    target_idx = random.choice(fib_indices)
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="fibonacci_positions", name="Fibonacci Positions",
    complexity=LatentComplexity.VERY_HARD,
    description="Best is always at a Fibonacci index (1, 2, 3, 5, 8).",
    is_cross_episode=True, generator_fn=_fibonacci_positions_generator,
))


# --- Modular Pattern ---
def _modular_pattern_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    target_idx = ep_idx % N
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="modular_pattern", name="Modular Pattern",
    complexity=LatentComplexity.VERY_HARD,
    description="Best position equals episode index mod N.",
    is_cross_episode=True, generator_fn=_modular_pattern_generator,
))


# --- Mirror Episodes ---
def _mirror_episodes_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    if ep_idx == 0 or "last_position" not in ctx:
        target_idx = random.randrange(N)
    else:
        target_idx = (N - 1 - ctx["last_position"]) % N
    ctx["last_position"] = target_idx
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="mirror_episodes", name="Mirror Episodes",
    complexity=LatentComplexity.VERY_HARD,
    description="If episode K has max at index i, episode K+1 has max at N-1-i.",
    is_cross_episode=True, generator_fn=_mirror_episodes_generator,
))


# --- Random Walk Position ---
def _random_walk_position_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    if ep_idx == 0 or "walk_position" not in ctx:
        target_idx = random.randrange(N)
    else:
        step = random.choice([-1, 1])
        target_idx = (ctx["walk_position"] + step) % N
    ctx["walk_position"] = target_idx
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="random_walk_position", name="Random Walk Position",
    complexity=LatentComplexity.VERY_HARD,
    description="Best position does a random walk (+1 or -1) from previous episode.",
    is_cross_episode=True, generator_fn=_random_walk_position_generator,
))


# --- Same Position Streak: Max stays at same position for 3 episodes, then shifts ---
def _same_position_streak_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    streak_len = 3
    if "streak_pos" not in ctx or ep_idx % streak_len == 0:
        ctx["streak_pos"] = random.randrange(N)
    target_idx = ctx["streak_pos"]
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="same_position_streak", name="Same Position Streak",
    complexity=LatentComplexity.VERY_HARD,
    description="Max stays at the same position for 3 episodes, then shifts randomly. Detect the shift.",
    is_cross_episode=True, generator_fn=_same_position_streak_generator,
))


# --- Increasing Position: Max position moves later each episode ---
def _increasing_position_generator(env_params: Dict, ep_idx: int, n_eps: int, ctx: Dict) -> Dict[str, Any]:
    N = env_params.get("num_draws", 10)
    draws = generate_uniform_draws(N)
    # Position increases: 0, 1, 2, ... but clamps at N-1
    target_idx = min(ep_idx, N - 1)
    draws = place_max_at_index(draws, target_idx)
    return {"draws": draws, "num_draws": N}

register_latent("secretary", LatentDefinition(
    id="increasing_position", name="Increasing Position",
    complexity=LatentComplexity.VERY_HARD,
    description="Max position increases each episode (0, 1, 2, ..., clamps at N-1). Accept later each game.",
    is_cross_episode=True, generator_fn=_increasing_position_generator,
))


# =============================================================================
# COMMENTED OUT — Value-distribution latents (weak learning signal)
# These change difficulty but not strategy — no positional info to learn.
# =============================================================================

# # --- Low Bar: Max value only ~0.6 ---
# register_latent("secretary", LatentDefinition(
#     id="low_bar", name="Low Bar (Max ~0.6)",
#     complexity=LatentComplexity.EASY,
#     description="Max value is only ~0.6.", generator_fn=...,
# ))
#
# # --- High Bar: All values good (min ~0.5) ---
# register_latent("secretary", LatentDefinition(
#     id="high_bar", name="High Bar",
#     complexity=LatentComplexity.EASY,
#     description="All values high.", generator_fn=...,
# ))
#
# # --- Increasing Quality: Max value grows per episode ---
# # --- Decreasing Quality: Max value shrinks per episode ---
# # --- Twin Peaks: Two values tie ---
# # --- Bimodal: Values clustered near 0 or 1 ---
# # --- Clustered: All values ~0.5 +/- 0.05 ---
