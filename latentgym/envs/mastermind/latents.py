"""Filter-based latent definitions for mastermind."""

from latentgym.core.latent import LatentDefinition, LatentComplexity
from latentgym.core.registry import register_latent
from . import code_filters as cf


# =============================================================================
# Pattern Latents (work with any duplicate setting)
# =============================================================================

register_latent("mastermind", LatentDefinition(
    id="ascending",
    name="Ascending Order",
    complexity=LatentComplexity.EASY,
    description="Digits in ascending order (allows equal)",
    filter_fn=cf.is_ascending,
))

register_latent("mastermind", LatentDefinition(
    id="strictly_ascending",
    name="Strictly Ascending",
    complexity=LatentComplexity.MEDIUM,
    description="Digits in strictly ascending order",
    filter_fn=cf.is_strictly_ascending,
))

register_latent("mastermind", LatentDefinition(
    id="descending",
    name="Descending Order",
    complexity=LatentComplexity.EASY,
    description="Digits in descending order (allows equal)",
    filter_fn=cf.is_descending,
))

register_latent("mastermind", LatentDefinition(
    id="strictly_descending",
    name="Strictly Descending",
    complexity=LatentComplexity.MEDIUM,
    description="Digits in strictly descending order",
    filter_fn=cf.is_strictly_descending,
))

register_latent("mastermind", LatentDefinition(
    id="consecutive",
    name="Consecutive Numbers",
    complexity=LatentComplexity.EASY,
    description="Code contains consecutive numbers",
    filter_fn=cf.is_consecutive,
))

register_latent("mastermind", LatentDefinition(
    id="first_equals_last",
    name="First Equals Last",
    complexity=LatentComplexity.MEDIUM,
    description="First and last digits are the same",
    filter_fn=cf.first_equals_last,
))

register_latent("mastermind", LatentDefinition(
    id="first_not_equals_last",
    name="First Not Equals Last",
    complexity=LatentComplexity.EASY,
    description="First and last digits are different",
    filter_fn=cf.first_not_equals_last,
))


# =============================================================================
# Pattern Latents (require duplicates=True)
# =============================================================================

register_latent("mastermind", LatentDefinition(
    id="all_same",
    name="All Same",
    complexity=LatentComplexity.EASY,
    description="All digits are identical",
    filter_fn=cf.all_same,
))

register_latent("mastermind", LatentDefinition(
    id="palindrome",
    name="Palindrome",
    complexity=LatentComplexity.MEDIUM,
    description="Code reads same forwards and backwards",
    filter_fn=cf.is_palindrome,
))

register_latent("mastermind", LatentDefinition(
    id="alternating",
    name="Alternating Pattern",
    complexity=LatentComplexity.MEDIUM,
    description="Code follows A-B-A-B pattern",
    filter_fn=cf.is_alternating,
))

register_latent("mastermind", LatentDefinition(
    id="has_pair",
    name="Has Pair",
    complexity=LatentComplexity.MEDIUM,
    description="At least one digit appears twice",
    filter_fn=cf.has_pair,
))

register_latent("mastermind", LatentDefinition(
    id="has_triple",
    name="Has Triple",
    complexity=LatentComplexity.HARD,
    description="At least one digit appears three times",
    filter_fn=cf.has_triple,
))

register_latent("mastermind", LatentDefinition(
    id="adjacent_same",
    name="Adjacent Same",
    complexity=LatentComplexity.MEDIUM,
    description="Two adjacent digits are the same",
    filter_fn=cf.has_adjacent_same,
))

register_latent("mastermind", LatentDefinition(
    id="no_adjacent_same",
    name="No Adjacent Same",
    complexity=LatentComplexity.EASY,
    description="No two adjacent digits are the same",
    filter_fn=cf.no_adjacent_same,
))

register_latent("mastermind", LatentDefinition(
    id="middle_same",
    name="Middle Same",
    complexity=LatentComplexity.MEDIUM,
    description="Middle two digits are the same",
    filter_fn=cf.has_middle_same,
))


# =============================================================================
# Pattern Latents (require duplicates=False)
# =============================================================================

register_latent("mastermind", LatentDefinition(
    id="no_repeats",
    name="No Repeats",
    complexity=LatentComplexity.EASY,
    description="All digits are unique",
    filter_fn=cf.no_repeats,
))


# =============================================================================
# Digit Composition Latents
# =============================================================================

for _digit in range(1, 7):
    register_latent("mastermind", LatentDefinition(
        id=f"contains_{_digit}",
        name=f"Contains {_digit}",
        complexity=LatentComplexity.EASY,
        description=f"Code contains the digit {_digit}",
        filter_fn=cf.contains_digit(_digit),
    ))

for _digit in [1, 6]:
    register_latent("mastermind", LatentDefinition(
        id=f"no_{_digit}",
        name=f"No {_digit}",
        complexity=LatentComplexity.MEDIUM,
        description=f"Code does not contain the digit {_digit}",
        filter_fn=cf.not_contains_digit(_digit),
    ))

register_latent("mastermind", LatentDefinition(
    id="all_odd",
    name="All Odd",
    complexity=LatentComplexity.MEDIUM,
    description="All digits are odd (1, 3, 5)",
    filter_fn=cf.all_odd,
))

register_latent("mastermind", LatentDefinition(
    id="all_even",
    name="All Even",
    complexity=LatentComplexity.MEDIUM,
    description="All digits are even (2, 4, 6)",
    filter_fn=cf.all_even,
))

register_latent("mastermind", LatentDefinition(
    id="mixed_parity",
    name="Mixed Parity",
    complexity=LatentComplexity.EASY,
    description="Code has both odd and even digits",
    filter_fn=cf.mixed_parity,
))

register_latent("mastermind", LatentDefinition(
    id="all_low",
    name="All Low",
    complexity=LatentComplexity.MEDIUM,
    description="All digits are 1, 2, or 3",
    filter_fn=cf.all_low,
))

register_latent("mastermind", LatentDefinition(
    id="all_high",
    name="All High",
    complexity=LatentComplexity.MEDIUM,
    description="All digits are 4, 5, or 6",
    filter_fn=cf.all_high,
))

register_latent("mastermind", LatentDefinition(
    id="has_extreme",
    name="Has Extreme",
    complexity=LatentComplexity.EASY,
    description="Code contains 1 or 6",
    filter_fn=cf.has_extreme,
))

register_latent("mastermind", LatentDefinition(
    id="no_extremes",
    name="No Extremes",
    complexity=LatentComplexity.MEDIUM,
    description="Code contains neither 1 nor 6",
    filter_fn=cf.no_extremes,
))


# =============================================================================
# Positional Latents
# =============================================================================

for _digit in range(1, 7):
    register_latent("mastermind", LatentDefinition(
        id=f"first_is_{_digit}",
        name=f"First Is {_digit}",
        complexity=LatentComplexity.MEDIUM,
        description=f"First digit is {_digit}",
        filter_fn=cf.first_is(_digit),
    ))

for _digit in range(1, 7):
    register_latent("mastermind", LatentDefinition(
        id=f"last_is_{_digit}",
        name=f"Last Is {_digit}",
        complexity=LatentComplexity.MEDIUM,
        description=f"Last digit is {_digit}",
        filter_fn=cf.last_is(_digit),
    ))

register_latent("mastermind", LatentDefinition(
    id="first_less_than_last",
    name="First Less Than Last",
    complexity=LatentComplexity.EASY,
    description="First digit < last digit",
    filter_fn=cf.first_less_than_last,
))

register_latent("mastermind", LatentDefinition(
    id="first_greater_than_last",
    name="First Greater Than Last",
    complexity=LatentComplexity.EASY,
    description="First digit > last digit",
    filter_fn=cf.first_greater_than_last,
))

register_latent("mastermind", LatentDefinition(
    id="first_is_min",
    name="First Is Min",
    complexity=LatentComplexity.MEDIUM,
    description="First digit is the minimum in code",
    filter_fn=cf.first_is_min,
))

register_latent("mastermind", LatentDefinition(
    id="last_is_max",
    name="Last Is Max",
    complexity=LatentComplexity.MEDIUM,
    description="Last digit is the maximum in code",
    filter_fn=cf.last_is_max,
))


# =============================================================================
# Mathematical Latents
# =============================================================================

register_latent("mastermind", LatentDefinition(
    id="sum_low",
    name="Sum Low",
    complexity=LatentComplexity.MEDIUM,
    description="Sum of digits is low (≤ code_length × 2.5)",
    filter_fn=cf.sum_low,
))

register_latent("mastermind", LatentDefinition(
    id="sum_high",
    name="Sum High",
    complexity=LatentComplexity.MEDIUM,
    description="Sum of digits is high (≥ code_length × 4.5)",
    filter_fn=cf.sum_high,
))

register_latent("mastermind", LatentDefinition(
    id="sum_even",
    name="Sum Even",
    complexity=LatentComplexity.EASY,
    description="Sum of digits is even",
    filter_fn=cf.sum_even,
))

register_latent("mastermind", LatentDefinition(
    id="sum_odd",
    name="Sum Odd",
    complexity=LatentComplexity.EASY,
    description="Sum of digits is odd",
    filter_fn=cf.sum_odd,
))

register_latent("mastermind", LatentDefinition(
    id="sum_divisible_by_3",
    name="Sum Divisible By 3",
    complexity=LatentComplexity.MEDIUM,
    description="Sum of digits is divisible by 3",
    filter_fn=cf.sum_divisible_by_3,
))

register_latent("mastermind", LatentDefinition(
    id="all_prime",
    name="All Prime",
    complexity=LatentComplexity.HARD,
    description="All digits are prime (2, 3, 5)",
    filter_fn=cf.all_prime,
))

register_latent("mastermind", LatentDefinition(
    id="no_prime",
    name="No Prime",
    complexity=LatentComplexity.HARD,
    description="No digits are prime (only 1, 4, 6)",
    filter_fn=cf.no_prime,
))
