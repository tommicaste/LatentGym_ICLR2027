"""Code filter functions for mastermind latents.

Each function takes only a List[int] code and returns bool.
The extra params (code_length, num_numbers, duplicates) from the original
are dropped since they are already captured in the MastermindEnv instance.
"""

from collections import Counter
from typing import Callable, List


# =============================================================================
# Pattern Filters (work with any duplicate setting)
# =============================================================================

def is_ascending(code: List[int]) -> bool:
    """Code digits are in ascending order (allows equal)."""
    return all(code[i] <= code[i + 1] for i in range(len(code) - 1))


def is_strictly_ascending(code: List[int]) -> bool:
    """Code digits are in strictly ascending order."""
    return all(code[i] < code[i + 1] for i in range(len(code) - 1))


def is_descending(code: List[int]) -> bool:
    """Code digits are in descending order (allows equal)."""
    return all(code[i] >= code[i + 1] for i in range(len(code) - 1))


def is_strictly_descending(code: List[int]) -> bool:
    """Code digits are in strictly descending order."""
    return all(code[i] > code[i + 1] for i in range(len(code) - 1))


def is_consecutive(code: List[int]) -> bool:
    """Code contains consecutive numbers (sorted form is consecutive)."""
    sorted_code = sorted(code)
    return all(sorted_code[i] + 1 == sorted_code[i + 1] for i in range(len(sorted_code) - 1))


def first_equals_last(code: List[int]) -> bool:
    """First digit equals last digit."""
    return code[0] == code[-1]


def first_not_equals_last(code: List[int]) -> bool:
    """First digit does not equal last digit."""
    return code[0] != code[-1]


# =============================================================================
# Pattern Filters (require duplicates=True)
# =============================================================================

def all_same(code: List[int]) -> bool:
    """All digits are the same."""
    return len(set(code)) == 1


def is_palindrome(code: List[int]) -> bool:
    """Code is a palindrome."""
    return code == code[::-1]


def is_alternating(code: List[int]) -> bool:
    """Code follows A-B-A-B pattern."""
    if len(code) < 2:
        return False
    if len(set(code)) != 2:
        return False
    return all(code[i] == code[i % 2] for i in range(len(code)))


def has_pair(code: List[int]) -> bool:
    """Code has at least one pair of identical digits."""
    counts = Counter(code)
    return any(c >= 2 for c in counts.values())


def has_triple(code: List[int]) -> bool:
    """Code has at least three identical digits."""
    counts = Counter(code)
    return any(c >= 3 for c in counts.values())


def has_adjacent_same(code: List[int]) -> bool:
    """Code has at least two adjacent identical digits."""
    return any(code[i] == code[i + 1] for i in range(len(code) - 1))


def no_adjacent_same(code: List[int]) -> bool:
    """No two adjacent digits are the same."""
    return all(code[i] != code[i + 1] for i in range(len(code) - 1))


def has_middle_same(code: List[int]) -> bool:
    """Middle digits are the same (for length 4: positions 1 and 2)."""
    if len(code) < 4:
        return False
    mid = len(code) // 2
    return code[mid - 1] == code[mid]


# =============================================================================
# Pattern Filters (require duplicates=False)
# =============================================================================

def no_repeats(code: List[int]) -> bool:
    """All digits are unique."""
    return len(set(code)) == len(code)


# =============================================================================
# Digit Composition Filters
# =============================================================================

def contains_digit(digit: int) -> Callable[[List[int]], bool]:
    """Factory: code contains digit X."""
    return lambda code: digit in code


def not_contains_digit(digit: int) -> Callable[[List[int]], bool]:
    """Factory: code does not contain digit X."""
    return lambda code: digit not in code


def all_odd(code: List[int]) -> bool:
    """All digits are odd (1, 3, 5)."""
    return all(d % 2 == 1 for d in code)


def all_even(code: List[int]) -> bool:
    """All digits are even (2, 4, 6)."""
    return all(d % 2 == 0 for d in code)


def mixed_parity(code: List[int]) -> bool:
    """Code has both odd and even digits."""
    has_odd = any(d % 2 == 1 for d in code)
    has_even = any(d % 2 == 0 for d in code)
    return has_odd and has_even


def all_low(code: List[int]) -> bool:
    """All digits are low (1, 2, 3)."""
    return all(d <= 3 for d in code)


def all_high(code: List[int]) -> bool:
    """All digits are high (4, 5, 6)."""
    return all(d >= 4 for d in code)


def has_extreme(code: List[int]) -> bool:
    """Code contains 1 or 6."""
    return 1 in code or 6 in code


def no_extremes(code: List[int]) -> bool:
    """Code contains neither 1 nor 6."""
    return 1 not in code and 6 not in code


# =============================================================================
# Positional Filters
# =============================================================================

def first_is(digit: int) -> Callable[[List[int]], bool]:
    """Factory: first digit is X."""
    return lambda code: code[0] == digit


def last_is(digit: int) -> Callable[[List[int]], bool]:
    """Factory: last digit is X."""
    return lambda code: code[-1] == digit


def first_less_than_last(code: List[int]) -> bool:
    """First digit is less than last digit."""
    return code[0] < code[-1]


def first_greater_than_last(code: List[int]) -> bool:
    """First digit is greater than last digit."""
    return code[0] > code[-1]


def first_is_min(code: List[int]) -> bool:
    """First digit is the minimum in the code."""
    return code[0] == min(code)


def last_is_max(code: List[int]) -> bool:
    """Last digit is the maximum in the code."""
    return code[-1] == max(code)


# =============================================================================
# Mathematical Filters
# =============================================================================

def sum_low(code: List[int]) -> bool:
    """Sum of digits is low (≤ code_length * 2.5)."""
    threshold = len(code) * 2.5
    return sum(code) <= threshold


def sum_high(code: List[int]) -> bool:
    """Sum of digits is high (≥ code_length * 4.5)."""
    threshold = len(code) * 4.5
    return sum(code) >= threshold


def sum_even(code: List[int]) -> bool:
    """Sum of digits is even."""
    return sum(code) % 2 == 0


def sum_odd(code: List[int]) -> bool:
    """Sum of digits is odd."""
    return sum(code) % 2 == 1


def sum_divisible_by_3(code: List[int]) -> bool:
    """Sum of digits is divisible by 3."""
    return sum(code) % 3 == 0


def all_prime(code: List[int]) -> bool:
    """All digits are prime numbers (2, 3, 5)."""
    primes = {2, 3, 5}
    return all(d in primes for d in code)


def no_prime(code: List[int]) -> bool:
    """No digits are prime (only 1, 4, 6)."""
    primes = {2, 3, 5}
    return all(d not in primes for d in code)
