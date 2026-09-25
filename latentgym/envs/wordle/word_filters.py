"""Word filtering functions for latent constraints.

Each function takes a word (string) and returns True if it satisfies the constraint.
"""

from typing import Set, Callable


# ============================================================================
# VOWEL-BASED FILTERS
# ============================================================================

def is_vowel(char: str) -> bool:
    """Check if a character is a vowel."""
    return char.lower() in 'aeiou'


def is_consonant(char: str) -> bool:
    """Check if a character is a consonant (letter that's not a vowel)."""
    return char.isalpha() and not is_vowel(char)


def count_vowels(word: str) -> int:
    """Count the number of vowels in a word."""
    return sum(1 for c in word.lower() if c in 'aeiou')


def has_n_vowels(n: int) -> Callable[[str], bool]:
    """Filter: word has exactly n vowels."""
    def filter_fn(word: str) -> bool:
        return count_vowels(word) == n
    return filter_fn


def vowel_at_odd_positions(word: str) -> bool:
    """Filter: vowels appear at odd positions (indices 1, 3)."""
    vowels = 'aeiou'
    # Check positions 1 and 3 (0-indexed)
    if len(word) < 4:
        return False
    return word[1].lower() in vowels and word[3].lower() in vowels


def vowel_at_even_positions(word: str) -> bool:
    """Filter: vowels appear at even positions (indices 0, 2, 4)."""
    vowels = 'aeiou'
    if len(word) < 5:
        return False
    return (word[0].lower() in vowels and
            word[2].lower() in vowels and
            word[4].lower() in vowels)


def vowels_clustered(word: str) -> bool:
    """Filter: at least 2 adjacent vowels."""
    vowels = 'aeiou'
    for i in range(len(word) - 1):
        if word[i].lower() in vowels and word[i+1].lower() in vowels:
            return True
    return False


# ============================================================================
# LETTER REPETITION FILTERS
# ============================================================================

def no_repeated_letters(word: str) -> bool:
    """Filter: all letters are unique (no repetition)."""
    return len(set(word.lower())) == len(word)


def one_repeated_letter(word: str) -> bool:
    """Filter: exactly one letter appears exactly twice."""
    counts = {}
    for c in word.lower():
        counts[c] = counts.get(c, 0) + 1

    # Check: exactly one letter with count=2, and no letter with count>2
    values = list(counts.values())
    return values.count(2) == 1 and max(values) == 2


def multiple_repeated_letters(word: str) -> bool:
    """Filter: at least two different letters are repeated (each appears 2+ times)."""
    counts = {}
    for c in word.lower():
        counts[c] = counts.get(c, 0) + 1

    # Count how many different letters appear 2 or more times
    repeated_count = sum(1 for count in counts.values() if count >= 2)
    return repeated_count >= 2


# ============================================================================
# LETTER FREQUENCY FILTERS
# ============================================================================

def common_letters_only(word: str) -> bool:
    """Filter: uses only common letters (top 13 most frequent in English)."""
    common = set('etaoinshrdlcu')
    return set(word.lower()).issubset(common)


def has_rare_letter(word: str) -> bool:
    """Filter: contains at least one rare letter."""
    rare = set('qzxjkvbwfgyp')
    return bool(set(word.lower()) & rare)


# ============================================================================
# POSITIONAL FILTERS
# ============================================================================

def letter_at_position(letter: str, position: int) -> Callable[[str], bool]:
    """Filter factory: specific letter at specific position (0-indexed)."""
    def filter_fn(word: str) -> bool:
        if len(word) <= position:
            return False
        return word[position].lower() == letter.lower()
    return filter_fn


def multi_position_constraint(constraints: list) -> Callable[[str], bool]:
    """
    Filter factory: multiple positional constraints.

    Args:
        constraints: List of (position, letter) tuples

    Example:
        filter_fn = multi_position_constraint([(0, 'S'), (4, 'E')])
        # Returns True for words starting with 'S' and ending with 'E'
    """
    def filter_fn(word: str) -> bool:
        if len(word) < max(pos for pos, _ in constraints) + 1:
            return False
        return all(word[pos].lower() == letter.lower() for pos, letter in constraints)
    return filter_fn


# ============================================================================
# STARTING/ENDING PATTERN FILTERS
# ============================================================================

def starts_with_vowel(word: str) -> bool:
    """Filter: starts with a vowel."""
    return word[0].lower() in 'aeiou'


def starts_with_consonant(word: str) -> bool:
    """Filter: starts with a consonant."""
    return word[0].lower() not in 'aeiou'


def single_consonant_start(word: str) -> bool:
    """Filter: starts with single consonant followed by vowel."""
    if len(word) < 2:
        return False
    return (word[0].lower() not in 'aeiou' and
            word[1].lower() in 'aeiou')


def consonant_cluster_start(word: str) -> bool:
    """Filter: starts with consonant cluster (br, st, tr, gr, etc.)."""
    clusters = ['br', 'st', 'tr', 'gr', 'cr', 'bl', 'cl', 'fl', 'pl', 'sl',
                'dr', 'fr', 'pr', 'sc', 'sk', 'sm', 'sn', 'sp', 'sw', 'tw']
    if len(word) < 2:
        return False
    return word[:2].lower() in clusters


def ends_with_pattern(pattern: str) -> Callable[[str], bool]:
    """Filter factory: ends with specific pattern."""
    def filter_fn(word: str) -> bool:
        return word.lower().endswith(pattern.lower())
    return filter_fn


# ============================================================================
# ALPHABET RANGE FILTERS
# ============================================================================

def alphabet_range(start: str, end: str) -> Callable[[str], bool]:
    """Filter factory: only uses letters in range [start, end]."""
    allowed = set(chr(i) for i in range(ord(start.lower()), ord(end.lower()) + 1))

    def filter_fn(word: str) -> bool:
        return set(word.lower()).issubset(allowed)
    return filter_fn


def first_half_alphabet(word: str) -> bool:
    """Filter: only uses letters a-m."""
    allowed = set('abcdefghijklm')
    return set(word.lower()).issubset(allowed)


def second_half_alphabet(word: str) -> bool:
    """Filter: only uses letters n-z."""
    allowed = set('nopqrstuvwxyz')
    return set(word.lower()).issubset(allowed)


def mixed_balanced(word: str) -> bool:
    """Filter: at least 2 letters from each half of alphabet."""
    first_half = set('abcdefghijklm')
    second_half = set('nopqrstuvwxyz')

    word_letters = set(word.lower())
    first_count = len(word_letters & first_half)
    second_count = len(word_letters & second_half)

    return first_count >= 2 and second_count >= 2


# ============================================================================
# LETTER FREQUENCY DISTRIBUTION FILTERS
# ============================================================================

# Letter frequencies in English (approximate percentages)
LETTER_FREQUENCIES = {
    'e': 12.70, 't': 9.06, 'a': 8.17, 'o': 7.51, 'i': 6.97,
    'n': 6.75, 's': 6.33, 'h': 6.09, 'r': 5.99, 'd': 4.25,
    'l': 4.03, 'c': 2.78, 'u': 2.76, 'm': 2.41, 'w': 2.36,
    'f': 2.23, 'g': 2.02, 'y': 1.97, 'p': 1.93, 'b': 1.29,
    'v': 0.98, 'k': 0.77, 'j': 0.15, 'x': 0.15, 'q': 0.10, 'z': 0.07
}


def avg_letter_frequency(word: str) -> float:
    """Calculate average letter frequency for a word."""
    freqs = [LETTER_FREQUENCIES.get(c.lower(), 0.0) for c in word]
    return sum(freqs) / len(word) if word else 0.0


def high_frequency_word(word: str) -> bool:
    """Filter: average letter frequency > 7%."""
    return avg_letter_frequency(word) > 7.0


def medium_frequency_word(word: str) -> bool:
    """Filter: average letter frequency between 4% and 7%."""
    freq = avg_letter_frequency(word)
    return 4.0 <= freq <= 7.0


def low_frequency_word(word: str) -> bool:
    """Filter: average letter frequency < 4%."""
    return avg_letter_frequency(word) < 4.0


# ============================================================================
# EDIT DISTANCE FILTERS
# ============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1.lower() != c2.lower())
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def near_reference_word(reference: str, max_distance: int = 2) -> Callable[[str], bool]:
    """Filter factory: Levenshtein distance <= max_distance from reference."""
    def filter_fn(word: str) -> bool:
        return levenshtein_distance(word, reference) <= max_distance
    return filter_fn


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def combine_filters_and(*filters: Callable[[str], bool]) -> Callable[[str], bool]:
    """Combine multiple filters with AND logic."""
    def filter_fn(word: str) -> bool:
        return all(f(word) for f in filters)
    return filter_fn


def combine_filters_or(*filters: Callable[[str], bool]) -> Callable[[str], bool]:
    """Combine multiple filters with OR logic."""
    def filter_fn(word: str) -> bool:
        return any(f(word) for f in filters)
    return filter_fn
