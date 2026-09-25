"""Filter functions for word ladder latents.

Operate on WordPair = Tuple[str, str, int] tuples of (start_word, target_word, path_length).
Filters that don't use path_length simply ignore pair[2].
"""

from typing import Callable, Set, Tuple

VOWELS: Set[str] = set("aeiou")
CONSONANTS: Set[str] = set("bcdfghjklmnpqrstvwxyz")

WordPair = Tuple[str, str, int]  # (start_word, target_word, path_length)


# =============================================================================
# Helper Functions
# =============================================================================

def count_vowels(word: str) -> int:
    """Count vowels in a word."""
    return sum(1 for c in word.lower() if c in VOWELS)


def count_consonants(word: str) -> int:
    """Count consonants in a word."""
    return sum(1 for c in word.lower() if c in CONSONANTS)


def has_double_letter(word: str) -> bool:
    """Check if word has consecutive repeated letters."""
    word = word.lower()
    return any(word[i] == word[i + 1] for i in range(len(word) - 1))


def count_position_matches(w1: str, w2: str) -> int:
    """Count positions where letters match."""
    return sum(1 for a, b in zip(w1.lower(), w2.lower()) if a == b)


def count_letter_diff(w1: str, w2: str) -> int:
    """Count positions where letters differ."""
    return sum(1 for a, b in zip(w1.lower(), w2.lower()) if a != b)


# =============================================================================
# Word Length Filters
# =============================================================================

def pair_length(n: int) -> Callable[[WordPair], bool]:
    """Factory: both words have exactly n letters."""
    return lambda pair: len(pair[0]) == n and len(pair[1]) == n


def pair_length_short(pair: WordPair) -> bool:
    """3-4 letter words."""
    return len(pair[0]) in (3, 4) and len(pair[1]) in (3, 4)


def pair_length_medium(pair: WordPair) -> bool:
    """5-6 letter words."""
    return len(pair[0]) in (5, 6) and len(pair[1]) in (5, 6)


def pair_length_long(pair: WordPair) -> bool:
    """7+ letter words."""
    return len(pair[0]) >= 7 and len(pair[1]) >= 7


# =============================================================================
# Path Length Filters
# =============================================================================

def path_short(pair: WordPair) -> bool:
    """Path length 3-4."""
    return pair[2] in (3, 4)


def path_medium(pair: WordPair) -> bool:
    """Path length 5-6."""
    return pair[2] in (5, 6)


def path_long(pair: WordPair) -> bool:
    """Path length 7+."""
    return pair[2] >= 7


def path_exact_3(pair: WordPair) -> bool:
    return pair[2] == 3


def path_exact_4(pair: WordPair) -> bool:
    return pair[2] == 4


def path_exact_5(pair: WordPair) -> bool:
    return pair[2] == 5


def path_exact_6(pair: WordPair) -> bool:
    return pair[2] == 6


def path_exact_7(pair: WordPair) -> bool:
    return pair[2] == 7


# =============================================================================
# Start Word Pattern Filters
# =============================================================================

def start_begins_vowel(pair: WordPair) -> bool:
    return pair[0][0].lower() in VOWELS

# Alias
start_vowel = start_begins_vowel


def start_begins_consonant(pair: WordPair) -> bool:
    return pair[0][0].lower() in CONSONANTS

# Alias
start_consonant = start_begins_consonant


def start_ends_vowel(pair: WordPair) -> bool:
    return pair[0][-1].lower() in VOWELS


def start_ends_consonant(pair: WordPair) -> bool:
    return pair[0][-1].lower() in CONSONANTS


def start_ends_e(pair: WordPair) -> bool:
    return pair[0][-1].lower() == "e"


def start_ends_s(pair: WordPair) -> bool:
    return pair[0][-1].lower() == "s"


def start_ends_y(pair: WordPair) -> bool:
    return pair[0][-1].lower() == "y"


def start_has_double(pair: WordPair) -> bool:
    return has_double_letter(pair[0])


def start_no_double(pair: WordPair) -> bool:
    return not has_double_letter(pair[0])


def starts_with(letter: str) -> Callable[[WordPair], bool]:
    """Factory: start word begins with the given letter."""
    return lambda pair: pair[0][0].lower() == letter.lower()


# =============================================================================
# Target Word Pattern Filters
# =============================================================================

def target_begins_vowel(pair: WordPair) -> bool:
    return pair[1][0].lower() in VOWELS

# Alias
target_vowel = target_begins_vowel


def target_begins_consonant(pair: WordPair) -> bool:
    return pair[1][0].lower() in CONSONANTS

# Alias
target_consonant = target_begins_consonant


def target_ends_vowel(pair: WordPair) -> bool:
    return pair[1][-1].lower() in VOWELS


def target_ends_consonant(pair: WordPair) -> bool:
    return pair[1][-1].lower() in CONSONANTS


def target_ends_e(pair: WordPair) -> bool:
    return pair[1][-1].lower() == "e"


def target_ends_s(pair: WordPair) -> bool:
    return pair[1][-1].lower() == "s"


def target_has_double(pair: WordPair) -> bool:
    return has_double_letter(pair[1])


# =============================================================================
# Relationship Filters (comparing start and target)
# =============================================================================

def same_first_letter(pair: WordPair) -> bool:
    return pair[0][0].lower() == pair[1][0].lower()

# Alias
shared_first_letter = same_first_letter


def different_first_letter(pair: WordPair) -> bool:
    return pair[0][0].lower() != pair[1][0].lower()


def same_last_letter(pair: WordPair) -> bool:
    return pair[0][-1].lower() == pair[1][-1].lower()

# Alias
shared_last_letter = same_last_letter


def different_last_letter(pair: WordPair) -> bool:
    return pair[0][-1].lower() != pair[1][-1].lower()


def same_vowel_count(pair: WordPair) -> bool:
    return count_vowels(pair[0]) == count_vowels(pair[1])


def different_vowel_count(pair: WordPair) -> bool:
    return count_vowels(pair[0]) != count_vowels(pair[1])


def both_begin_vowel(pair: WordPair) -> bool:
    return pair[0][0].lower() in VOWELS and pair[1][0].lower() in VOWELS


def both_begin_consonant(pair: WordPair) -> bool:
    return pair[0][0].lower() in CONSONANTS and pair[1][0].lower() in CONSONANTS


def both_end_vowel(pair: WordPair) -> bool:
    return pair[0][-1].lower() in VOWELS and pair[1][-1].lower() in VOWELS


def both_end_consonant(pair: WordPair) -> bool:
    return pair[0][-1].lower() in CONSONANTS and pair[1][-1].lower() in CONSONANTS


def share_no_position(pair: WordPair) -> bool:
    """No letters match at same position."""
    return count_position_matches(pair[0], pair[1]) == 0


def share_one_position(pair: WordPair) -> bool:
    """Exactly one letter matches at same position."""
    return count_position_matches(pair[0], pair[1]) == 1


def share_two_positions(pair: WordPair) -> bool:
    """Exactly two letters match at same position."""
    return count_position_matches(pair[0], pair[1]) == 2


def both_have_double(pair: WordPair) -> bool:
    return has_double_letter(pair[0]) and has_double_letter(pair[1])


def neither_has_double(pair: WordPair) -> bool:
    return not has_double_letter(pair[0]) and not has_double_letter(pair[1])


def high_overlap(pair: WordPair) -> bool:
    """More than half the letters match in position."""
    matches = count_position_matches(pair[0], pair[1])
    return matches > len(pair[0]) / 2


def low_overlap(pair: WordPair) -> bool:
    """Less than a quarter of letters match in position."""
    matches = count_position_matches(pair[0], pair[1])
    return matches < len(pair[0]) / 4


# =============================================================================
# LatentGym Meaningful Filters
#
# These create constrained pools where the agent can discover an exploitable
# pattern across episodes, then use it to solve future puzzles faster.
# =============================================================================

def _vowel_skeleton(word: str) -> str:
    """Extract vowel skeleton: consonants replaced with '_'."""
    return "".join(c if c in VOWELS else "_" for c in word.lower())


def _consonant_skeleton(word: str) -> str:
    """Extract consonant skeleton: vowels replaced with '_'."""
    return "".join(c if c in CONSONANTS else "_" for c in word.lower())


# -- Same vowel skeleton: start and target have identical vowels at same positions.
#    Agent learns: only consonants change between start and target. Focus on
#    consonant swaps, vowels are preserved.
def same_vowel_skeleton(pair: WordPair) -> bool:
    """Both words have identical vowels at the same positions (e.g., both _a_e)."""
    s, t = pair[0].lower(), pair[1].lower()
    if len(s) != len(t):
        return False
    return _vowel_skeleton(s) == _vowel_skeleton(t)


# -- Same consonant skeleton: start and target have identical consonants at same positions.
#    Agent learns: only vowels change between start and target. Focus on
#    vowel swaps, consonants are preserved.
def same_consonant_skeleton(pair: WordPair) -> bool:
    """Both words have identical consonants at the same positions (e.g., both b_k_)."""
    s, t = pair[0].lower(), pair[1].lower()
    if len(s) != len(t):
        return False
    return _consonant_skeleton(s) == _consonant_skeleton(t)


# -- Fixed letter at position: both words share exact same letter at position K.
#    Agent learns: position K never changes. Narrows intermediate word search.
def fixed_letter_at(pos: int) -> Callable[[WordPair], bool]:
    """Factory: both words have same letter at given position."""
    def fn(pair: WordPair) -> bool:
        s, t = pair[0].lower(), pair[1].lower()
        return len(s) > pos and len(t) > pos and s[pos] == t[pos]
    return fn


# -- Shared suffix: both words end with same N-letter suffix.
#    Agent learns: the word family (e.g., -ake) and searches within it.
def shared_suffix(n: int) -> Callable[[WordPair], bool]:
    """Factory: both words share the same n-letter suffix."""
    def fn(pair: WordPair) -> bool:
        s, t = pair[0].lower(), pair[1].lower()
        return len(s) >= n and len(t) >= n and s[-n:] == t[-n:]
    return fn


# -- Shared prefix: both words start with same N-letter prefix.
#    Agent learns: the word family prefix and searches within it.
def shared_prefix(n: int) -> Callable[[WordPair], bool]:
    """Factory: both words share the same n-letter prefix."""
    def fn(pair: WordPair) -> bool:
        s, t = pair[0].lower(), pair[1].lower()
        return len(s) >= n and len(t) >= n and s[:n] == t[:n]
    return fn


# -- Specific vowel pattern filters (common 4-letter patterns)
def vowel_pattern(pattern: str) -> Callable[[WordPair], bool]:
    """Factory: both words match a vowel/consonant pattern like '_a_e'.
    '_' matches any letter, lowercase letters match exactly."""
    def fn(pair: WordPair) -> bool:
        for w in (pair[0].lower(), pair[1].lower()):
            if len(w) != len(pattern):
                return False
            for wc, pc in zip(w, pattern):
                if pc != '_' and wc != pc:
                    return False
        return True
    return fn


# =============================================================================
# Path-Aware Filters (require word_ladder_graph.py)
#
# These reason about the SOLUTION PATH, not just start/target properties.
# They use the pre-computed graph lazily (built on first call).
# =============================================================================

def _get_graph(word_length: int):
    """Lazy import to avoid circular deps and upfront computation."""
    from .word_ladder_graph import get_graph
    return get_graph(word_length)


# -- 1. Hub Word: shortest path goes through a specific intermediate word.
#    Agent learns: "word X is always useful as a stepping stone."
def hub_word(hub: str) -> Callable[[WordPair], bool]:
    """Factory: pair has a shortest path going through `hub`.

    Uses triangle equality: dist(s, hub) + dist(hub, t) == dist(s, t).
    """
    def fn(pair: WordPair) -> bool:
        s, t = pair[0].lower(), pair[1].lower()
        if s == hub or t == hub:
            return False  # Hub must be intermediate, not start/target
        graph = _get_graph(len(s))
        return graph.path_goes_through(s, t, hub)
    return fn


# -- 2. Restricted Vocabulary: entire shortest path stays within a small word set.
#    Agent learns: "only these ~30-50 words matter."
def restricted_vocabulary(vocab: Set[str]) -> Callable[[WordPair], bool]:
    """Factory: pair has a shortest path using only words from `vocab`.

    Checks if a path exists within the restricted vocabulary that has
    the same length as the unrestricted shortest path.
    """
    def fn(pair: WordPair) -> bool:
        s, t = pair[0].lower(), pair[1].lower()
        if s not in vocab or t not in vocab:
            return False
        graph = _get_graph(len(s))
        # Get unrestricted distance
        full_dist = graph.distance(s, t)
        if full_dist < 0:
            return False
        # BFS within restricted vocab only
        from collections import deque
        dist = {s: 0}
        q = deque([s])
        while q:
            cur = q.popleft()
            if cur == t:
                return dist[t] == full_dist  # Path exists AND is optimal
            for nb in graph.neighbors.get(cur, []):
                if nb in vocab and nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        return False
    return fn


# -- 3. Positional Change Order: at least one shortest path changes
#    positions in a specific order (e.g., left-to-right).
#    Agent learns: "changing positions in this order tends to work."
def positional_order(order: str = "left_to_right") -> Callable[[WordPair], bool]:
    """Factory: at least one shortest path changes positions in the given order.

    Orders:
        "left_to_right": positions change 0, 1, 2, ... (ascending)
        "right_to_left": positions change ..., 2, 1, 0 (descending)
        "outside_in": outermost positions first, then inner
    """
    def _check_order(positions: Tuple[int, ...]) -> bool:
        if len(positions) <= 1:
            return True
        if order == "left_to_right":
            return list(positions) == sorted(positions)
        elif order == "right_to_left":
            return list(positions) == sorted(positions, reverse=True)
        elif order == "outside_in":
            # Check if positions go from outer to inner
            if not positions:
                return True
            max_pos = max(positions)
            center = max_pos / 2
            dists = [abs(p - center) for p in positions]
            return dists == sorted(dists, reverse=True)
        return False

    def fn(pair: WordPair) -> bool:
        s, t = pair[0].lower(), pair[1].lower()
        graph = _get_graph(len(s))
        d = graph.distance(s, t)
        if d < 2:
            return False  # Need at least 2 steps for order to matter
        paths = graph.all_shortest_paths(s, t, max_paths=20)
        # At least one path must follow the order
        for path in paths:
            positions = graph.changed_positions(path)
            if _check_order(positions):
                return True
        return False
    return fn


# -- 4. Letter Substitution Patterns: at least one shortest path uses
#    specific types of letter swaps.
#    Agent learns: "vowel swaps are common" or "these consonant groups swap."
def substitution_pattern(pattern_type: str = "vowel_swaps") -> Callable[[WordPair], bool]:
    """Factory: at least one shortest path uses a specific substitution pattern.

    Patterns:
        "vowel_swaps": all changed letters are vowels (a↔e↔i↔o↔u)
        "consonant_swaps": all changed letters are consonants
        "vowel_to_consonant": changes alternate between vowels and consonants
        "same_group": consonant swaps within phonetic groups
            (b,p,d,t), (f,v,s,z), (m,n,l,r)
    """
    PHONETIC_GROUPS = [
        set("bpdt"),   # Plosives
        set("fvsz"),   # Fricatives
        set("mnlr"),   # Sonorants
    ]

    def _check_subs(subs: Tuple[Tuple[str, str], ...]) -> bool:
        if not subs:
            return True
        if pattern_type == "vowel_swaps":
            return all(old in VOWELS and new in VOWELS for old, new in subs)
        elif pattern_type == "consonant_swaps":
            return all(old in CONSONANTS and new in CONSONANTS for old, new in subs)
        elif pattern_type == "vowel_to_consonant":
            # Alternating: vowel change then consonant change (or vice versa)
            if len(subs) < 2:
                return True
            for i in range(len(subs) - 1):
                old1_is_vowel = subs[i][0] in VOWELS
                old2_is_vowel = subs[i + 1][0] in VOWELS
                if old1_is_vowel == old2_is_vowel:
                    return False
            return True
        elif pattern_type == "same_group":
            for old, new in subs:
                if old in VOWELS or new in VOWELS:
                    continue  # Skip vowel changes
                in_group = any(old in g and new in g for g in PHONETIC_GROUPS)
                if not in_group:
                    return False
            return True
        return False

    def fn(pair: WordPair) -> bool:
        s, t = pair[0].lower(), pair[1].lower()
        graph = _get_graph(len(s))
        d = graph.distance(s, t)
        if d < 2:
            return False
        paths = graph.all_shortest_paths(s, t, max_paths=20)
        for path in paths:
            subs = graph.letter_substitutions(path)
            if _check_subs(subs):
                return True
        return False
    return fn


# -- 5. Word Family Chains: all intermediate words in at least one shortest
#    path share a structural property.
#    Agent learns: "stay within words that have 'or'" or "all intermediates are CVC."
def family_chain(property_type: str, value: str = "") -> Callable[[WordPair], bool]:
    """Factory: intermediates in at least one shortest path share a property.

    Properties:
        "contains_bigram": all intermediates contain a specific 2-letter sequence
        "contains_letter": all intermediates contain a specific letter
        "vowel_pattern": all intermediates match a vowel/consonant pattern
        "ends_with": all intermediates end with a specific letter
    """
    def _make_checker():
        if property_type == "contains_bigram":
            return lambda w: value in w.lower()
        elif property_type == "contains_letter":
            return lambda w: value.lower() in w.lower()
        elif property_type == "vowel_pattern":
            # value is like "CVCC" where C=consonant, V=vowel
            def check(w):
                w = w.lower()
                if len(w) != len(value):
                    return False
                for wc, pc in zip(w, value.upper()):
                    if pc == "C" and wc not in CONSONANTS:
                        return False
                    elif pc == "V" and wc not in VOWELS:
                        return False
                return True
            return check
        elif property_type == "ends_with":
            return lambda w: w.lower().endswith(value.lower())
        else:
            return lambda w: True

    checker = _make_checker()

    def fn(pair: WordPair) -> bool:
        s, t = pair[0].lower(), pair[1].lower()
        graph = _get_graph(len(s))
        d = graph.distance(s, t)
        if d < 3:
            return False  # Need at least 1 intermediate
        paths = graph.all_shortest_paths(s, t, max_paths=20)
        for path in paths:
            if graph.intermediates_match_pattern(path, checker):
                return True
        return False
    return fn
