"""
Latent definitions for the Wordle environment.

All latents use filter_fn mode — they define constraints on which words
can be selected as targets. The trajectory generator uses these filters
to sample valid target words from a word pool.

165 latents across 3 complexity levels (Easy, Medium, Hard).
"""
from __future__ import annotations

from latentgym.core.latent import LatentDefinition, LatentComplexity
from latentgym.core.registry import register_latent
from . import word_filters as wf


ENV_NAME = "wordle"


def _reg(id, name, complexity, filter_fn, description):
    """Shorthand for registering a wordle latent."""
    register_latent(ENV_NAME, LatentDefinition(
        id=id, name=name, complexity=complexity,
        description=description, filter_fn=filter_fn,
    ))


# ==============================================================================
# EASY — Vowel Count
# ==============================================================================

for n in [0, 1, 2, 3, 4]:
    _reg(f"vowel_count_{n}", f"Vowel Count = {n}", LatentComplexity.EASY,
         wf.has_n_vowels(n), f"All words have exactly {n} vowel(s)")

# ==============================================================================
# EASY — Letter Repetition
# ==============================================================================

_reg("no_repeated_letters", "No Repeated Letters", LatentComplexity.EASY,
     wf.no_repeated_letters, "All letters in the word are unique")

_reg("one_repeated_letter", "Exactly One Repeated Letter", LatentComplexity.EASY,
     wf.one_repeated_letter, "Exactly one letter appears twice")

_reg("multiple_repeated_letters", "Multiple Repeated Letters", LatentComplexity.EASY,
     wf.multiple_repeated_letters, "At least two different letters are repeated")

# ==============================================================================
# EASY — Common vs Rare Letters
# ==============================================================================

_reg("common_letters_only", "Common Letters Only", LatentComplexity.EASY,
     wf.common_letters_only, "Uses only common letters: E,T,A,O,I,N,S,H,R,D,L,C,U")

_reg("has_rare_letter", "Has Rare Letter", LatentComplexity.EASY,
     wf.has_rare_letter, "Contains at least one rare letter: Q,Z,X,J,K,V,B,W,F,G,Y,P")

# ==============================================================================
# EASY — Unique Letter Count
# ==============================================================================

for n in [3, 4, 5]:
    _reg(f"unique_letters_{n}", f"Exactly {n} Unique Letters", LatentComplexity.EASY,
         lambda word, n=n: len(set(word.lower())) == n,
         f"Words with exactly {n} distinct letters")

# ==============================================================================
# EASY — Double Letter Patterns
# ==============================================================================

COMMON_DOUBLES = ['ee', 'll', 'ss', 'oo', 'tt', 'ff', 'rr', 'nn', 'pp', 'dd']
for double in COMMON_DOUBLES:
    letter = double[0].upper()
    _reg(f"double_{letter}", f"Contains Double {letter}", LatentComplexity.EASY,
         lambda word, d=double: d in word.lower(),
         f"Words containing double '{letter}' ({double})")

_reg("has_double_letter", "Has Any Double Letter", LatentComplexity.EASY,
     lambda word: any(word[i] == word[i+1] for i in range(len(word)-1)),
     "Words containing any consecutive double letter")


# ==============================================================================
# MEDIUM — Vowel Position Pattern
# ==============================================================================

_reg("vowel_odd_positions", "Vowels at Odd Positions", LatentComplexity.MEDIUM,
     wf.vowel_at_odd_positions, "Vowels appear at positions 1 and 3 (0-indexed)")

_reg("vowel_even_positions", "Vowels at Even Positions", LatentComplexity.MEDIUM,
     wf.vowel_at_even_positions, "Vowels appear at positions 0, 2, and 4")

_reg("vowels_clustered", "Vowels Clustered", LatentComplexity.MEDIUM,
     wf.vowels_clustered, "At least 2 adjacent vowels")

# ==============================================================================
# MEDIUM — Alphabet Range
# ==============================================================================

_reg("first_half_alphabet", "First Half Alphabet (A-M)", LatentComplexity.MEDIUM,
     wf.first_half_alphabet, "Only uses letters from A to M")

_reg("second_half_alphabet", "Second Half Alphabet (N-Z)", LatentComplexity.MEDIUM,
     wf.second_half_alphabet, "Only uses letters from N to Z")

_reg("mixed_balanced", "Mixed Balanced Alphabet", LatentComplexity.MEDIUM,
     wf.mixed_balanced, "At least 2 letters from each half of alphabet")

# ==============================================================================
# MEDIUM — Starting Pattern
# ==============================================================================

_reg("consonant_cluster_start", "Consonant Cluster Start", LatentComplexity.MEDIUM,
     wf.consonant_cluster_start, "Starts with consonant cluster (BR, ST, TR, etc.)")

_reg("vowel_start", "Vowel Start", LatentComplexity.MEDIUM,
     wf.starts_with_vowel, "Starts with a vowel")

_reg("single_consonant_start", "Single Consonant Start", LatentComplexity.MEDIUM,
     wf.single_consonant_start, "Starts with single consonant followed by vowel")

# Extended starting patterns
COMMON_STARTS = ['S', 'T', 'B', 'P', 'C', 'M', 'D', 'R']
for letter in COMMON_STARTS:
    if letter != 'T':  # T already covered by position_T_at_0
        _reg(f"starts_with_{letter}", f"Starts with '{letter}'", LatentComplexity.MEDIUM,
             wf.letter_at_position(letter, 0), f"First letter is always '{letter}'")

# ==============================================================================
# MEDIUM — Specific Letter at Position
# ==============================================================================

_reg("position_E_at_1", "Letter 'E' at Position 1", LatentComplexity.MEDIUM,
     wf.letter_at_position('E', 1), "Second letter (index 1) is always 'E'")

_reg("position_T_at_0", "Letter 'T' at Position 0", LatentComplexity.MEDIUM,
     wf.letter_at_position('T', 0), "First letter is always 'T'")

_reg("position_R_at_4", "Letter 'R' at Position 4", LatentComplexity.MEDIUM,
     wf.letter_at_position('R', 4), "Last letter (index 4) is always 'R'")

# Extended positional: 8 common letters × 5 positions (minus 3 already registered)
COMMON_LETTERS = ['E', 'T', 'A', 'O', 'I', 'N', 'S', 'R']
ALREADY_REGISTERED = {"position_E_at_1", "position_T_at_0", "position_R_at_4"}
for letter in COMMON_LETTERS:
    for pos in range(5):
        lid = f"position_{letter}_at_{pos}"
        if lid not in ALREADY_REGISTERED:
            _reg(lid, f"Letter '{letter}' at Position {pos}", LatentComplexity.MEDIUM,
                 wf.letter_at_position(letter, pos), f"Position {pos} is always '{letter}'")

# ==============================================================================
# MEDIUM — Consonant Cluster Patterns
# ==============================================================================

CONSONANT_CLUSTERS = [
    ('br', 'BR'), ('st', 'ST'), ('tr', 'TR'), ('gr', 'GR'), ('pr', 'PR'),
    ('ch', 'CH'), ('sh', 'SH'), ('th', 'TH'), ('bl', 'BL'), ('cl', 'CL'),
    ('fl', 'FL'), ('pl', 'PL'), ('sl', 'SL'), ('sp', 'SP'), ('sc', 'SC'),
]
for cluster, upper in CONSONANT_CLUSTERS:
    _reg(f"cluster_start_{upper}-cluster", f"Starts with {upper}", LatentComplexity.MEDIUM,
         lambda word, c=cluster: word.lower().startswith(c),
         f"Words starting with '{upper}' consonant cluster")

# ==============================================================================
# MEDIUM — Letter Pair Patterns
# ==============================================================================

LETTER_PAIRS = [
    ('ea', 'EA'), ('ou', 'OU'), ('ai', 'AI'), ('oo', 'OO'), ('ee', 'EE'),
    ('er', 'ER'), ('ar', 'AR'), ('or', 'OR'), ('ing', 'ING'), ('tion', 'TION'),
]
for pair, upper in LETTER_PAIRS:
    _reg(f"contains_{upper}", f"Contains {upper}", LatentComplexity.MEDIUM,
         lambda word, p=pair: p in word.lower(),
         f"Words containing '{upper}' letter sequence")

# ==============================================================================
# MEDIUM — Vowel-Consonant Patterns
# ==============================================================================

_reg("pattern_CVCVC", "CVCVC Pattern", LatentComplexity.MEDIUM,
     lambda word: (len(word) == 5 and wf.is_consonant(word[0]) and wf.is_vowel(word[1])
                   and wf.is_consonant(word[2]) and wf.is_vowel(word[3]) and wf.is_consonant(word[4])),
     "Consonant-Vowel alternating pattern: C-V-C-V-C")

_reg("pattern_CVCCV", "CVCCV Pattern", LatentComplexity.MEDIUM,
     lambda word: (len(word) == 5 and wf.is_consonant(word[0]) and wf.is_vowel(word[1])
                   and wf.is_consonant(word[2]) and wf.is_consonant(word[3]) and wf.is_vowel(word[4])),
     "Pattern: C-V-C-C-V")

_reg("pattern_VCCVC", "VCCVC Pattern", LatentComplexity.MEDIUM,
     lambda word: (len(word) == 5 and wf.is_vowel(word[0]) and wf.is_consonant(word[1])
                   and wf.is_consonant(word[2]) and wf.is_vowel(word[3]) and wf.is_consonant(word[4])),
     "Pattern: V-C-C-V-C")


# ==============================================================================
# HARD — Ending Patterns
# ==============================================================================

ENDINGS = [
    ('er', 'ER'), ('ly', 'LY'), ('ed', 'ED'),
    ('al', 'AL'), ('le', 'LE'), ('nt', 'NT'), ('st', 'ST'), ('ch', 'CH'),
    ('sh', 'SH'), ('th', 'TH'), ('ng', 'NG'), ('nd', 'ND'), ('se', 'SE'),
]
for pattern, upper in ENDINGS:
    _reg(f"ending_{upper}", f"Ending -{upper}", LatentComplexity.HARD,
         wf.ends_with_pattern(pattern), f"Words ending in '{upper}'")

# ==============================================================================
# HARD — Letter Frequency Distribution
# ==============================================================================

_reg("letter_freq_high", "High Frequency Letters", LatentComplexity.HARD,
     wf.high_frequency_word, "Average letter frequency > 7%")

_reg("letter_freq_medium", "Medium Frequency Letters", LatentComplexity.HARD,
     wf.medium_frequency_word, "Average letter frequency 4-7%")

_reg("letter_freq_low", "Low Frequency Letters", LatentComplexity.HARD,
     wf.low_frequency_word, "Average letter frequency < 4%")

# ==============================================================================
# HARD — Edit Distance Clustering
# ==============================================================================

REFERENCE_WORDS = ['crane', 'stale', 'proud', 'slate', 'raise', 'roast', 'tears']
for ref in REFERENCE_WORDS:
    _reg(f"near_{ref.upper()}", f"Near {ref.upper()} (distance ≤ 2)", LatentComplexity.HARD,
         wf.near_reference_word(ref, 2), f"Levenshtein distance ≤ 2 from '{ref.upper()}'")

# ==============================================================================
# HARD — Multi-Position Constraints
# ==============================================================================

_reg("multi_position_S_start_E_end", "Starts with 'S', Ends with 'E'", LatentComplexity.HARD,
     wf.multi_position_constraint([(0, 'S'), (4, 'E')]), "Position 0='S' AND Position 4='E'")

_reg("multi_position_R_at_1_N_at_3", "'R' at Position 1, 'N' at Position 3", LatentComplexity.HARD,
     wf.multi_position_constraint([(1, 'R'), (3, 'N')]), "Position 1='R' AND Position 3='N'")

_reg("multi_position_A_at_2", "'A' at Position 2 (Middle)", LatentComplexity.HARD,
     wf.letter_at_position('A', 2), "Middle letter (position 2) is always 'A'")

MULTI_POS = [
    ([(0, 'T'), (4, 'E')], "T_start_E_end", "Starts with 'T', Ends with 'E'"),
    ([(0, 'S'), (4, 'T')], "S_start_T_end", "Starts with 'S', Ends with 'T'"),
    ([(0, 'B'), (4, 'E')], "B_start_E_end", "Starts with 'B', Ends with 'E'"),
    ([(0, 'C'), (4, 'E')], "C_start_E_end", "Starts with 'C', Ends with 'E'"),
    ([(0, 'P'), (4, 'E')], "P_start_E_end", "Starts with 'P', Ends with 'E'"),
    ([(1, 'A'), (4, 'T')], "A_at_1_T_end", "'A' at Position 1, Ends with 'T'"),
    ([(1, 'E'), (4, 'R')], "E_at_1_R_end", "'E' at Position 1, Ends with 'R'"),
    ([(1, 'O'), (4, 'T')], "O_at_1_T_end", "'O' at Position 1, Ends with 'T'"),
    ([(2, 'A'), (4, 'E')], "A_at_2_E_end", "'A' at Position 2, Ends with 'E'"),
    ([(2, 'I'), (4, 'T')], "I_at_2_T_end", "'I' at Position 2, Ends with 'T'"),
    ([(2, 'O'), (4, 'T')], "O_at_2_T_end", "'O' at Position 2, Ends with 'T'"),
    ([(0, 'S'), (4, 'S')], "S_start_S_end", "Starts and Ends with 'S'"),
    ([(0, 'T'), (4, 'T')], "T_start_T_end", "Starts and Ends with 'T'"),
]
for constraints, suffix, desc in MULTI_POS:
    lid = f"multi_position_{suffix}"
    if lid != "multi_position_S_start_E_end":  # Already registered
        _reg(lid, desc, LatentComplexity.HARD,
             wf.multi_position_constraint(constraints), f"{desc} - Multiple positional constraint")

# ==============================================================================
# HARD — Semantic Categories
# ==============================================================================

SEMANTIC_CATEGORIES = {
    'fruits': ['apple', 'berry', 'grape', 'lemon', 'melon', 'peach', 'mango'],
    'body_parts': ['brain', 'chest', 'heart', 'thumb', 'ankle', 'elbow', 'wrist'],
    'colors': ['black', 'white', 'brown', 'green', 'coral', 'beige', 'azure'],
    'animals': ['horse', 'mouse', 'tiger', 'whale', 'snake', 'sheep', 'beast'],
    'food': ['bread', 'flour', 'grain', 'sugar', 'rice', 'beans', 'pasta'],
    'tools': ['brush', 'knife', 'drill', 'sword', 'screw', 'wrench', 'clamp'],
    'clothing': ['shirt', 'dress', 'scarf', 'apron', 'jeans', 'skirt', 'glove'],
    'nature': ['field', 'grass', 'plant', 'stone', 'water', 'earth', 'ocean'],
    'buildings': ['house', 'tower', 'cabin', 'store', 'hotel', 'lodge', 'shack'],
    'emotions': ['happy', 'angry', 'pride', 'shame', 'worry', 'guilt', 'grief'],
    'weather': ['storm', 'cloud', 'frost', 'flood', 'sunny', 'rainy', 'snowy'],
    'metals': ['brass', 'steel', 'bronze', 'iron', 'copper', 'silver', 'gold'],
}
for cat, words in SEMANTIC_CATEGORIES.items():
    word_set = set(w.lower() for w in words)
    _reg(f"category_{cat}", f"Category: {cat.replace('_', ' ').title()}", LatentComplexity.HARD,
         lambda word, ws=word_set: word.lower() in ws,
         f"Words in semantic category: {cat.replace('_', ' ')}")

# ==============================================================================
# HARD — Word Attributes
# ==============================================================================

WORD_ATTRIBUTES = {
    'actions': ['fight', 'drink', 'sleep', 'write', 'speak', 'dance', 'laugh', 'think', 'learn'],
    'abstract_concepts': ['truth', 'faith', 'peace', 'power', 'force', 'worth', 'value', 'pride'],
    'time_related': ['night', 'today', 'month', 'early', 'later', 'never', 'always'],
    'size_descriptors': ['large', 'small', 'giant', 'thick', 'broad', 'tight', 'loose'],
    'materials': ['cloth', 'glass', 'paper', 'stone', 'wood', 'metal', 'brick'],
    'quantities': ['whole', 'piece', 'share', 'total', 'bunch', 'group', 'pairs'],
}
for attr, words in WORD_ATTRIBUTES.items():
    word_set = set(w.lower() for w in words)
    _reg(f"attribute_{attr}", f"Attribute: {attr.replace('_', ' ').title()}", LatentComplexity.HARD,
         lambda word, ws=word_set: word.lower() in ws,
         f"Words with attribute: {attr.replace('_', ' ')}")
