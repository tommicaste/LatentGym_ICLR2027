"""Filter-based latent definitions for hangman.

All latents from the original textarena-multi-episode-with-latents hangman
implementation are registered here, organised by category:

1.  Word Length          (12 latents) — EASY
2.  Vowel Patterns       (11 latents) — EASY / MEDIUM
3.  Starting Letters     (15 latents) — MEDIUM
4.  Ending Patterns      (20 latents) — MEDIUM
5.  Letter Frequency      (9 latents) — MEDIUM
6.  Structural Patterns  (24 latents) — MEDIUM / HARD
7.  Semantic Categories  (10 latents) — HARD
8.  Difficulty            (4 latents) — EASY / MEDIUM / HARD
"""

from latentgym.core.latent import LatentDefinition, LatentComplexity
from latentgym.core.registry import register_latent
from . import word_filters as wf

# ══════════════════════════════════════════════════════════════════════════════
# 1. WORD LENGTH  (EASY) — 12 latents
# ══════════════════════════════════════════════════════════════════════════════

# Specific lengths: 3 through 10
for _n in range(3, 11):
    register_latent("hangman", LatentDefinition(
        id=f"length_{_n}",
        name=f"{_n}-Letter Words",
        complexity=LatentComplexity.EASY,
        description=f"Words with exactly {_n} letters",
        filter_fn=lambda w, l=_n: wf.filter_by_length(w, l),
    ))

# 11+ letter words
register_latent("hangman", LatentDefinition(
    id="length_11_plus",
    name="11+ Letter Words",
    complexity=LatentComplexity.EASY,
    description="Words with 11 or more letters",
    filter_fn=lambda w: len(w) >= 11,
))

# Length categories
register_latent("hangman", LatentDefinition(
    id="length_short",
    name="Short Words (3-5)",
    complexity=LatentComplexity.EASY,
    description="Short words with 3-5 letters",
    filter_fn=wf.length_short,
))

register_latent("hangman", LatentDefinition(
    id="length_medium",
    name="Medium Words (6-8)",
    complexity=LatentComplexity.EASY,
    description="Medium-length words with 6-8 letters",
    filter_fn=wf.length_medium,
))

register_latent("hangman", LatentDefinition(
    id="length_long",
    name="Long Words (9+)",
    complexity=LatentComplexity.EASY,
    description="Long words with 9 or more letters",
    filter_fn=wf.length_long,
))

# ══════════════════════════════════════════════════════════════════════════════
# 2. VOWEL PATTERNS — 11 latents
# ══════════════════════════════════════════════════════════════════════════════

# Exact vowel counts: 0 through 5  (EASY)
for _c in range(0, 6):
    register_latent("hangman", LatentDefinition(
        id=f"vowel_count_{_c}",
        name=f"{_c} Vowels",
        complexity=LatentComplexity.EASY,
        description=f"Words with exactly {_c} vowels",
        filter_fn=lambda w, c=_c: wf.filter_by_vowel_count(w, c),
    ))

# 6+ vowels  (EASY)
register_latent("hangman", LatentDefinition(
    id="vowel_count_6_plus",
    name="6+ Vowels",
    complexity=LatentComplexity.EASY,
    description="Words with 6 or more vowels",
    filter_fn=lambda w: wf.filter_by_min_vowels(w, 6),
))

# Vowel ratios  (EASY)
register_latent("hangman", LatentDefinition(
    id="vowel_ratio_low",
    name="Low Vowel Ratio (<25%)",
    complexity=LatentComplexity.EASY,
    description="Words where less than 25% of letters are vowels",
    filter_fn=wf.filter_vowel_ratio_low,
))

register_latent("hangman", LatentDefinition(
    id="vowel_ratio_high",
    name="High Vowel Ratio (>45%)",
    complexity=LatentComplexity.EASY,
    description="Words where more than 45% of letters are vowels",
    filter_fn=wf.filter_vowel_ratio_high,
))

# Vowel / consonant balance  (MEDIUM)
register_latent("hangman", LatentDefinition(
    id="vowel_heavy",
    name="Vowel-Heavy Words",
    complexity=LatentComplexity.MEDIUM,
    description="Words with more vowels than consonants",
    filter_fn=wf.vowel_heavy,
))

register_latent("hangman", LatentDefinition(
    id="consonant_heavy",
    name="Consonant-Heavy Words",
    complexity=LatentComplexity.MEDIUM,
    description="Words with more consonants than vowels",
    filter_fn=wf.consonant_heavy,
))

# ══════════════════════════════════════════════════════════════════════════════
# 3. STARTING LETTERS  (MEDIUM) — 15 latents
# ══════════════════════════════════════════════════════════════════════════════

for _letter in ['S', 'C', 'P', 'B', 'M', 'T', 'A', 'R', 'D', 'F', 'H', 'L', 'W', 'G', 'N']:
    register_latent("hangman", LatentDefinition(
        id=f"starts_with_{_letter}",
        name=f"Starts with {_letter}",
        complexity=LatentComplexity.MEDIUM,
        description=f"Words starting with the letter {_letter}",
        filter_fn=lambda w, l=_letter: wf.filter_by_starting_letter(w, l),
    ))

# ══════════════════════════════════════════════════════════════════════════════
# 4. ENDING PATTERNS  (MEDIUM) — 20 latents
# ══════════════════════════════════════════════════════════════════════════════

# Single-letter endings
for _letter in ['E', 'S', 'D', 'Y', 'N', 'T', 'R', 'L', 'G']:
    register_latent("hangman", LatentDefinition(
        id=f"ends_with_{_letter}",
        name=f"Ends with {_letter}",
        complexity=LatentComplexity.MEDIUM,
        description=f"Words ending with the letter {_letter}",
        filter_fn=lambda w, l=_letter: wf.filter_by_ending_letter(w, l),
    ))

# Common suffixes
for _suffix in ['ING', 'TION', 'LY', 'ED', 'ER', 'EST', 'NESS', 'MENT', 'ABLE', 'LESS', 'FUL']:
    register_latent("hangman", LatentDefinition(
        id=f"ending_{_suffix}",
        name=f"Ends with -{_suffix}",
        complexity=LatentComplexity.MEDIUM,
        description=f"Words ending with the suffix -{_suffix}",
        filter_fn=lambda w, s=_suffix: wf.filter_by_ending(w, s),
    ))

# ══════════════════════════════════════════════════════════════════════════════
# 5. LETTER FREQUENCY  (MEDIUM) — 9 latents
# ══════════════════════════════════════════════════════════════════════════════

register_latent("hangman", LatentDefinition(
    id="common_letters_only",
    name="Common Letters Only",
    complexity=LatentComplexity.MEDIUM,
    description="Words containing only common letters (E, T, A, O, I, N, S, H, R, D, L, U)",
    filter_fn=wf.common_letters_only,
))

register_latent("hangman", LatentDefinition(
    id="has_rare_letter",
    name="Has Rare Letter",
    complexity=LatentComplexity.MEDIUM,
    description="Words containing rare letters (Q, X, Z, or J)",
    filter_fn=wf.has_rare_letter,
))

# Specific rare letters
for _letter in ['Q', 'X', 'Z', 'J']:
    register_latent("hangman", LatentDefinition(
        id=f"contains_{_letter}",
        name=f"Contains {_letter}",
        complexity=LatentComplexity.MEDIUM,
        description=f"Words containing the letter {_letter}",
        filter_fn=lambda w, l=_letter: wf.filter_by_contains_letter(w, l),
    ))

# No E (most common letter)
register_latent("hangman", LatentDefinition(
    id="no_E",
    name="No Letter E",
    complexity=LatentComplexity.MEDIUM,
    description="Words that do not contain the letter E",
    filter_fn=lambda w: wf.filter_by_not_contains_letter(w, 'E'),
))

# Frequency scores
register_latent("hangman", LatentDefinition(
    id="high_frequency_score",
    name="High Frequency Letters",
    complexity=LatentComplexity.MEDIUM,
    description="Words with high average letter frequency (score > 6.0)",
    filter_fn=lambda w: wf.filter_by_letter_frequency(w, 6.0, above=True),
))

register_latent("hangman", LatentDefinition(
    id="low_frequency_score",
    name="Low Frequency Letters",
    complexity=LatentComplexity.MEDIUM,
    description="Words with low average letter frequency (score < 4.0)",
    filter_fn=lambda w: wf.filter_by_letter_frequency(w, 4.0, above=False),
))

# ══════════════════════════════════════════════════════════════════════════════
# 6. STRUCTURAL PATTERNS — 24 latents
# ══════════════════════════════════════════════════════════════════════════════

# Repeated letters  (MEDIUM)
register_latent("hangman", LatentDefinition(
    id="no_repeated_letters",
    name="No Repeated Letters",
    complexity=LatentComplexity.MEDIUM,
    description="Words with all unique letters (no letter appears twice)",
    filter_fn=wf.no_repeated_letters,
))

register_latent("hangman", LatentDefinition(
    id="has_repeated_letters",
    name="Has Repeated Letters",
    complexity=LatentComplexity.MEDIUM,
    description="Words where at least one letter appears more than once",
    filter_fn=wf.filter_by_has_repeated_letters,
))

# Double letters  (MEDIUM)
register_latent("hangman", LatentDefinition(
    id="has_double_letter",
    name="Has Double Letter",
    complexity=LatentComplexity.MEDIUM,
    description="Words with consecutive identical letters (e.g., LL, EE, SS)",
    filter_fn=wf.filter_by_has_double_letter,
))

# Specific double letters  (HARD)
for _letter in ['E', 'L', 'S', 'O', 'T', 'R', 'N', 'P', 'F', 'M']:
    register_latent("hangman", LatentDefinition(
        id=f"double_{_letter}",
        name=f"Has Double {_letter}",
        complexity=LatentComplexity.HARD,
        description=f"Words containing the double letter {_letter}{_letter}",
        filter_fn=lambda w, l=_letter: wf.filter_by_specific_double(w, l),
    ))

# Consonant clusters
register_latent("hangman", LatentDefinition(
    id="has_start_cluster",
    name="Starts with Consonant Cluster",
    complexity=LatentComplexity.MEDIUM,
    description="Words starting with a consonant cluster (BL, CH, ST, etc.)",
    filter_fn=wf.filter_has_start_cluster,
))

# Specific consonant clusters  (HARD)
for _cluster in ['CH', 'SH', 'TH', 'ST', 'BR', 'TR', 'PR', 'GR', 'CR', 'SP']:
    register_latent("hangman", LatentDefinition(
        id=f"cluster_{_cluster.lower()}",
        name=f"Starts with {_cluster}-",
        complexity=LatentComplexity.HARD,
        description=f"Words starting with the consonant cluster {_cluster}",
        filter_fn=lambda w, c=_cluster: wf.filter_by_start_cluster(w, c),
    ))

# ══════════════════════════════════════════════════════════════════════════════
# 7. SEMANTIC CATEGORIES  (HARD) — 10 latents
# ══════════════════════════════════════════════════════════════════════════════

for _category in [
    'animals', 'colors', 'food', 'body_parts', 'nature',
    'actions', 'objects', 'places', 'emotions', 'weather',
]:
    register_latent("hangman", LatentDefinition(
        id=f"category_{_category}",
        name=f"Category: {_category.replace('_', ' ').title()}",
        complexity=LatentComplexity.HARD,
        description=f"Words belonging to the semantic category: {_category}",
        filter_fn=lambda w, cat=_category: wf.filter_by_category(w, cat),
    ))

# ══════════════════════════════════════════════════════════════════════════════
# 8. DIFFICULTY  — 4 latents
# ══════════════════════════════════════════════════════════════════════════════

register_latent("hangman", LatentDefinition(
    id="easy_word",
    name="Easy Words",
    complexity=LatentComplexity.EASY,
    description="Easy words: 4-6 letters, common letters, high frequency",
    filter_fn=wf.filter_easy_word,
))

register_latent("hangman", LatentDefinition(
    id="medium_word",
    name="Medium Words",
    complexity=LatentComplexity.MEDIUM,
    description="Medium difficulty words: 5-8 letters, moderate frequency",
    filter_fn=wf.filter_medium_word,
))

register_latent("hangman", LatentDefinition(
    id="hard_word",
    name="Hard Words",
    complexity=LatentComplexity.HARD,
    description="Hard words: 7+ letters, low frequency letters",
    filter_fn=wf.filter_hard_word,
))

register_latent("hangman", LatentDefinition(
    id="all_words",
    name="All Words",
    complexity=LatentComplexity.EASY,
    description="All words (no constraint)",
    filter_fn=lambda w: True,
))
