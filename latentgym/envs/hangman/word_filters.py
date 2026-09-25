"""Word filter functions for hangman latents.

Each filter function takes a word (str) and returns True if it matches the
constraint.  Helper factories return a Callable[[str], bool] for parameterised
filters so they integrate cleanly with LatentDefinition.filter_fn.
"""

from typing import Callable, List, Optional, Set

# =============================================================================
# Constants
# =============================================================================

VOWELS = set("AEIOU")
CONSONANTS = set("BCDFGHJKLMNPQRSTVWXYZ")
COMMON_LETTERS = set("ETAOINSHRDLU")  # Most frequent English letters
RARE_LETTERS = set("QXZJ")

# Letter frequency scores (higher = more common)
LETTER_FREQUENCY = {
    'E': 12.7, 'T': 9.1, 'A': 8.2, 'O': 7.5, 'I': 7.0, 'N': 6.7, 'S': 6.3,
    'H': 6.1, 'R': 6.0, 'D': 4.3, 'L': 4.0, 'C': 2.8, 'U': 2.8, 'M': 2.4,
    'W': 2.4, 'F': 2.2, 'G': 2.0, 'Y': 2.0, 'P': 1.9, 'B': 1.5, 'V': 1.0,
    'K': 0.8, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07,
}

# Common word endings
COMMON_ENDINGS = {
    'ING', 'TION', 'SION', 'LY', 'ED', 'ER', 'EST', 'NESS', 'MENT',
    'ABLE', 'IBLE', 'LESS', 'FUL', 'IOUS', 'EOUS', 'AL', 'EN', 'ISH',
    'IVE', 'OUS', 'URE', 'ANT', 'ENT', 'ARY', 'ORY',
}

# Consonant clusters that can start words
START_CLUSTERS = {
    'BL', 'BR', 'CH', 'CL', 'CR', 'DR', 'FL', 'FR', 'GL', 'GR',
    'PL', 'PR', 'SC', 'SH', 'SK', 'SL', 'SM', 'SN', 'SP', 'ST',
    'SW', 'TH', 'TR', 'TW', 'WH', 'WR', 'SCR', 'SPR', 'STR', 'THR',
}

# Semantic categories (curated word lists)
SEMANTIC_CATEGORIES = {
    'animals': {
        'cat', 'dog', 'bird', 'fish', 'lion', 'tiger', 'bear', 'wolf', 'deer',
        'mouse', 'rat', 'horse', 'cow', 'pig', 'sheep', 'goat', 'chicken',
        'duck', 'eagle', 'hawk', 'owl', 'snake', 'frog', 'turtle', 'rabbit',
        'fox', 'elephant', 'monkey', 'gorilla', 'whale', 'shark', 'dolphin',
        'seal', 'penguin', 'zebra', 'giraffe', 'hippo', 'rhino', 'camel',
        'kangaroo', 'koala', 'panda', 'leopard', 'cheetah', 'jaguar', 'panther',
        'squirrel', 'beaver', 'otter', 'badger', 'skunk', 'raccoon', 'moose',
        'buffalo', 'bison', 'antelope', 'gazelle', 'llama', 'alpaca', 'donkey',
        'mule', 'pony', 'colt', 'foal', 'stallion', 'mare', 'bull', 'calf',
        'lamb', 'ram', 'ewe', 'kid', 'piglet', 'chick', 'duckling', 'gosling',
        'cub', 'pup', 'kitten', 'puppy', 'bunny', 'hamster', 'gerbil', 'parrot',
        'canary', 'sparrow', 'robin', 'crow', 'raven', 'pigeon', 'dove', 'swan',
        'goose', 'turkey', 'peacock', 'flamingo', 'pelican', 'stork', 'heron',
        'crane', 'vulture', 'condor', 'falcon', 'kite', 'osprey', 'bat', 'moth',
        'butterfly', 'bee', 'wasp', 'ant', 'spider', 'scorpion', 'crab', 'lobster',
        'shrimp', 'clam', 'oyster', 'snail', 'slug', 'worm', 'caterpillar',
    },
    'colors': {
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown',
        'black', 'white', 'gray', 'grey', 'gold', 'silver', 'bronze', 'copper',
        'crimson', 'scarlet', 'maroon', 'burgundy', 'rose', 'coral', 'salmon',
        'peach', 'apricot', 'amber', 'tan', 'beige', 'cream', 'ivory', 'khaki',
        'olive', 'lime', 'emerald', 'jade', 'teal', 'cyan', 'aqua', 'turquoise',
        'navy', 'indigo', 'violet', 'lavender', 'magenta', 'fuchsia', 'plum',
        'mauve', 'lilac', 'orchid', 'periwinkle', 'azure', 'cobalt', 'sapphire',
        'charcoal', 'slate', 'ash', 'pearl', 'champagne', 'rust', 'auburn',
    },
    'food': {
        'bread', 'rice', 'pasta', 'meat', 'fish', 'chicken', 'beef', 'pork',
        'lamb', 'bacon', 'ham', 'sausage', 'egg', 'cheese', 'butter', 'milk',
        'cream', 'yogurt', 'soup', 'salad', 'sandwich', 'pizza', 'burger',
        'steak', 'roast', 'grill', 'fry', 'bake', 'cake', 'pie', 'cookie',
        'toast', 'muffin', 'donut', 'bagel', 'croissant', 'waffle',
        'pancake', 'cereal', 'oatmeal', 'fruit', 'apple', 'banana', 'orange',
        'grape', 'berry', 'melon', 'peach', 'pear', 'plum', 'cherry', 'mango',
        'lemon', 'lime', 'coconut', 'pineapple', 'strawberry', 'blueberry',
        'raspberry', 'vegetable', 'carrot', 'potato', 'tomato', 'onion',
        'garlic', 'pepper', 'corn', 'bean', 'pea', 'spinach', 'lettuce',
        'cabbage', 'broccoli', 'cauliflower', 'celery', 'cucumber', 'squash',
        'pumpkin', 'mushroom', 'olive', 'pickle', 'sauce', 'gravy', 'ketchup',
        'mustard', 'mayo', 'salt', 'sugar', 'honey', 'syrup', 'jam',
        'jelly', 'peanut', 'almond', 'walnut', 'cashew', 'hazelnut', 'pecan',
    },
    'body_parts': {
        'head', 'face', 'eye', 'ear', 'nose', 'mouth', 'lip', 'tooth', 'teeth',
        'tongue', 'chin', 'cheek', 'jaw', 'neck', 'throat', 'shoulder', 'arm',
        'elbow', 'wrist', 'hand', 'finger', 'thumb', 'nail', 'palm', 'fist',
        'chest', 'breast', 'rib', 'back', 'spine', 'hip', 'waist', 'belly',
        'stomach', 'leg', 'thigh', 'knee', 'calf', 'ankle', 'foot', 'feet',
        'toe', 'heel', 'sole', 'skin', 'bone', 'muscle', 'blood', 'vein',
        'artery', 'heart', 'lung', 'liver', 'kidney', 'brain', 'nerve',
        'hair', 'scalp', 'forehead', 'eyebrow', 'eyelash', 'eyelid', 'pupil',
        'nostril', 'gum', 'tonsil', 'adam',
    },
    'nature': {
        'tree', 'flower', 'grass', 'leaf', 'branch', 'root', 'trunk', 'bark',
        'seed', 'fruit', 'bush', 'shrub', 'vine', 'moss', 'fern', 'weed',
        'plant', 'forest', 'jungle', 'wood', 'grove', 'meadow', 'field',
        'garden', 'park', 'lawn', 'pond', 'lake', 'river', 'stream', 'creek',
        'brook', 'spring', 'waterfall', 'ocean', 'sea', 'beach', 'shore',
        'coast', 'island', 'bay', 'gulf', 'cove', 'reef', 'wave', 'tide',
        'current', 'mountain', 'hill', 'valley', 'canyon', 'cliff', 'cave',
        'rock', 'stone', 'boulder', 'pebble', 'sand', 'mud', 'dirt', 'soil',
        'clay', 'dust', 'ash', 'volcano', 'lava', 'glacier', 'iceberg', 'snow',
        'ice', 'frost', 'rain', 'storm', 'thunder', 'lightning', 'wind',
        'breeze', 'gust', 'tornado', 'hurricane', 'cloud', 'fog', 'mist',
        'dew', 'rainbow', 'sun', 'moon', 'star', 'sky', 'dawn', 'dusk',
        'sunrise', 'sunset', 'twilight', 'midnight', 'noon',
    },
    'actions': {
        'run', 'walk', 'jump', 'hop', 'skip', 'leap', 'climb', 'crawl', 'swim',
        'fly', 'fall', 'roll', 'spin', 'turn', 'twist', 'bend', 'stretch',
        'reach', 'grab', 'hold', 'drop', 'throw', 'catch', 'push', 'pull',
        'lift', 'carry', 'drag', 'slide', 'kick', 'hit', 'punch', 'slap',
        'tap', 'knock', 'shake', 'wave', 'point', 'touch', 'feel', 'rub',
        'scratch', 'squeeze', 'press', 'crush', 'break', 'tear', 'cut',
        'slice', 'chop', 'stab', 'poke', 'dig', 'bury', 'plant', 'grow',
        'bloom', 'wilt', 'die', 'live', 'breathe', 'eat', 'drink', 'chew',
        'swallow', 'taste', 'smell', 'see', 'look', 'watch', 'hear', 'listen',
        'speak', 'talk', 'say', 'tell', 'ask', 'answer', 'shout', 'yell',
        'scream', 'whisper', 'sing', 'hum', 'whistle', 'laugh', 'cry', 'smile',
        'frown', 'wink', 'blink', 'stare', 'glare', 'sleep', 'wake', 'dream',
        'think', 'know', 'learn', 'teach', 'read', 'write', 'draw', 'paint',
        'build', 'make', 'create', 'destroy', 'fix', 'repair', 'clean', 'wash',
    },
    'objects': {
        'table', 'chair', 'desk', 'bed', 'sofa', 'couch', 'lamp', 'light',
        'door', 'window', 'wall', 'floor', 'ceiling', 'roof', 'room', 'house',
        'building', 'tower', 'bridge', 'road', 'street', 'path', 'stairs',
        'elevator', 'car', 'truck', 'bus', 'train', 'plane', 'boat', 'ship',
        'bicycle', 'motorcycle', 'wheel', 'engine', 'phone', 'computer',
        'screen', 'keyboard', 'mouse', 'camera', 'television', 'radio',
        'clock', 'watch', 'mirror', 'picture', 'frame', 'book', 'paper',
        'pen', 'pencil', 'eraser', 'ruler', 'scissors', 'knife', 'fork',
        'spoon', 'plate', 'bowl', 'cup', 'glass', 'bottle', 'jar', 'box',
        'bag', 'basket', 'bucket', 'bin', 'can', 'pot', 'pan', 'oven',
        'stove', 'refrigerator', 'sink', 'faucet', 'toilet', 'shower', 'bath',
        'towel', 'soap', 'brush', 'comb', 'toothbrush', 'razor', 'blanket',
        'pillow', 'sheet', 'curtain', 'carpet', 'rug', 'mat', 'umbrella',
        'key', 'lock', 'chain', 'rope', 'wire', 'cable', 'plug', 'socket',
    },
    'places': {
        'home', 'house', 'apartment', 'room', 'kitchen', 'bathroom', 'bedroom',
        'office', 'school', 'college', 'university', 'library', 'museum',
        'theater', 'cinema', 'stadium', 'arena', 'gym', 'pool', 'park',
        'garden', 'zoo', 'farm', 'ranch', 'factory', 'warehouse', 'store',
        'shop', 'market', 'mall', 'restaurant', 'cafe', 'bar', 'hotel',
        'motel', 'hospital', 'clinic', 'pharmacy', 'bank', 'post', 'station',
        'airport', 'port', 'harbor', 'dock', 'pier', 'bridge', 'tunnel',
        'highway', 'road', 'street', 'avenue', 'alley', 'lane', 'path',
        'trail', 'beach', 'shore', 'coast', 'island', 'mountain', 'hill',
        'valley', 'forest', 'jungle', 'desert', 'plain', 'prairie', 'meadow',
        'swamp', 'marsh', 'river', 'lake', 'pond', 'ocean', 'sea', 'bay',
        'city', 'town', 'village', 'suburb', 'downtown', 'country', 'nation',
        'state', 'province', 'region', 'district', 'neighborhood', 'block',
    },
    'emotions': {
        'happy', 'sad', 'angry', 'mad', 'upset', 'calm', 'peaceful', 'relaxed',
        'stressed', 'anxious', 'nervous', 'worried', 'scared', 'afraid',
        'frightened', 'terrified', 'brave', 'bold', 'confident', 'proud',
        'ashamed', 'guilty', 'sorry', 'grateful', 'thankful', 'hopeful',
        'hopeless', 'desperate', 'lonely', 'alone', 'loved', 'loving', 'kind',
        'gentle', 'tender', 'warm', 'cold', 'cruel', 'mean', 'nice', 'sweet',
        'bitter', 'jealous', 'envious', 'greedy', 'selfish', 'generous',
        'humble', 'arrogant', 'shy', 'excited', 'bored', 'curious',
        'surprised', 'amazed', 'shocked', 'confused', 'puzzled', 'certain',
        'doubtful', 'suspicious', 'trusting', 'loyal', 'faithful', 'devoted',
    },
    'weather': {
        'sun', 'sunny', 'cloud', 'cloudy', 'rain', 'rainy', 'snow', 'snowy',
        'wind', 'windy', 'storm', 'stormy', 'thunder', 'lightning', 'fog',
        'foggy', 'mist', 'misty', 'haze', 'hazy', 'frost', 'frosty', 'ice',
        'icy', 'sleet', 'hail', 'drizzle', 'shower', 'downpour', 'flood',
        'drought', 'heat', 'cold', 'warm', 'cool', 'hot', 'freezing', 'mild',
        'humid', 'dry', 'wet', 'damp', 'breeze', 'gust', 'gale', 'tornado',
        'hurricane', 'typhoon', 'cyclone', 'blizzard', 'rainbow', 'dew',
        'temperature', 'degree', 'climate', 'season', 'spring', 'summer',
        'autumn', 'fall', 'winter',
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def count_vowels(word: str) -> int:
    """Count the number of vowels in a word."""
    return sum(1 for c in word.upper() if c in VOWELS)


def count_consonants(word: str) -> int:
    """Count the number of consonants in a word."""
    return sum(1 for c in word.upper() if c in CONSONANTS)


def get_vowel_ratio(word: str) -> float:
    """Get the ratio of vowels to total letters."""
    if not word:
        return 0.0
    alpha_chars = [c for c in word.upper() if c.isalpha()]
    if not alpha_chars:
        return 0.0
    return count_vowels(word) / len(alpha_chars)


def has_repeated_letters(word: str) -> bool:
    """Check if word has any repeated letters."""
    upper = word.upper()
    return len(upper) != len(set(upper))


def has_double_letter(word: str) -> bool:
    """Check if word has consecutive identical letters (e.g., 'LL' in 'hello')."""
    upper = word.upper()
    for i in range(len(upper) - 1):
        if upper[i] == upper[i + 1] and upper[i].isalpha():
            return True
    return False


def get_double_letters(word: str) -> Set[str]:
    """Get all double letters in a word."""
    upper = word.upper()
    doubles: Set[str] = set()
    for i in range(len(upper) - 1):
        if upper[i] == upper[i + 1] and upper[i].isalpha():
            doubles.add(upper[i])
    return doubles


def get_letter_frequency_score(word: str) -> float:
    """Calculate average letter frequency score for a word."""
    upper = word.upper()
    alpha_chars = [c for c in upper if c.isalpha()]
    if not alpha_chars:
        return 0.0
    return sum(LETTER_FREQUENCY.get(c, 0) for c in alpha_chars) / len(alpha_chars)


def contains_only_common_letters(word: str) -> bool:
    """Check if word contains only common letters (ETAOINSHRDLU)."""
    upper = word.upper()
    return all(c in COMMON_LETTERS for c in upper if c.isalpha())


def contains_rare_letter(word: str) -> bool:
    """Check if word contains rare letters (Q, X, Z, J)."""
    upper = word.upper()
    return any(c in RARE_LETTERS for c in upper if c.isalpha())


def get_consonant_cluster_at_start(word: str) -> Optional[str]:
    """Get the consonant cluster at the start of the word, if any."""
    upper = word.upper()
    if not upper:
        return None
    # Check for 3-letter clusters first
    if len(upper) >= 3 and upper[:3] in START_CLUSTERS:
        return upper[:3]
    # Then 2-letter clusters
    if len(upper) >= 2 and upper[:2] in START_CLUSTERS:
        return upper[:2]
    return None


def get_ending(word: str, length: int = 3) -> str:
    """Get the ending of a word."""
    return word.upper()[-length:] if len(word) >= length else word.upper()


# =============================================================================
# Filter Functions — Word Length
# =============================================================================

def has_length(n: int) -> Callable[[str], bool]:
    """Return a filter that matches words of exactly n letters."""
    return lambda w: len(w) == n


def filter_by_length(word: str, length: int) -> bool:
    """Check if word has exactly the specified length."""
    return len(word) == length


def filter_by_min_length(word: str, min_len: int) -> bool:
    """Check if word has at least the specified length."""
    return len(word) >= min_len


def filter_by_max_length(word: str, max_len: int) -> bool:
    """Check if word has at most the specified length."""
    return len(word) <= max_len


def filter_by_length_range(word: str, min_len: int, max_len: int) -> bool:
    """Check if word length is within the specified range."""
    return min_len <= len(word) <= max_len


def length_short(word: str) -> bool:
    """Check if word is short (3-5 letters)."""
    return 3 <= len(word) <= 5


def length_medium(word: str) -> bool:
    """Check if word is medium length (6-8 letters)."""
    return 6 <= len(word) <= 8


def length_long(word: str) -> bool:
    """Check if word is long (9+ letters)."""
    return len(word) >= 9


# keep original aliases used by existing latents.py
filter_length_short = length_short
filter_length_medium = length_medium
filter_length_long = length_long


# =============================================================================
# Filter Functions — Vowels
# =============================================================================

def has_n_vowels(n: int) -> Callable[[str], bool]:
    """Return a filter that matches words with exactly n vowels."""
    return lambda w: count_vowels(w) == n


def filter_by_vowel_count(word: str, count: int) -> bool:
    """Check if word has exactly the specified number of vowels."""
    return count_vowels(word) == count


def filter_by_min_vowels(word: str, min_count: int) -> bool:
    """Check if word has at least the specified number of vowels."""
    return count_vowels(word) >= min_count


def filter_vowel_ratio_low(word: str) -> bool:
    """Check if word has low vowel ratio (<25%)."""
    return get_vowel_ratio(word) < 0.25


def filter_vowel_ratio_high(word: str) -> bool:
    """Check if word has high vowel ratio (>45%)."""
    return get_vowel_ratio(word) > 0.45


def vowel_heavy(word: str) -> bool:
    """Check if word has more vowels than consonants."""
    return count_vowels(word) > count_consonants(word)


def consonant_heavy(word: str) -> bool:
    """Check if word has more consonants than vowels."""
    return count_consonants(word) > count_vowels(word)


# original-style aliases
filter_vowel_heavy = vowel_heavy
filter_consonant_heavy = consonant_heavy


# =============================================================================
# Filter Functions — Starting / Ending Letters
# =============================================================================

def starts_with(letter: str) -> Callable[[str], bool]:
    """Return a filter that matches words starting with the given letter."""
    return lambda w: w.upper().startswith(letter.upper())


def ends_with(letter_or_suffix: str) -> Callable[[str], bool]:
    """Return a filter that matches words ending with the given letter/suffix."""
    return lambda w: w.upper().endswith(letter_or_suffix.upper())


def filter_by_starting_letter(word: str, letter: str) -> bool:
    """Check if word starts with the specified letter."""
    return word.upper().startswith(letter.upper())


def filter_by_ending_letter(word: str, letter: str) -> bool:
    """Check if word ends with the specified letter."""
    return word.upper().endswith(letter.upper())


def filter_by_ending(word: str, ending: str) -> bool:
    """Check if word ends with the specified suffix."""
    return word.upper().endswith(ending.upper())


def filter_by_contains_letter(word: str, letter: str) -> bool:
    """Check if word contains the specified letter."""
    return letter.upper() in word.upper()


def filter_by_not_contains_letter(word: str, letter: str) -> bool:
    """Check if word does NOT contain the specified letter."""
    return letter.upper() not in word.upper()


def contains_letter(letter: str) -> Callable[[str], bool]:
    """Return a filter that matches words containing the given letter."""
    return lambda w: letter.upper() in w.upper()


# Named suffix filters kept for backwards-compatibility
def ends_with_ing(w: str) -> bool:
    return w.upper().endswith("ING")


def ends_with_tion(w: str) -> bool:
    return w.upper().endswith("TION")


def ends_with_ed(w: str) -> bool:
    return w.upper().endswith("ED")


def ends_with_ly(w: str) -> bool:
    return w.upper().endswith("LY")


# =============================================================================
# Filter Functions — Letter Patterns
# =============================================================================

def no_repeated_letters(word: str) -> bool:
    """Check if word has no repeated letters."""
    upper = word.upper()
    return len(upper) == len(set(upper))


def filter_by_no_repeated_letters(word: str) -> bool:
    """Check if word has no repeated letters."""
    return no_repeated_letters(word)


def filter_by_has_repeated_letters(word: str) -> bool:
    """Check if word has repeated letters."""
    return has_repeated_letters(word)


def filter_by_has_double_letter(word: str) -> bool:
    """Check if word has a double letter (consecutive identical letters)."""
    return has_double_letter(word)


def filter_by_specific_double(word: str, letter: str) -> bool:
    """Check if word has a specific double letter."""
    double = letter.upper() * 2
    return double in word.upper()


def filter_by_letter_frequency(word: str, threshold: float, above: bool = True) -> bool:
    """Check if word's average letter frequency is above/below threshold."""
    score = get_letter_frequency_score(word)
    return score >= threshold if above else score < threshold


def common_letters_only(word: str) -> bool:
    """Check if word contains only common letters (ETAOINSHRDLU)."""
    return contains_only_common_letters(word)


def has_rare_letter(word: str) -> bool:
    """Check if word contains a rare letter (Q, X, Z, J)."""
    return contains_rare_letter(word)


# original-style aliases
filter_common_letters_only = common_letters_only
filter_has_rare_letter = has_rare_letter


# =============================================================================
# Filter Functions — Consonant Clusters
# =============================================================================

def filter_by_start_cluster(word: str, cluster: str) -> bool:
    """Check if word starts with the specified consonant cluster."""
    return word.upper().startswith(cluster.upper())


def filter_has_start_cluster(word: str) -> bool:
    """Check if word starts with any consonant cluster."""
    return get_consonant_cluster_at_start(word) is not None


# =============================================================================
# Filter Functions — Semantic Categories
# =============================================================================

def filter_by_category(word: str, category: str) -> bool:
    """Check if word belongs to the specified semantic category."""
    if category not in SEMANTIC_CATEGORIES:
        return False
    return word.lower() in SEMANTIC_CATEGORIES[category]


def get_available_categories() -> List[str]:
    """Get list of available semantic categories."""
    return list(SEMANTIC_CATEGORIES.keys())


# =============================================================================
# Filter Functions — Word Difficulty
# =============================================================================

def filter_easy_word(word: str) -> bool:
    """
    Check if word is 'easy' - short, common letters, no unusual patterns.
    Easy: 4-6 letters, high frequency score, no rare letters.
    """
    if not (4 <= len(word) <= 6):
        return False
    if contains_rare_letter(word):
        return False
    if get_letter_frequency_score(word) < 5.0:
        return False
    return True


def filter_medium_word(word: str) -> bool:
    """
    Check if word is 'medium' difficulty.
    Medium: 5-8 letters, moderate frequency, may have doubles.
    """
    if not (5 <= len(word) <= 8):
        return False
    freq = get_letter_frequency_score(word)
    if freq < 3.0 or freq > 7.0:
        return False
    return True


def filter_hard_word(word: str) -> bool:
    """
    Check if word is 'hard' - longer, less common letters, unusual patterns.
    Hard: 7+ letters, low frequency, may have rare letters.
    """
    if len(word) < 7:
        return False
    if get_letter_frequency_score(word) > 5.0:
        return False
    return True
