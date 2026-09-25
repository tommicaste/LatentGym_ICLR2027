# Hangman

Reveal a hidden word by guessing letters (or the whole word) before the turn budget runs out.
Across episodes the target words satisfy a latent property, so an agent that infers the property
can prioritize the right letters.

- **Latent mode:** filter
- **Defaults:** up to 10 turns/episode, 10 episodes
- **Latents:** 105 · **Prompts:** `no_info`, `some_info`, `full_info` · **Feedbacks:** `standard` (default), `information`
- Wraps `textarena.envs.Hangman.env.HangmanEnv`.

## Game Dynamics

- A hidden word is chosen from a word pool filtered by the active latent.
- Each turn the agent guesses a letter `[A]` or the full word `[APPLE]`. Correct letters are
  revealed in position; a wrong guess costs an attempt.
- **Reward:** `1.0` if the word is fully revealed or guessed; otherwise partial credit =
  `letters_revealed / word_length`.

## Latent Mode: Filter

Hangman uses `filter_fn` latents. Each latent defines a property that the target word must satisfy. The trajectory generator filters a word pool using the latent's `filter_candidate(word)` method, then samples target words from the filtered set.

The trajectory JSON contains `{"target_word": "apple"}` per episode.

## Latent Catalog (105 latents)

### Word Length (12 latents)

| ID | Complexity | Description |
|----|-----------|-------------|
| `length_3` | Easy | Words with exactly 3 letters |
| `length_4` | Easy | Words with exactly 4 letters |
| `length_5` | Easy | Words with exactly 5 letters |
| `length_6` | Easy | Words with exactly 6 letters |
| `length_7` | Easy | Words with exactly 7 letters |
| `length_8` | Easy | Words with exactly 8 letters |
| `length_9` | Easy | Words with exactly 9 letters |
| `length_10` | Easy | Words with exactly 10 letters |
| `length_11_plus` | Easy | Words with 11 or more letters |
| `length_short` | Easy | Short words with 3–5 letters |
| `length_medium` | Easy | Medium-length words with 6–8 letters |
| `length_long` | Easy | Long words with 9+ letters |

### Vowel Patterns (11 latents)

| ID | Complexity | Description |
|----|-----------|-------------|
| `vowel_count_0` | Easy | Words with exactly 0 vowels |
| `vowel_count_1` | Easy | Words with exactly 1 vowel |
| `vowel_count_2` | Easy | Words with exactly 2 vowels |
| `vowel_count_3` | Easy | Words with exactly 3 vowels |
| `vowel_count_4` | Easy | Words with exactly 4 vowels |
| `vowel_count_5` | Easy | Words with exactly 5 vowels |
| `vowel_count_6_plus` | Easy | Words with 6 or more vowels |
| `vowel_ratio_low` | Easy | Words where less than 25% of letters are vowels |
| `vowel_ratio_high` | Easy | Words where more than 45% of letters are vowels |
| `vowel_heavy` | Medium | Words with more vowels than consonants |
| `consonant_heavy` | Medium | Words with more consonants than vowels |

### Starting Letters (15 latents)

| ID | Complexity | Description |
|----|-----------|-------------|
| `starts_with_A` | Medium | Words starting with A |
| `starts_with_B` | Medium | Words starting with B |
| `starts_with_C` | Medium | Words starting with C |
| `starts_with_D` | Medium | Words starting with D |
| `starts_with_F` | Medium | Words starting with F |
| `starts_with_G` | Medium | Words starting with G |
| `starts_with_H` | Medium | Words starting with H |
| `starts_with_L` | Medium | Words starting with L |
| `starts_with_M` | Medium | Words starting with M |
| `starts_with_N` | Medium | Words starting with N |
| `starts_with_P` | Medium | Words starting with P |
| `starts_with_R` | Medium | Words starting with R |
| `starts_with_S` | Medium | Words starting with S |
| `starts_with_T` | Medium | Words starting with T |
| `starts_with_W` | Medium | Words starting with W |

### Ending Patterns (20 latents)

| ID | Complexity | Description |
|----|-----------|-------------|
| `ends_with_D` | Medium | Words ending with D |
| `ends_with_E` | Medium | Words ending with E |
| `ends_with_G` | Medium | Words ending with G |
| `ends_with_L` | Medium | Words ending with L |
| `ends_with_N` | Medium | Words ending with N |
| `ends_with_R` | Medium | Words ending with R |
| `ends_with_S` | Medium | Words ending with S |
| `ends_with_T` | Medium | Words ending with T |
| `ends_with_Y` | Medium | Words ending with Y |
| `ending_ABLE` | Medium | Words ending with suffix -ABLE |
| `ending_ED` | Medium | Words ending with suffix -ED |
| `ending_ER` | Medium | Words ending with suffix -ER |
| `ending_EST` | Medium | Words ending with suffix -EST |
| `ending_FUL` | Medium | Words ending with suffix -FUL |
| `ending_ING` | Medium | Words ending with suffix -ING |
| `ending_LESS` | Medium | Words ending with suffix -LESS |
| `ending_LY` | Medium | Words ending with suffix -LY |
| `ending_MENT` | Medium | Words ending with suffix -MENT |
| `ending_NESS` | Medium | Words ending with suffix -NESS |
| `ending_TION` | Medium | Words ending with suffix -TION |

### Letter Frequency (9 latents)

| ID | Complexity | Description |
|----|-----------|-------------|
| `common_letters_only` | Medium | Only common letters (E, T, A, O, I, N, S, H, R, D, L, U) |
| `contains_J` | Medium | Contains the letter J |
| `contains_Q` | Medium | Contains the letter Q |
| `contains_X` | Medium | Contains the letter X |
| `contains_Z` | Medium | Contains the letter Z |
| `has_rare_letter` | Medium | Contains rare letters (Q, X, Z, or J) |
| `high_frequency_score` | Medium | High average letter frequency (score > 6.0) |
| `low_frequency_score` | Medium | Low average letter frequency (score < 4.0) |
| `no_E` | Medium | Does not contain the letter E |

### Structural Patterns (24 latents)

| ID | Complexity | Description |
|----|-----------|-------------|
| `has_double_letter` | Medium | Consecutive identical letters (e.g., LL, EE) |
| `has_repeated_letters` | Medium | At least one letter appears more than once |
| `has_start_cluster` | Medium | Starts with a consonant cluster (BL, CH, ST, etc.) |
| `no_repeated_letters` | Medium | All letters unique |
| `cluster_br` | Hard | Starts with consonant cluster BR |
| `cluster_ch` | Hard | Starts with consonant cluster CH |
| `cluster_cr` | Hard | Starts with consonant cluster CR |
| `cluster_gr` | Hard | Starts with consonant cluster GR |
| `cluster_pr` | Hard | Starts with consonant cluster PR |
| `cluster_sh` | Hard | Starts with consonant cluster SH |
| `cluster_sp` | Hard | Starts with consonant cluster SP |
| `cluster_st` | Hard | Starts with consonant cluster ST |
| `cluster_th` | Hard | Starts with consonant cluster TH |
| `cluster_tr` | Hard | Starts with consonant cluster TR |
| `double_E` | Hard | Contains double letter EE |
| `double_F` | Hard | Contains double letter FF |
| `double_L` | Hard | Contains double letter LL |
| `double_M` | Hard | Contains double letter MM |
| `double_N` | Hard | Contains double letter NN |
| `double_O` | Hard | Contains double letter OO |
| `double_P` | Hard | Contains double letter PP |
| `double_R` | Hard | Contains double letter RR |
| `double_S` | Hard | Contains double letter SS |
| `double_T` | Hard | Contains double letter TT |

### Semantic Categories (10 latents)

| ID | Complexity | Description |
|----|-----------|-------------|
| `category_actions` | Hard | Action verbs |
| `category_animals` | Hard | Animal names |
| `category_body_parts` | Hard | Body parts |
| `category_colors` | Hard | Color words |
| `category_emotions` | Hard | Emotion words |
| `category_food` | Hard | Food items |
| `category_nature` | Hard | Nature terms |
| `category_objects` | Hard | Common objects |
| `category_places` | Hard | Place names |
| `category_weather` | Hard | Weather terms |

### Difficulty Meta-Latents (4 latents)

| ID | Complexity | Description |
|----|-----------|-------------|
| `all_words` | Easy | No filter (any word) |
| `easy_word` | Easy | Easy words: 4–6 letters, common letters, high frequency |
| `medium_word` | Medium | Medium difficulty: 5–8 letters, moderate frequency |
| `hard_word` | Hard | Hard words: 7+ letters, low frequency letters |

**Complexity distribution**: Easy 23, Medium 51, Hard 31

## Prompt Variants

| ID | Info Level |
|----|------------|
| `no_info` | Standard hangman rules, no mention of word patterns |
| `some_info` | Hints that target words may share properties across episodes |
| `full_info` | Explicitly states words follow a hidden property constraint |

## Feedback Variants

| ID | After each episode the agent sees… |
|---|---|
| `standard` (default) | Whether it revealed the word, and the score |
| `information` | The above **plus** the ground-truth word, every episode regardless of success |

## Episode Flow

```
┌───────────────────────────────────────────────────────────────────┐
│ Episode (up to 10 turns)                                          │
│                                                                   │
│  Turn 1 (letter guess):                                           │
│    Agent: "[E]"                                                   │
│    Env: "_ E _ _ _ (5 letters, 5 attempts left)"                 │
│    done = False                                                   │
│                                                                   │
│  Turn N (word guess):                                             │
│    Agent: "[apple]"                                               │
│    Env: "Correct! The word was apple."                            │
│    done = True, reward = 1.0                                      │
│                                                                   │
│  OR after the turn limit (failed):                                │
│    done = True, reward = letters revealed / word length           │
│                                                                   │
│  Ground truth (from trajectory JSON):                             │
│    {"target_word": "apple"}                                       │
└───────────────────────────────────────────────────────────────────┘
```

## Usage

```python
import latentgym.envs.hangman
from latentgym.core import FullyDefinedEnv, make_env

# With trajectory file (standard)
fd = FullyDefinedEnv(env_name="hangman", latent_id="ends_with_ING",
                      prompt_id="full_info", feedback_id="standard", num_episodes=10)
env = make_env(fd, trajectory_path="traj.json")

# With seed + word pool (quick testing)
env = make_env(fd, seed=42, candidate_pool_path="word_lists/words.txt")
```

## Trajectory Generation

```python
from latentgym.envs.hangman.trajectory_generator import generate_hangman_trajectories

generate_hangman_trajectories(
    latent_id="ends_with_ING",
    num_episodes=10,
    n_trajectories=100,
    seed=42,
    candidate_pool_path="path/to/words.txt",
    output_dir="data/eval/hangman_ending_ING/",
)
```

Requires a word pool file (one word per line). The generator filters the pool and reports how many words pass the filter.

## Latent Examples

### Word Length
- `length_3`: "cat", "dog", "run" — very easy, few letters to guess
- `length_7`: "kitchen", "blanket" — harder, more letter combinations
- `length_11_plus`: "acknowledge", "anniversary" — very hard, large search space

### Structural Patterns
- `double_L`: words containing "ll" — "balloon", "million", "valley"
- `cluster_th`: words starting with "th" — "through", "thought", "theater"
- `no_repeated_letters`: "world", "black", "first" — all letters unique

### Semantic Categories
- `category_animals`: "elephant", "giraffe", "penguin" — closed word set
- `category_weather`: "thunder", "blizzard", "drought" — small pool, expect reuse across episodes

### Ending Patterns
- `ending_ING`: "running", "playing", "singing" — very common suffix
- `ending_TION`: "station", "education" — requires longer words (8+ letters)
- `ending_ABLE`: "capable", "reliable" — also requires longer words

### Discovery Strategy
For Hangman, the hidden constraint affects **letter frequency**. If all words end in "-ING", the agent should learn to guess I, N, G early. If all words are `category_animals`, guessing common animal letters (A, E, R, N) is optimal.

## TextArena Dependency

Wraps `textarena.envs.Hangman.env.HangmanEnv`. After reset, the adapter overrides the full game state: `target_word`, `target_letters`, `current_board`, `guessed_letters`, and `tries_left` — ensuring the trajectory-specified word is used regardless of TextArena's internal random selection. The `HangmanEnv` is constructed with `word_list=[target_word]` as an additional guard.
