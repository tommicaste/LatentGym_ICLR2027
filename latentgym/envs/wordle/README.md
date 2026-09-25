# Wordle

Guess a hidden 5-letter word; each guess returns per-letter feedback — Green (right letter, right
position), Yellow (right letter, wrong position), Gray (not in word). Across episodes the target
words satisfy a latent property, so an agent that infers the property can narrow its guesses.

- **Latent mode:** filter
- **Defaults:** word length 5, up to 10 turns/episode, 5 episodes
- **Latents:** 165 · **Prompts:** `no_info`, `some_info`, `full_info` · **Feedbacks:** `standard`, `information`
- Wraps `textarena.envs.Wordle.env.WordleEnv`.

## Game Dynamics

- A hidden 5-letter word is chosen from an English word list by the latent.
- Each turn the agent guesses `[crane]` and receives G/Y/X feedback per letter.
- **Reward:** `1.0` if the word is guessed; otherwise partial credit equal to the fraction of
  letters correctly placed in the final guess.

## Episode Flow

```
┌───────────────────────────────────────────────────────────────────┐
│ Episode (up to 10 turns)                                          │
│                                                                   │
│  Attempt 1:                                                       │
│    Agent: "[crane]"                                               │
│    Env: "[crane]\nG Y X X Y"  (G=right pos, Y=wrong pos, X=not in word) │
│    done = False                                                   │
│                                                                   │
│  Attempt 2:                                                       │
│    Agent: "[stale]"                                               │
│    Env: "[stale]\nG G G G G"  (all correct!)                     │
│    done = True, reward = 1.0                                      │
│                                                                   │
│  OR after the turn limit (failed):                                │
│    done = True, reward = fraction of letters correctly placed     │
│                                                                   │
│  Ground truth (from trajectory JSON):                             │
│    {"target_word": "stale", "word_length": 5}                    │
└───────────────────────────────────────────────────────────────────┘
```

## Latent Mode: Filter

Wordle uses `filter_fn` latents. Each latent defines a constraint on which words can be the target. The trajectory generator filters a word pool using `filter_fn`, then samples target words from the filtered set.

The trajectory JSON contains `{"target_word": "crane", "word_length": 5}` per episode.

## Latent Catalog (165 latents)

### Easy (24 latents)

| ID | Description |
|----|-------------|
| `common_letters_only` | Uses only common letters: E,T,A,O,I,N,S,H,R,D,L,C,U |
| `double_D` | Words containing double 'D' (dd) |
| `double_E` | Words containing double 'E' (ee) |
| `double_F` | Words containing double 'F' (ff) |
| `double_L` | Words containing double 'L' (ll) |
| `double_N` | Words containing double 'N' (nn) |
| `double_O` | Words containing double 'O' (oo) |
| `double_P` | Words containing double 'P' (pp) |
| `double_R` | Words containing double 'R' (rr) |
| `double_S` | Words containing double 'S' (ss) |
| `double_T` | Words containing double 'T' (tt) |
| `has_double_letter` | Words containing any consecutive double letter |
| `has_rare_letter` | Contains at least one rare letter: Q,Z,X,J,K,V,B,W,F,G,Y,P |
| `multiple_repeated_letters` | At least two different letters are repeated |
| `no_repeated_letters` | All letters in the word are unique |
| `one_repeated_letter` | Exactly one letter appears twice |
| `unique_letters_3` | Words with exactly 3 distinct letters |
| `unique_letters_4` | Words with exactly 4 distinct letters |
| `unique_letters_5` | Words with exactly 5 distinct letters |
| `vowel_count_0` | All words have exactly 0 vowels |
| `vowel_count_1` | All words have exactly 1 vowel |
| `vowel_count_2` | All words have exactly 2 vowels |
| `vowel_count_3` | All words have exactly 3 vowels |
| `vowel_count_4` | All words have exactly 4 vowels |

### Medium (84 latents)

**Vowel/Alphabet Patterns (7)**

| ID | Description |
|----|-------------|
| `first_half_alphabet` | Only uses letters from A to M |
| `mixed_balanced` | At least 2 letters from each half of alphabet |
| `second_half_alphabet` | Only uses letters from N to Z |
| `vowel_even_positions` | Vowels appear at positions 0, 2, and 4 |
| `vowel_odd_positions` | Vowels appear at positions 1 and 3 (0-indexed) |
| `vowel_start` | Starts with a vowel |
| `vowels_clustered` | At least 2 adjacent vowels |

**Starting Patterns (10)**

| ID | Description |
|----|-------------|
| `consonant_cluster_start` | Starts with consonant cluster (BR, ST, TR, etc.) |
| `single_consonant_start` | Starts with single consonant followed by vowel |
| `starts_with_B` | First letter is always 'B' |
| `starts_with_C` | First letter is always 'C' |
| `starts_with_D` | First letter is always 'D' |
| `starts_with_M` | First letter is always 'M' |
| `starts_with_P` | First letter is always 'P' |
| `starts_with_R` | First letter is always 'R' |
| `starts_with_S` | First letter is always 'S' |

**Specific Letter at Position (37)**: `position_{L}_at_{P}` for letters E, T, A, O, I, N, S, R × positions 0–4

| ID | Description |
|----|-------------|
| `position_E_at_0` .. `position_E_at_4` | Letter 'E' at position 0, 1, 2, 3, or 4 |
| `position_T_at_0` .. `position_T_at_4` | Letter 'T' at position 0, 1, 2, 3, or 4 |
| `position_A_at_0` .. `position_A_at_4` | Letter 'A' at position 0, 1, 2, 3, or 4 |
| `position_O_at_0` .. `position_O_at_4` | Letter 'O' at position 0, 1, 2, 3, or 4 |
| `position_I_at_0` .. `position_I_at_4` | Letter 'I' at position 0, 1, 2, 3, or 4 |
| `position_N_at_0` .. `position_N_at_4` | Letter 'N' at position 0, 1, 2, 3, or 4 |
| `position_S_at_0` .. `position_S_at_4` | Letter 'S' at position 0, 1, 2, 3, or 4 |
| `position_R_at_0` .. `position_R_at_4` | Letter 'R' at position 0, 1, 2, 3, or 4 |

(37 total: 8 letters × 5 positions minus 3 already counted under starting patterns)

**Consonant Clusters at Start (15)**

| ID | Description |
|----|-------------|
| `cluster_start_BL-cluster` | Words starting with BL |
| `cluster_start_BR-cluster` | Words starting with BR |
| `cluster_start_CH-cluster` | Words starting with CH |
| `cluster_start_CL-cluster` | Words starting with CL |
| `cluster_start_FL-cluster` | Words starting with FL |
| `cluster_start_GR-cluster` | Words starting with GR |
| `cluster_start_PL-cluster` | Words starting with PL |
| `cluster_start_PR-cluster` | Words starting with PR |
| `cluster_start_SC-cluster` | Words starting with SC |
| `cluster_start_SH-cluster` | Words starting with SH |
| `cluster_start_SL-cluster` | Words starting with SL |
| `cluster_start_SP-cluster` | Words starting with SP |
| `cluster_start_ST-cluster` | Words starting with ST |
| `cluster_start_TH-cluster` | Words starting with TH |
| `cluster_start_TR-cluster` | Words starting with TR |

**Letter Pairs/Sequences (10)**

| ID | Description |
|----|-------------|
| `contains_AI` | Words containing 'AI' |
| `contains_AR` | Words containing 'AR' |
| `contains_EA` | Words containing 'EA' |
| `contains_EE` | Words containing 'EE' |
| `contains_ER` | Words containing 'ER' |
| `contains_ING` | Words containing 'ING' |
| `contains_OO` | Words containing 'OO' |
| `contains_OR` | Words containing 'OR' |
| `contains_OU` | Words containing 'OU' |
| `contains_TION` | Words containing 'TION' |

**V/C Patterns (3)**

| ID | Description |
|----|-------------|
| `pattern_CVCCV` | Pattern: C-V-C-C-V |
| `pattern_CVCVC` | Consonant-Vowel alternating: C-V-C-V-C |
| `pattern_VCCVC` | Pattern: V-C-C-V-C |

### Hard (57 latents)

**Ending Patterns (13)**

| ID | Description |
|----|-------------|
| `ending_AL` | Words ending in 'AL' |
| `ending_CH` | Words ending in 'CH' |
| `ending_ED` | Words ending in 'ED' |
| `ending_ER` | Words ending in 'ER' |
| `ending_LE` | Words ending in 'LE' |
| `ending_LY` | Words ending in 'LY' |
| `ending_ND` | Words ending in 'ND' |
| `ending_NG` | Words ending in 'NG' |
| `ending_NT` | Words ending in 'NT' |
| `ending_SE` | Words ending in 'SE' |
| `ending_SH` | Words ending in 'SH' |
| `ending_ST` | Words ending in 'ST' |
| `ending_TH` | Words ending in 'TH' |

**Letter Frequency (3)**

| ID | Description |
|----|-------------|
| `letter_freq_high` | Average letter frequency > 7% |
| `letter_freq_low` | Average letter frequency < 4% |
| `letter_freq_medium` | Average letter frequency 4–7% |

**Near Reference Word (7)**

| ID | Description |
|----|-------------|
| `near_CRANE` | Levenshtein distance ≤ 2 from 'CRANE' |
| `near_PROUD` | Levenshtein distance ≤ 2 from 'PROUD' |
| `near_RAISE` | Levenshtein distance ≤ 2 from 'RAISE' |
| `near_ROAST` | Levenshtein distance ≤ 2 from 'ROAST' |
| `near_SLATE` | Levenshtein distance ≤ 2 from 'SLATE' |
| `near_STALE` | Levenshtein distance ≤ 2 from 'STALE' |
| `near_TEARS` | Levenshtein distance ≤ 2 from 'TEARS' |

**Multi-Position Constraints (16)**

| ID | Description |
|----|-------------|
| `multi_position_A_at_1_T_end` | 'A' at position 1, ends with 'T' |
| `multi_position_A_at_2` | Middle letter (position 2) is always 'A' |
| `multi_position_A_at_2_E_end` | 'A' at position 2, ends with 'E' |
| `multi_position_B_start_E_end` | Starts with 'B', ends with 'E' |
| `multi_position_C_start_E_end` | Starts with 'C', ends with 'E' |
| `multi_position_E_at_1_R_end` | 'E' at position 1, ends with 'R' |
| `multi_position_I_at_2_T_end` | 'I' at position 2, ends with 'T' |
| `multi_position_O_at_1_T_end` | 'O' at position 1, ends with 'T' |
| `multi_position_O_at_2_T_end` | 'O' at position 2, ends with 'T' |
| `multi_position_P_start_E_end` | Starts with 'P', ends with 'E' |
| `multi_position_R_at_1_N_at_3` | Position 1='R' AND position 3='N' |
| `multi_position_S_start_E_end` | Position 0='S' AND position 4='E' |
| `multi_position_S_start_S_end` | Starts and ends with 'S' |
| `multi_position_S_start_T_end` | Starts with 'S', ends with 'T' |
| `multi_position_T_start_E_end` | Starts with 'T', ends with 'E' |
| `multi_position_T_start_T_end` | Starts and ends with 'T' |

**Semantic Categories (12)**

| ID | Description |
|----|-------------|
| `category_animals` | Words in category: animals |
| `category_body_parts` | Words in category: body parts |
| `category_buildings` | Words in category: buildings |
| `category_clothing` | Words in category: clothing |
| `category_colors` | Words in category: colors |
| `category_emotions` | Words in category: emotions |
| `category_food` | Words in category: food |
| `category_fruits` | Words in category: fruits |
| `category_metals` | Words in category: metals |
| `category_nature` | Words in category: nature |
| `category_tools` | Words in category: tools |
| `category_weather` | Words in category: weather |

**Word Attributes (6)**

| ID | Description |
|----|-------------|
| `attribute_abstract_concepts` | Words with attribute: abstract concepts |
| `attribute_actions` | Words with attribute: actions |
| `attribute_materials` | Words with attribute: materials |
| `attribute_quantities` | Words with attribute: quantities |
| `attribute_size_descriptors` | Words with attribute: size descriptors |
| `attribute_time_related` | Words with attribute: time related |

**Complexity distribution**: Easy 24, Medium 84, Hard 57

## Prompt Variants

| ID | Info Level |
|----|------------|
| no_info | Standard wordle rules, no mention of word patterns |
| some_info | Hints that target words may share properties across episodes |
| full_info | Explicitly states words follow a hidden constraint pattern |

## Feedback Variants

| ID | After each episode the agent sees… |
|---|---|
| `standard` (default) | Whether it guessed the word, and the score |
| `information` | The above **plus** the ground-truth target word, every episode regardless of success |

## Usage

```python
import latentgym.envs.wordle
from latentgym.core import FullyDefinedEnv, make_env

# With trajectory file (standard)
fd = FullyDefinedEnv(env_name="wordle", latent_id="vowel_count_2",
                      prompt_id="full_info", feedback_id="standard", num_episodes=5)
env = make_env(fd, trajectory_path="traj.json")

# With seed + word pool (quick testing)
env = make_env(fd, seed=42, candidate_pool_path="word_lists/5letter.txt")
```

## Trajectory Generation

```python
from latentgym.envs.wordle.trajectory_generator import generate_wordle_trajectories

generate_wordle_trajectories(
    latent_id="vowel_count_2",
    num_episodes=5,
    n_trajectories=100,
    seed=42,
    candidate_pool_path="path/to/5letter_words.txt",
    output_dir="data/eval/wordle_vowel_count_2/",
    sampling="without_replacement",  # default; use "with_replacement" if pool is small
)
```

Requires a word pool file (one word per line). The generator filters the pool and reports how many words pass the filter.

## Episode Transition

Between episodes, wordle adds a note:
> "Note: the secret word may or may not have changed. Do not assume your previous guesses still apply."

This prevents the agent from assuming the same word across episodes.

## Latent Examples

### Vowel Count
- `vowel_count_0`: "crypt", "glyph", "lymph" (rare — few 5-letter words have 0 vowels)
- `vowel_count_2`: "apple", "bread", "green" (most common bucket)
- `vowel_count_4`: "audio", "adieu" (rare — very constrained pool)

### Positional Constraints
- `position_E_at_1`: "bread", "steam", "pearl" — second letter is always E
- `position_S_at_0`: "start", "store", "sleep" — first letter is always S

### Edit Distance Clusters
- `near_CRANE`: words within Levenshtein distance 2 of "crane" — "crave", "crane", "craze", "brane"
- Pool sizes vary greatly: `near_CRANE` may have 20+ words, `near_PROUD` may have <10

### Multi-Position
- `multi_position_S_start_E_end`: Position 0='S' AND Position 4='E' — "shake", "shine", "stale"
- `multi_position_T_start_T_end`: Starts and ends with 'T' — "toast", "trait", "trust"

### Semantic Categories
- `category_fruits`: closed set — "apple", "berry", "grape", "lemon", "melon", "peach", "mango"
- `category_metals`: "brass", "steel", "bronze", "iron", "copper", "silver", "gold"
- Pool sizes are small (5–10 words). With 5+ episodes, expect word reuse.

## Word Pool Notes

- Some latents have very small pools: semantic categories (5–10 words), multi-position
  constraints (10–30 words), `vowel_count_0` (<5 words). The trajectory generator reports pool
  size per latent — check it before generating large datasets.
- Default sampling is `without_replacement` (each episode gets a different target), falling back
  to `with_replacement` if the filtered pool is exhausted.

## TextArena Dependency

Wraps `textarena.envs.Wordle.env.WordleEnv`. The adapter handles wordle-specific output parsing and reward extraction.
