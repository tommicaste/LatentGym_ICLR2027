"""Filter-based latent definitions for word ladder.

Path-aware latents that reason about the solution path, not just start/target properties.
These create meaningful continual learning signals — patterns that, once discovered,
give the agent a strategic advantage in future episodes.

Categories:
1. Hub word — optimal path goes through a specific intermediate word
2. Restricted vocabulary — path stays within a small word set
3. Positional change order — positions change in a consistent order
4. Letter substitution patterns — specific types of letter swaps recur
5. Word family chains — all intermediates share a structural property

Commented-out categories (start/target property latents — weak learning signal
because the agent already sees both endpoints):
- Skeleton constraints
- Fixed letter at position
- Shared suffix/prefix
- Vowel patterns
"""

from latentgym.core.latent import LatentDefinition, LatentComplexity
from latentgym.core.registry import register_latent
from . import word_pair_filters as wpf


# =============================================================================
# COMMENTED OUT: Start/target property latents
# These constrain the relationship between start and target, but since the
# agent sees both words, knowing the pattern doesn't help solve faster.
# Kept here for reference / future use.
# =============================================================================

# # 1. Skeleton constraints
# register_latent("wordladder", LatentDefinition(
#     id="same_vowel_skeleton",
#     name="Same Vowel Skeleton",
#     complexity=LatentComplexity.MEDIUM,
#     description="Start and target have identical vowels at same positions (only consonants differ)",
#     filter_fn=wpf.same_vowel_skeleton,
# ))
#
# register_latent("wordladder", LatentDefinition(
#     id="same_consonant_skeleton",
#     name="Same Consonant Skeleton",
#     complexity=LatentComplexity.HARD,
#     description="Start and target have identical consonants at same positions (only vowels differ)",
#     filter_fn=wpf.same_consonant_skeleton,
# ))
#
# # 2. Fixed letter at a specific position
# for pos in range(4):
#     register_latent("wordladder", LatentDefinition(
#         id=f"fixed_pos_{pos}",
#         name=f"Fixed Letter at Position {pos}",
#         complexity=LatentComplexity.MEDIUM,
#         description=f"Start and target share the same letter at position {pos}",
#         filter_fn=wpf.fixed_letter_at(pos),
#     ))
#
# # 3. Word family: shared suffix or prefix
# register_latent("wordladder", LatentDefinition(
#     id="shared_suffix_2", name="Shared 2-Letter Suffix",
#     complexity=LatentComplexity.MEDIUM,
#     description="Start and target share the same 2-letter suffix",
#     filter_fn=wpf.shared_suffix(2),
# ))
# register_latent("wordladder", LatentDefinition(
#     id="shared_suffix_3", name="Shared 3-Letter Suffix",
#     complexity=LatentComplexity.HARD,
#     description="Start and target share the same 3-letter suffix",
#     filter_fn=wpf.shared_suffix(3),
# ))
# register_latent("wordladder", LatentDefinition(
#     id="shared_prefix_2", name="Shared 2-Letter Prefix",
#     complexity=LatentComplexity.MEDIUM,
#     description="Start and target share the same 2-letter prefix",
#     filter_fn=wpf.shared_prefix(2),
# ))
# register_latent("wordladder", LatentDefinition(
#     id="shared_prefix_3", name="Shared 3-Letter Prefix",
#     complexity=LatentComplexity.HARD,
#     description="Start and target share the same 3-letter prefix",
#     filter_fn=wpf.shared_prefix(3),
# ))
#
# # 4. Specific vowel patterns
# register_latent("wordladder", LatentDefinition(
#     id="pattern_a_e", name="Pattern _a_e", complexity=LatentComplexity.MEDIUM,
#     description="Both words match _a_e pattern", filter_fn=wpf.vowel_pattern("_a_e"),
# ))
# register_latent("wordladder", LatentDefinition(
#     id="pattern_i_e", name="Pattern _i_e", complexity=LatentComplexity.MEDIUM,
#     description="Both words match _i_e pattern", filter_fn=wpf.vowel_pattern("_i_e"),
# ))
# register_latent("wordladder", LatentDefinition(
#     id="pattern_o_e", name="Pattern _o_e", complexity=LatentComplexity.MEDIUM,
#     description="Both words match _o_e pattern", filter_fn=wpf.vowel_pattern("_o_e"),
# ))
# register_latent("wordladder", LatentDefinition(
#     id="pattern_a_", name="Pattern _a_", complexity=LatentComplexity.EASY,
#     description="Both words match _a_ pattern", filter_fn=wpf.vowel_pattern("_a_"),
# ))
# register_latent("wordladder", LatentDefinition(
#     id="pattern_o_", name="Pattern _o_", complexity=LatentComplexity.EASY,
#     description="Both words match _o_ pattern", filter_fn=wpf.vowel_pattern("_o_"),
# ))
# register_latent("wordladder", LatentDefinition(
#     id="pattern_i_", name="Pattern _i_", complexity=LatentComplexity.EASY,
#     description="Both words match _i_ pattern", filter_fn=wpf.vowel_pattern("_i_"),
# ))


# =============================================================================
# 5. PATH-AWARE LATENTS
#    These reason about the solution path, not just start/target properties.
#    The graph is built lazily on first filter call.
# =============================================================================

# -- 5a. Hub Word Latent (generator-based)
#    On first episode: discovers top hub words from the graph, picks one randomly.
#    All episodes in the trajectory use pairs whose shortest path goes through
#    that hub. The hub word is recorded in episode_config as ground truth.
#    Different trajectories get different hubs.

def _canonical_pair(a, b):
    """Canonical form so (a,b) and (b,a) are treated as the same pair."""
    return (min(a, b), max(a, b))


def _find_pairs_through_hub(graph, hub):
    """Find all (start, target) pairs whose shortest path goes through hub."""
    pairs = []
    for start in graph.words:
        if start == hub:
            continue
        for target in graph.words:
            if target == hub or target <= start:
                continue
            if graph.path_goes_through(start, target, hub):
                d = graph.distance(start, target)
                if 3 <= d <= 6:
                    pairs.append((start, target))
    return pairs


def _hub_word_generator(word_length: int = 4):
    """Generator: picks a different hub word per trajectory, samples pairs through it.

    Closure state tracks hub rotation and cross-trajectory pair usage:
    - Each trajectory gets a different hub (cycles through top hubs)
    - When hubs are exhausted, reuses hubs with most fresh pairs
    - A pair (or its reverse) can appear at most MAX_PAIR_REUSE times across all trajectories
    - Within a trajectory, no pair or its reverse is repeated
    """
    MAX_PAIR_REUSE = 16
    # Closure state persists across trajectories
    _state = {}

    def _init(graph):
        top_hubs = graph.hub_words(top_k=50)
        if not top_hubs:
            raise ValueError(f"No hub words found for {word_length}-letter words")
        _state["graph"] = graph
        _state["all_hubs"] = [h for h, _ in top_hubs]
        _state["hub_idx"] = 0
        _state["hub_pairs"] = {}       # hub -> list of canonical pairs
        _state["pair_usage"] = {}      # canonical_pair -> count across trajectories

    def _get_hub_pairs(hub):
        """Get (and cache) all valid pairs through a hub."""
        if hub not in _state["hub_pairs"]:
            pairs = _find_pairs_through_hub(_state["graph"], hub)
            _state["hub_pairs"][hub] = pairs
        return _state["hub_pairs"][hub]

    def _fresh_pairs(hub):
        """Pairs through hub not yet used MAX_PAIR_REUSE times globally."""
        return [p for p in _get_hub_pairs(hub)
                if _state["pair_usage"].get(_canonical_pair(*p), 0) < MAX_PAIR_REUSE]

    def generator_fn(env_params, episode_idx, num_episodes, context):
        import random as _random

        if not _state:
            from .word_ladder_graph import get_graph
            _init(get_graph(word_length))

        if "hub" not in context:
            # 1. Try next unused hub
            hub = None
            fresh = []
            while _state["hub_idx"] < len(_state["all_hubs"]):
                candidate = _state["all_hubs"][_state["hub_idx"]]
                _state["hub_idx"] += 1
                fresh = _fresh_pairs(candidate)
                if len(fresh) >= num_episodes:
                    hub = candidate
                    break

            # 2. Ran out of fresh hubs — reuse the one with most fresh pairs
            if hub is None:
                best_hub = None
                best_fresh = []
                for h in _state["all_hubs"]:
                    f = _fresh_pairs(h)
                    if len(f) > len(best_fresh):
                        best_hub = h
                        best_fresh = f
                if best_hub and len(best_fresh) >= num_episodes:
                    hub = best_hub
                    fresh = best_fresh
                else:
                    raise ValueError(
                        f"No hub with >= {num_episodes} fresh pairs for "
                        f"{word_length}-letter words"
                    )

            context["hub"] = hub
            context["valid_pairs"] = fresh
            context["used_in_traj"] = set()  # canonical pairs used in THIS trajectory

        # 3. Within trajectory: no pair or its reverse
        available = [p for p in context["valid_pairs"]
                     if _canonical_pair(*p) not in context["used_in_traj"]]
        if not available:
            available = context["valid_pairs"]  # fallback

        pair = _random.choice(available)
        cp = _canonical_pair(*pair)
        context["used_in_traj"].add(cp)
        _state["pair_usage"][cp] = _state["pair_usage"].get(cp, 0) + 1

        return {
            "start_word": pair[0],
            "target_word": pair[1],
            "hub_word": context["hub"],
            "word_length": word_length,
            "max_turns_per_episode": env_params.get("max_turns_per_episode", 20),
        }
    return generator_fn

for wl in [3, 4, 5]:
    register_latent("wordladder", LatentDefinition(
        id=f"hub_word_{wl}letter",
        name=f"Hub Word ({wl}-letter)",
        complexity=LatentComplexity.HARD,
        description=(
            f"All {wl}-letter word pairs share a common hub word on their optimal path. "
            f"The specific hub varies per trajectory. Agent discovers the hub and uses it "
            f"to split puzzles into two easier sub-problems."
        ),
        generator_fn=_hub_word_generator(wl),
    ))


# -- 5b. Restricted Vocabulary Latent (generator-based)
#    On first episode: finds a densely connected subgraph of ~40 words.
#    All episodes use pairs solvable within that vocabulary.
#    The vocabulary is recorded in episode_config as ground truth.

def _find_vocab_pairs(graph, vocab):
    """Find all pairs solvable optimally within a restricted vocabulary."""
    from collections import deque
    pairs = []
    vocab_set = set(vocab)
    vocab_list = sorted(vocab)
    for i, start in enumerate(vocab_list):
        for target in vocab_list[i + 1:]:
            full_dist = graph.distance(start, target)
            if full_dist < 2 or full_dist > 6:
                continue
            # BFS within restricted vocab
            dist = {start: 0}
            q = deque([start])
            found = False
            while q:
                cur = q.popleft()
                if cur == target:
                    found = (dist[target] == full_dist)
                    break
                for nb in graph.neighbors.get(cur, []):
                    if nb in vocab_set and nb not in dist:
                        dist[nb] = dist[cur] + 1
                        q.append(nb)
            if found:
                pairs.append((start, target))
    return pairs


def _restricted_vocab_generator(word_length: int = 4, vocab_size: int = 40):
    """Generator: different vocabulary per trajectory, cap pair reuse at 12 globally.

    Precomputes many distinct vocabularies on first call. Each trajectory gets
    a different vocab. Cross-trajectory pair reuse capped at 12.
    Within a trajectory, no pair or its reverse is repeated.
    """
    _state = {}

    def _init(graph):
        # Precompute many vocabs with different seeds
        vocabs = []
        seen_vocab_keys = set()
        for seed_offset in range(200):
            v = graph.find_restricted_vocabulary(size=vocab_size, seed=seed_offset)
            key = tuple(sorted(v))
            if key not in seen_vocab_keys:
                seen_vocab_keys.add(key)
                pairs = _find_vocab_pairs(graph, v)
                if len(pairs) >= 5:  # Only keep vocabs with enough pairs
                    vocabs.append((sorted(v), pairs))
        _state["graph"] = graph
        _state["vocabs"] = vocabs          # list of (vocab_list, pairs)
        _state["vocab_idx"] = 0
        _state["pair_usage"] = {}          # canonical_pair -> count

    def _fresh_pairs_for(pairs):
        return [p for p in pairs
                if _state["pair_usage"].get(_canonical_pair(*p), 0) < 12]

    def generator_fn(env_params, episode_idx, num_episodes, context):
        import random as _random

        if not _state:
            from .word_ladder_graph import get_graph
            _init(get_graph(word_length))

        if not _state["vocabs"]:
            raise ValueError(f"No valid vocabularies for {word_length}-letter words")

        if "vocab" not in context:
            # 1. Try next unused vocab
            vocab_list = None
            fresh = []
            while _state["vocab_idx"] < len(_state["vocabs"]):
                v, pairs = _state["vocabs"][_state["vocab_idx"]]
                _state["vocab_idx"] += 1
                fresh = _fresh_pairs_for(pairs)
                if len(fresh) >= num_episodes:
                    vocab_list = v
                    break

            # 2. Ran out of fresh vocabs — reuse the one with most fresh pairs
            if vocab_list is None:
                best_v = None
                best_fresh = []
                for v, pairs in _state["vocabs"]:
                    f = _fresh_pairs_for(pairs)
                    if len(f) > len(best_fresh):
                        best_v = v
                        best_fresh = f
                if best_v and len(best_fresh) >= num_episodes:
                    vocab_list = best_v
                    fresh = best_fresh
                else:
                    raise ValueError(
                        f"No vocabulary with >= {num_episodes} fresh pairs for "
                        f"{word_length}-letter words"
                    )

            context["vocab"] = vocab_list
            context["valid_pairs"] = fresh
            context["used_in_traj"] = set()

        # 3. Within trajectory: no pair or its reverse
        available = [p for p in context["valid_pairs"]
                     if _canonical_pair(*p) not in context["used_in_traj"]]
        if not available:
            available = context["valid_pairs"]  # fallback

        pair = _random.choice(available)
        cp = _canonical_pair(*pair)
        context["used_in_traj"].add(cp)
        _state["pair_usage"][cp] = _state["pair_usage"].get(cp, 0) + 1

        return {
            "start_word": pair[0],
            "target_word": pair[1],
            "vocabulary": context["vocab"],
            "word_length": word_length,
            "max_turns_per_episode": env_params.get("max_turns_per_episode", 20),
        }
    return generator_fn

for wl in [3, 4, 5]:
    register_latent("wordladder", LatentDefinition(
        id=f"restricted_vocab_{wl}letter",
        name=f"Restricted Vocabulary ({wl}-letter)",
        complexity=LatentComplexity.MEDIUM,
        description=(
            f"All {wl}-letter word pairs can be solved using only ~40 specific words. "
            f"The vocabulary varies per trajectory. Agent learns which words are 'in play'."
        ),
        generator_fn=_restricted_vocab_generator(wl, vocab_size=40),
    ))


# -- 5c. Positional Change Order Latents
#    At least one shortest path changes letter positions in a consistent order.
#    Agent learns: "change left-to-right" or "change right-to-left."
#    Signal is noisy (multiple valid paths) but discoverable over many episodes.

register_latent("wordladder", LatentDefinition(
    id="order_left_to_right",
    name="Position Order: Left to Right",
    complexity=LatentComplexity.HARD,
    description="At least one optimal path changes positions left-to-right (0,1,2,...). Agent learns the ordering strategy.",
    filter_fn=wpf.positional_order("left_to_right"),
))

register_latent("wordladder", LatentDefinition(
    id="order_right_to_left",
    name="Position Order: Right to Left",
    complexity=LatentComplexity.HARD,
    description="At least one optimal path changes positions right-to-left (...,2,1,0). Agent learns the ordering strategy.",
    filter_fn=wpf.positional_order("right_to_left"),
))

register_latent("wordladder", LatentDefinition(
    id="order_outside_in",
    name="Position Order: Outside In",
    complexity=LatentComplexity.HARD,
    description="At least one optimal path changes outer positions first, then inner. Agent learns the ordering strategy.",
    filter_fn=wpf.positional_order("outside_in"),
))


# -- 5d. Letter Substitution Pattern Latents
#    Specific types of letter swaps recur across episodes.
#    Agent learns: "vowel swaps are common" or "consonants swap within groups."

register_latent("wordladder", LatentDefinition(
    id="subs_vowel_swaps",
    name="Substitution: Vowel Swaps Only",
    complexity=LatentComplexity.HARD,
    description="At least one optimal path only changes vowels (a↔e↔i↔o↔u). Agent learns vowel rotation strategy.",
    filter_fn=wpf.substitution_pattern("vowel_swaps"),
))

register_latent("wordladder", LatentDefinition(
    id="subs_consonant_swaps",
    name="Substitution: Consonant Swaps Only",
    complexity=LatentComplexity.HARD,
    description="At least one optimal path only changes consonants. Agent learns consonant swap strategy.",
    filter_fn=wpf.substitution_pattern("consonant_swaps"),
))

register_latent("wordladder", LatentDefinition(
    id="subs_alternating",
    name="Substitution: Alternating Vowel/Consonant",
    complexity=LatentComplexity.HARD,
    description="At least one optimal path alternates between vowel and consonant changes.",
    filter_fn=wpf.substitution_pattern("vowel_to_consonant"),
))

register_latent("wordladder", LatentDefinition(
    id="subs_phonetic_group",
    name="Substitution: Same Phonetic Group",
    complexity=LatentComplexity.HARD,
    description="Consonant swaps stay within phonetic groups (b/p/d/t, f/v/s/z, m/n/l/r).",
    filter_fn=wpf.substitution_pattern("same_group"),
))


# -- 5e. Word Family Chain Latents
#    All intermediate words share a structural property.
#    Agent learns: "stay within words containing 'or'" or "all intermediates end in 'e'."

register_latent("wordladder", LatentDefinition(
    id="family_contains_or",
    name="Family Chain: Contains 'or'",
    complexity=LatentComplexity.MEDIUM,
    description="All intermediate words contain 'or' (e.g., cord→core→bore→born). Agent learns to stay in the 'or' family.",
    filter_fn=wpf.family_chain("contains_bigram", "or"),
))

register_latent("wordladder", LatentDefinition(
    id="family_contains_an",
    name="Family Chain: Contains 'an'",
    complexity=LatentComplexity.MEDIUM,
    description="All intermediate words contain 'an'. Agent learns to stay in the 'an' family.",
    filter_fn=wpf.family_chain("contains_bigram", "an"),
))

register_latent("wordladder", LatentDefinition(
    id="family_contains_at",
    name="Family Chain: Contains 'at'",
    complexity=LatentComplexity.MEDIUM,
    description="All intermediate words contain 'at'. Agent learns to stay in the 'at' family.",
    filter_fn=wpf.family_chain("contains_bigram", "at"),
))

register_latent("wordladder", LatentDefinition(
    id="family_ends_e",
    name="Family Chain: Ends With 'e'",
    complexity=LatentComplexity.MEDIUM,
    description="All intermediate words end with 'e'. Agent learns to stay in the '-e' word space.",
    filter_fn=wpf.family_chain("ends_with", "e"),
))

register_latent("wordladder", LatentDefinition(
    id="family_ends_d",
    name="Family Chain: Ends With 'd'",
    complexity=LatentComplexity.MEDIUM,
    description="All intermediate words end with 'd'. Agent learns to stay in the '-d' word space.",
    filter_fn=wpf.family_chain("ends_with", "d"),
))

register_latent("wordladder", LatentDefinition(
    id="family_pattern_cvcc",
    name="Family Chain: CVCC Pattern",
    complexity=LatentComplexity.HARD,
    description="All intermediate words follow consonant-vowel-consonant-consonant pattern.",
    filter_fn=wpf.family_chain("vowel_pattern", "CVCC"),
))

register_latent("wordladder", LatentDefinition(
    id="family_pattern_cvcv",
    name="Family Chain: CVCV Pattern",
    complexity=LatentComplexity.HARD,
    description="All intermediate words follow consonant-vowel-consonant-vowel pattern.",
    filter_fn=wpf.family_chain("vowel_pattern", "CVCV"),
))
