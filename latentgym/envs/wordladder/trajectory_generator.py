"""Trajectory generator for word ladder (filter-based latents on word pairs).

Generates all valid word pairs by building neighbor maps and running BFS
per word length. This is done once and cached, making filtering fast.
"""

import random
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from latentgym.core.registry import get_latent
from latentgym.core.trajectory_utils import Trajectory, Manifest, save_dataset


# ============================================================================
# Word pair generation via BFS (replaces slow per-pair TextArena reset)
# ============================================================================

_PAIR_CACHE: Dict[Tuple[int, int, int], List[Tuple[str, str]]] = {}


def _compute_optimal_path(start: str, target: str) -> List[str]:
    """Compute one shortest path between start and target using the word ladder graph.

    Returns the path as a list of words [start, ..., target], or [start, target]
    if they differ by 1 letter, or [start] if start == target.
    """
    from .word_ladder_graph import get_graph
    graph = get_graph(len(start))
    paths = graph.all_shortest_paths(start, target, max_paths=1)
    if paths:
        return paths[0]
    return [start, target]  # Fallback if no path found


def _enrich_episode_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Add optimal_path and optimal_steps to an episode config.

    If episode_config has a hub_word, computes the path THROUGH the hub
    (start → hub → target) to ensure the ground truth path uses the hub.
    """
    start = config.get("start_word", "")
    target = config.get("target_word", "")
    hub = config.get("hub_word", "")
    if start and target:
        if hub and hub != start and hub != target:
            # Compute path through hub: start→hub + hub→target
            path_to_hub = _compute_optimal_path(start, hub)
            path_from_hub = _compute_optimal_path(hub, target)
            # Merge: remove duplicate hub at junction
            path = path_to_hub + path_from_hub[1:]
        else:
            path = _compute_optimal_path(start, target)
        config["optimal_path"] = path
        config["optimal_steps"] = len(path) - 1
    return config


def _load_word_list() -> List[str]:
    """Load NLTK en-basic word list."""
    import nltk
    from nltk.corpus import words
    try:
        words.words()
    except LookupError:
        nltk.download("words", quiet=True)
    return words.words("en-basic")


def _build_neighbor_map(bucket: List[str]) -> Dict[str, List[str]]:
    """Build neighbor map for words of the same length."""
    word_set = set(bucket)
    nbrs: Dict[str, List[str]] = {w: [] for w in bucket}
    for word in bucket:
        for i in range(len(word)):
            for c in "abcdefghijklmnopqrstuvwxyz":
                if c != word[i]:
                    cand = word[:i] + c + word[i + 1:]
                    if cand in word_set:
                        nbrs[word].append(cand)
    return nbrs


def _find_all_pairs(
    nbrs: Dict[str, List[str]],
    min_dist: int,
    max_dist: int,
) -> List[Tuple[str, str]]:
    """BFS from every word to find all (start, target) pairs at valid distances."""
    pairs: Set[Tuple[str, str]] = set()
    for start in nbrs:
        visited = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            d = visited[cur]
            if d >= max_dist:
                continue
            for nb in nbrs.get(cur, []):
                if nb not in visited:
                    visited[nb] = d + 1
                    if min_dist <= d + 1 <= max_dist:
                        pairs.add((start, nb))
                    q.append(nb)
    return list(pairs)


def _generate_word_pairs(
    word_lengths: List[int] = [3, 4, 5],
    min_distance: int = 2,
    max_distance: int = 6,
) -> List[Tuple[str, str]]:
    """Generate all valid word pairs for given lengths and distance range.

    Builds neighbor maps and runs BFS once per word length. Results are cached.
    """
    all_pairs: List[Tuple[str, str]] = []
    all_words = _load_word_list()

    for wl in word_lengths:
        cache_key = (wl, min_distance, max_distance)
        if cache_key in _PAIR_CACHE:
            all_pairs.extend(_PAIR_CACHE[cache_key])
            continue

        bucket = list(set(w.lower() for w in all_words if len(w) == wl and w.isalpha()))
        if len(bucket) < 2:
            _PAIR_CACHE[cache_key] = []
            continue

        print(f"  Building word graph for length {wl} ({len(bucket)} words)...")
        nbrs = _build_neighbor_map(bucket)

        print(f"  Finding pairs (distance {min_distance}-{max_distance})...")
        pairs = _find_all_pairs(nbrs, min_distance, max_distance)
        print(f"  Found {len(pairs)} pairs for length {wl}")

        _PAIR_CACHE[cache_key] = pairs
        all_pairs.extend(pairs)

    return all_pairs


# ============================================================================
# Trajectory generation
# ============================================================================

def generate_wordladder_trajectories(
    latent_id: str,
    num_episodes: int = 5,
    n_trajectories: int = 100,
    seed: int = 42,
    candidate_pool_path: Optional[str] = None,
    output_dir: str = "data/eval/wordladder/",
    filtered_pool_dir: str = "",
    env_params: Optional[Dict[str, Any]] = None,
    word_lengths: List[int] = [3, 4, 5],
) -> List[Trajectory]:
    """Generate trajectory JSONs for word ladder.

    Can use:
    1. Pre-filtered pool directory (fastest — from filter_pools.py)
    2. Pre-generated word pairs from candidate_pool_path ("start target" per line)
    3. Auto-generated pairs via BFS (builds graph once per word length, then filters)
    """
    if env_params is None:
        env_params = {"max_turns_per_episode": 20}

    latent = get_latent("wordladder", latent_id)

    # ── Generator-based latents (hub_word, restricted_vocab) ──
    if latent.latent_mode == "generator":
        trajectories = []
        for traj_idx in range(n_trajectories):
            traj_seed = seed + traj_idx
            # Seed global random for generator compatibility
            old_state = random.getstate()
            random.seed(traj_seed)

            context: Dict = {}
            episodes = []
            try:
                for ep_idx in range(num_episodes):
                    config = latent.generate_episode_config(
                        env_params, ep_idx, num_episodes, context
                    )
                    _enrich_episode_config(config)
                    episodes.append(config)
            except ValueError as e:
                random.setstate(old_state)
                print(f"  SKIPPING {latent_id} traj_{traj_idx}: {e}")
                continue

            # Check if pool was too small (too many duplicate episodes)
            pairs = [(e.get("start_word"), e.get("target_word")) for e in episodes]
            unique = len(set(pairs))
            if unique < len(pairs) * 0.5:  # More than half are duplicates
                random.setstate(old_state)
                n_valid = len(context.get("valid_pairs", []))
                print(f"  WARNING: {latent_id} traj_{traj_idx}: only {unique}/{len(pairs)} unique episodes (pool size: {n_valid})")

            random.setstate(old_state)

            traj = Trajectory(
                trajectory_id=f"traj_{traj_idx:03d}",
                latent_id=latent_id,
                episodes=episodes,
                metadata={
                    "seed": traj_seed,
                    "num_episodes": num_episodes,
                    "env_params": env_params,
                    "context": {k: v for k, v in context.items()
                                if isinstance(v, (int, float, str, bool))
                                or (isinstance(v, list) and len(v) < 100)
                                or (isinstance(v, tuple) and len(v) < 100)},
                },
            )
            trajectories.append(traj)

        manifest = Manifest(
            env_name="wordladder",
            latent_id=latent_id,
            num_episodes=num_episodes,
            seed=seed,
            n_trajectories=n_trajectories,
            trajectory_files=[],
            extra={"env_params": env_params, "latent_mode": "generator"},
        )
        save_dataset(trajectories, output_dir, manifest)
        return trajectories

    # ── Filter-based latents (all others) ──
    assert latent.latent_mode == "filter", f"Latent '{latent_id}' has unknown mode"

    # Try pre-filtered pool first
    filtered = None
    if filtered_pool_dir:
        from latentgym.data.filter_pools import load_filtered_pool
        raw = load_filtered_pool(filtered_pool_dir, latent_id)
        if raw:
            filtered = [(parts[0], parts[1]) for line in raw for parts in [line.split()] if len(parts) >= 2] if isinstance(raw[0], str) else raw
            print(f"Latent '{latent_id}': loaded {len(filtered)} pre-filtered pairs")

    # Fall back to loading + runtime filtering
    all_pairs = []
    if not filtered:
        if candidate_pool_path:
            with open(candidate_pool_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        all_pairs.append((parts[0].lower(), parts[1].lower()))
        else:
            all_pairs = _generate_word_pairs(
                word_lengths=word_lengths,
                min_distance=2,
                max_distance=6,
            )

        filtered = [p for p in all_pairs if latent.filter_candidate(p)]
        print(f"Latent '{latent_id}': {len(filtered)}/{len(all_pairs)} pairs pass filter")

    if not filtered:
        raise ValueError(f"No word pairs pass filter for latent '{latent_id}'")

    # Deduplicate: treat (a, b) and (b, a) as the same pair — keep canonical form
    seen = set()
    deduped = []
    for p in filtered:
        key = tuple(sorted([p[0], p[1]]))
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    if len(deduped) < len(filtered):
        print(f"  Deduplicated: {len(filtered)} → {len(deduped)} pairs (removed reverse duplicates)")
    filtered = deduped

    # Group by word length, keep only lengths with enough pairs for one trajectory
    by_length: Dict[int, List[Tuple[str, str]]] = {}
    for p in filtered:
        wl = len(p[0])
        by_length.setdefault(wl, []).append(p)
    by_length = {wl: pairs for wl, pairs in by_length.items() if len(pairs) >= num_episodes}
    available_lengths = sorted(by_length.keys())

    if not available_lengths:
        total = sum(len(v) for v in by_length.values()) if by_length else len(filtered)
        print(f"  SKIPPING '{latent_id}': no word length has >= {num_episodes} pairs (total filtered: {len(filtered)})")
        return []

    # Each pair can appear in at most MAX_PAIR_REUSE trajectories
    MAX_PAIR_REUSE = 16

    # Cap trajectories by total reuse budget: total_usable * MAX_PAIR_REUSE / num_episodes.
    total_usable = sum(len(v) for v in by_length.values())
    actual_n_trajectories = min(n_trajectories, total_usable * MAX_PAIR_REUSE // num_episodes)
    if actual_n_trajectories < n_trajectories:
        print(f"  '{latent_id}': {total_usable} usable pairs (×{MAX_PAIR_REUSE} reuse) — creating {actual_n_trajectories}/{n_trajectories} trajectories")
    best_wl = max(available_lengths, key=lambda wl: len(by_length[wl]))
    full_pool = list(by_length[best_wl])
    pair_usage: Dict[Tuple[str, str], int] = {p: 0 for p in full_pool}

    trajectories = []
    for traj_idx in range(actual_n_trajectories):
        rng = random.Random(seed + traj_idx)
        # Available pairs: those used fewer than MAX_PAIR_REUSE times
        available = [p for p in full_pool if pair_usage[p] < MAX_PAIR_REUSE]
        if len(available) < num_episodes:
            # Not enough fresh pairs — stop creating trajectories
            break
        selected = rng.sample(available, num_episodes)
        for p in selected:
            pair_usage[p] += 1
        episodes = []
        for pair in selected:
            config = {
                "start_word": pair[0],
                "target_word": pair[1],
                "max_turns_per_episode": env_params.get("max_turns_per_episode", env_params.get("max_turns", 20)),
            }
            _enrich_episode_config(config)
            episodes.append(config)
        traj = Trajectory(
            trajectory_id=f"traj_{traj_idx:03d}",
            latent_id=latent_id,
            episodes=episodes,
        )
        trajectories.append(traj)

    manifest = Manifest(
        env_name="wordladder",
        latent_id=latent_id,
        num_episodes=num_episodes,
        seed=seed,
        n_trajectories=n_trajectories,
        trajectory_files=[],
        extra={
            "env_params": env_params,
            "total_pairs": len(all_pairs) if all_pairs else len(filtered),
            "filtered_pairs": len(filtered),
        },
    )
    save_dataset(trajectories, output_dir, manifest)

    return trajectories
