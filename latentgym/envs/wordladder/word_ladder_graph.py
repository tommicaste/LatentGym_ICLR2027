"""Word ladder graph infrastructure for path-aware latents.

Pre-computes word graphs, pairwise distances, shortest paths, and hub words.
Used by filter functions that need to reason about solution paths, not just
start/target properties.

Lazily initialized — the graph is built on first use and cached.

Usage:
    graph = get_graph(word_length=4)

    # Distances
    graph.distance("cold", "warm")  # → 4

    # Hub check (triangle equality)
    graph.path_goes_through("cold", "warm", "cord")  # → True

    # All shortest paths
    graph.all_shortest_paths("cold", "warm")  # → [["cold","cord","word","ward","warm"], ...]

    # Hub words (most frequently on shortest paths)
    graph.hub_words(top_k=10)  # → ["cord", "core", "word", ...]
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Cache: (word_length,) → WordLadderGraph
_GRAPH_CACHE: Dict[int, "WordLadderGraph"] = {}


def get_graph(word_length: int = 4) -> "WordLadderGraph":
    """Get or build a cached word ladder graph for the given word length."""
    if word_length not in _GRAPH_CACHE:
        logger.info(f"Building word ladder graph for length {word_length}...")
        _GRAPH_CACHE[word_length] = WordLadderGraph(word_length)
    return _GRAPH_CACHE[word_length]


class WordLadderGraph:
    """Pre-computed word ladder graph with distances and path queries."""

    def __init__(self, word_length: int = 4):
        self.word_length = word_length
        self.words: List[str] = []
        self.word_set: Set[str] = set()
        self.neighbors: Dict[str, List[str]] = {}
        self._dist_cache: Dict[str, Dict[str, int]] = {}
        self._hub_cache: Optional[List[Tuple[str, int]]] = None

        self._build()

    def _build(self):
        """Load words and build neighbor map."""
        try:
            from nltk.corpus import words
            try:
                word_list = words.words("en-basic")
            except LookupError:
                import nltk
                nltk.download("words", quiet=True)
                word_list = words.words("en-basic")
        except ImportError:
            logger.warning("NLTK not available, using empty word list")
            return

        # Filter to target length, lowercase, alpha only
        self.words = sorted(set(
            w.lower() for w in word_list
            if len(w) == self.word_length and w.isalpha()
        ))
        self.word_set = set(self.words)
        logger.info(f"  {len(self.words)} words of length {self.word_length}")

        # Build neighbor map
        for word in self.words:
            nbs = []
            for i in range(self.word_length):
                for c in "abcdefghijklmnopqrstuvwxyz":
                    if c != word[i]:
                        cand = word[:i] + c + word[i + 1:]
                        if cand in self.word_set:
                            nbs.append(cand)
            self.neighbors[word] = nbs

        logger.info(f"  Graph built: {len(self.words)} nodes, "
                    f"{sum(len(v) for v in self.neighbors.values()) // 2} edges")

    def _ensure_bfs(self, source: str):
        """Run BFS from source if not cached."""
        if source in self._dist_cache:
            return
        dist = {source: 0}
        q = deque([source])
        while q:
            cur = q.popleft()
            for nb in self.neighbors.get(cur, []):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        self._dist_cache[source] = dist

    def distance(self, a: str, b: str) -> int:
        """Shortest path distance between two words. Returns -1 if unreachable."""
        if a not in self.word_set or b not in self.word_set:
            return -1
        self._ensure_bfs(a)
        return self._dist_cache[a].get(b, -1)

    def path_goes_through(self, start: str, target: str, hub: str) -> bool:
        """Check if some shortest path from start to target goes through hub.

        Uses triangle equality: dist(s,h) + dist(h,t) == dist(s,t)
        """
        d_st = self.distance(start, target)
        if d_st < 0:
            return False
        d_sh = self.distance(start, hub)
        d_ht = self.distance(hub, target)
        if d_sh < 0 or d_ht < 0:
            return False
        return d_sh + d_ht == d_st

    def all_shortest_paths(self, start: str, target: str, max_paths: int = 50) -> List[List[str]]:
        """Find all shortest paths between two words (capped at max_paths).

        Returns list of paths, where each path is a list of words.
        """
        if start not in self.word_set or target not in self.word_set:
            return []

        # BFS to build parent map
        dist = {start: 0}
        parents: Dict[str, List[str]] = {start: []}
        q = deque([start])
        found = False
        while q:
            cur = q.popleft()
            if cur == target:
                found = True
                # Don't break — finish this distance level for all shortest paths
                continue
            if found and dist[cur] >= dist[target]:
                break
            for nb in self.neighbors.get(cur, []):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    parents[nb] = [cur]
                    q.append(nb)
                elif dist[nb] == dist[cur] + 1:
                    parents[nb].append(cur)

        if target not in dist:
            return []

        # Backtrack to find all paths
        paths: List[List[str]] = []

        def backtrack(node: str, path: List[str]):
            if len(paths) >= max_paths:
                return
            if node == start:
                paths.append(list(reversed(path)))
                return
            for p in parents[node]:
                backtrack(p, path + [p])

        backtrack(target, [target])
        return paths

    def hub_words(self, top_k: int = 20, sample_pairs: int = 500) -> List[Tuple[str, int]]:
        """Find words that appear most frequently on shortest paths.

        Samples random pairs and counts how often each intermediate word
        appears on shortest paths. Returns [(word, count), ...] sorted by count.
        """
        if self._hub_cache is not None:
            return self._hub_cache[:top_k]

        import random
        rng = random.Random(42)

        # Sample pairs with distance >= 3
        valid_pairs = []
        sampled_words = rng.sample(self.words, min(100, len(self.words)))
        for start in sampled_words:
            self._ensure_bfs(start)
            for target, d in self._dist_cache[start].items():
                if 3 <= d <= 6:
                    valid_pairs.append((start, target))

        if not valid_pairs:
            self._hub_cache = []
            return []

        pairs = rng.sample(valid_pairs, min(sample_pairs, len(valid_pairs)))

        # Count intermediate word appearances
        counts: Dict[str, int] = {}
        for start, target in pairs:
            paths = self.all_shortest_paths(start, target, max_paths=5)
            for path in paths:
                for word in path[1:-1]:  # Exclude start and target
                    counts[word] = counts.get(word, 0) + 1

        self._hub_cache = sorted(counts.items(), key=lambda x: -x[1])
        return self._hub_cache[:top_k]

    def find_restricted_vocabulary(self, size: int = 40, seed: Optional[int] = None) -> Set[str]:
        """Find a densely connected subgraph of approximately `size` words.

        Starts from a high-degree word (randomized by seed) and greedily adds
        neighbors that maximize internal connectivity. Different seeds produce
        different vocabularies.
        """
        if not self.words:
            return set()

        import random as _rng

        # Sort by degree, then pick a starting word
        by_degree = sorted(self.words, key=lambda w: len(self.neighbors.get(w, [])), reverse=True)
        if seed is not None:
            # Pick from top 20 high-degree words randomly
            r = _rng.Random(seed)
            start_word = r.choice(by_degree[:min(20, len(by_degree))])
        else:
            start_word = by_degree[0]

        vocab = {start_word}

        # Greedily add words that have the most connections to existing vocab
        candidates = set(self.neighbors.get(start_word, []))
        while len(vocab) < size and candidates:
            best = max(candidates, key=lambda w: sum(1 for nb in self.neighbors.get(w, []) if nb in vocab))
            vocab.add(best)
            candidates.discard(best)
            candidates.update(nb for nb in self.neighbors.get(best, []) if nb not in vocab)

        return vocab

    def changed_positions(self, path: List[str]) -> Tuple[int, ...]:
        """For each step in the path, which position changed?"""
        positions = []
        for i in range(len(path) - 1):
            for j in range(len(path[i])):
                if path[i][j] != path[i + 1][j]:
                    positions.append(j)
                    break
        return tuple(positions)

    def letter_substitutions(self, path: List[str]) -> Tuple[Tuple[str, str], ...]:
        """For each step, what letter was substituted (old, new)?"""
        subs = []
        for i in range(len(path) - 1):
            for j in range(len(path[i])):
                if path[i][j] != path[i + 1][j]:
                    subs.append((path[i][j], path[i + 1][j]))
                    break
        return tuple(subs)

    def intermediates_match_pattern(self, path: List[str], pattern_fn) -> bool:
        """Check if all intermediate words (excluding start/target) match a pattern."""
        if len(path) <= 2:
            return True  # No intermediates
        return all(pattern_fn(w) for w in path[1:-1])
