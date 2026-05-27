"""Three feed policies that select 5 posts per turn from a topic-filtered pool.

- random: uniform sample (seeded)
- recency: deterministic chronological by post id
- engagement_max: epsilon-greedy bandit (eps=0.1) over (stance, intensity) buckets.
  Reward: like/share = +1, skip = -0.2. Top-scoring buckets get served; ties broken
  by recency. Posts are not repeated within a single run.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def load_pool(jsonl_path: Path, topic: str | None = None) -> list[dict]:
    posts = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            if topic is None or p["topic"] == topic:
                posts.append(p)
    posts.sort(key=lambda p: p["id"])
    return posts


class FeedPolicy:
    def __init__(self, name: str, pool: list[dict], seed: int = 0):
        self.name = name
        self.pool = pool
        self.seed = seed
        self.served: set = set()

    def next_batch(self, k: int = 5) -> list[dict]:
        raise NotImplementedError

    def update(self, reactions: list[dict]) -> None:
        pass


class RandomFeed(FeedPolicy):
    def __init__(self, name: str, pool: list[dict], seed: int = 0):
        super().__init__(name, pool, seed)
        self.rng = random.Random(seed)

    def next_batch(self, k: int = 5) -> list[dict]:
        unseen = [p for p in self.pool if p["id"] not in self.served]
        if len(unseen) < k:
            self.served.clear()
            unseen = list(self.pool)
        choice = self.rng.sample(unseen, k)
        for p in choice:
            self.served.add(p["id"])
        return choice


class RecencyFeed(FeedPolicy):
    """Chronological — earliest unseen posts first."""

    def __init__(self, name: str, pool: list[dict], seed: int = 0):
        super().__init__(name, pool, seed)
        self._order = sorted(self.pool, key=lambda p: p["id"])
        self._idx = 0

    def next_batch(self, k: int = 5) -> list[dict]:
        out: list[dict] = []
        while len(out) < k:
            if self._idx >= len(self._order):
                self._idx = 0
            p = self._order[self._idx]
            self._idx += 1
            out.append(p)
        for p in out:
            self.served.add(p["id"])
        return out


class EngagementMaxFeed(FeedPolicy):
    """epsilon-greedy bandit over (stance, intensity) buckets."""

    def __init__(self, name: str, pool: list[dict], seed: int = 0, eps: float = 0.1):
        super().__init__(name, pool, seed)
        self.eps = eps
        self.rng = random.Random(seed)
        self.scores: dict[tuple, float] = {}
        for p in self.pool:
            self.scores.setdefault((p["stance"], p["intensity"]), 0.0)

    def _bucket(self, p: dict) -> tuple:
        return (p["stance"], p["intensity"])

    def next_batch(self, k: int = 5) -> list[dict]:
        unseen = [p for p in self.pool if p["id"] not in self.served]
        if len(unseen) < k:
            self.served.clear()
            unseen = list(self.pool)
        chosen: list[dict] = []
        for _ in range(k):
            avail = [p for p in unseen if p["id"] not in {c["id"] for c in chosen}]
            if not avail:
                break
            if self.rng.random() < self.eps:
                pick = self.rng.choice(avail)
            else:
                avail.sort(key=lambda p: (-self.scores[self._bucket(p)], p["id"]))
                pick = avail[0]
            chosen.append(pick)
        for p in chosen:
            self.served.add(p["id"])
        return chosen

    def update(self, reactions: list[dict]) -> None:
        for r in reactions:
            p = r["post"]
            a = r.get("action", "SKIP").upper()
            delta = 1.0 if a in ("LIKE", "SHARE") else -0.2
            self.scores[self._bucket(p)] = self.scores.get(self._bucket(p), 0.0) + delta


class ShuffledFeed(FeedPolicy):
    """Mitigation: engagement-max but each served batch is randomly shuffled.

    Same content distribution but breaks within-batch order coherence.
    """
    def __init__(self, name: str, pool: list[dict], seed: int = 0, eps: float = 0.1):
        super().__init__(name, pool, seed)
        self._em = EngagementMaxFeed("inner", pool, seed=seed, eps=eps)
        self.rng = random.Random(seed + 7919)

    def next_batch(self, k: int = 5) -> list[dict]:
        batch = self._em.next_batch(k)
        self.rng.shuffle(batch)
        for p in batch:
            self.served.add(p["id"])
        return batch

    def update(self, reactions: list[dict]) -> None:
        self._em.update(reactions)


class BalancedFeed(FeedPolicy):
    """Mitigation: serve exactly k=5 posts, one per stance bucket (forcing diversity).

    Stances are -2, -1, 0, +1, +2 — 5 buckets total. Within each bucket, pick the
    first unseen post by id.
    """
    def __init__(self, name: str, pool: list[dict], seed: int = 0):
        super().__init__(name, pool, seed)
        self.rng = random.Random(seed)
        self._by_stance: dict[int, list[dict]] = {}
        for p in self.pool:
            self._by_stance.setdefault(p["stance"], []).append(p)
        for s in self._by_stance:
            self._by_stance[s].sort(key=lambda p: p["id"])
        # Cursors per stance
        self._cursors = {s: 0 for s in self._by_stance}

    def next_batch(self, k: int = 5) -> list[dict]:
        out: list[dict] = []
        stances = sorted(self._by_stance.keys())  # -2..+2 typically
        for s in stances[:k]:
            posts_s = self._by_stance[s]
            if not posts_s:
                continue
            idx = self._cursors[s] % len(posts_s)
            out.append(posts_s[idx])
            self._cursors[s] += 1
        # If fewer than k stances available, fill with random posts
        while len(out) < k:
            avail = [p for p in self.pool if p["id"] not in {x["id"] for x in out}]
            if not avail:
                break
            out.append(self.rng.choice(avail))
        for p in out:
            self.served.add(p["id"])
        return out


class DisclosedRecencyFeed(RecencyFeed):
    """Mitigation: recency feed but the agent is TOLD it's recency-ordered.

    Hooks into the persona via a runtime annotation; the agent_loop reads the
    `disclosure` attribute when constructing the system message.
    """
    disclosure = (
        " NOTE: the posts you are about to see are ordered strictly by recency "
        "(oldest first), not by personalized engagement. Take that into account."
    )


def make_policy(name: str, pool: list[dict], seed: int) -> FeedPolicy:
    name = name.lower()
    if name == "random":
        return RandomFeed(name="random", pool=pool, seed=seed)
    if name == "recency":
        return RecencyFeed(name="recency", pool=pool, seed=seed)
    if name == "engagement_max":
        return EngagementMaxFeed(name="engagement_max", pool=pool, seed=seed)
    if name == "shuffled":
        return ShuffledFeed(name="shuffled", pool=pool, seed=seed)
    if name == "balanced":
        return BalancedFeed(name="balanced", pool=pool, seed=seed)
    if name == "disclosed":
        return DisclosedRecencyFeed(name="disclosed", pool=pool, seed=seed)
    raise ValueError(f"Unknown policy: {name}")


if __name__ == "__main__":
    pool_path = Path("posts/pool.jsonl")
    pool = load_pool(pool_path, topic="remote_work")
    print(f"Pool: {len(pool)} posts for remote_work")
    for name in ["random", "recency", "engagement_max"]:
        pol = make_policy(name, pool, seed=0)
        batch = pol.next_batch(5)
        print(f"\n{name} first batch:")
        for p in batch:
            print(f"  [s={p['stance']:+d} {p['intensity']}] {p['text'][:100]}")
