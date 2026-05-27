"""Generate ~300 social-media posts via `claude -p` subprocess.

Topics × stances × intensities × 5 posts = 100 per topic = 300 total.

Stances: -2 strong-against, -1 lean-against, 0 neutral, +1 lean-for, +2 strong-for.
Intensities: calm, measured, heated, inflammatory.

We batch ALL 5 posts for one (stance, intensity) bucket into a single `claude -p` call.
That is 5 stances × 4 intensities × 3 topics = 60 claude calls.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

TOPICS = {
    "remote_work": "Remote work vs return-to-office policies",
    "ai_regulation": "AI regulation and the role of open-source models",
    "nuclear_energy": "Nuclear energy as a climate solution",
    "ubi": "Universal Basic Income as economic policy",
    "gene_editing": "Human gene editing and CRISPR ethics",
}

STANCE_LABELS = {
    -2: "strongly against the position",
    -1: "leaning against the position",
     0: "neutral / undecided on the position",
     1: "leaning in favor of the position",
     2: "strongly in favor of the position",
}

INTENSITIES = {
    "calm": "calm and reflective tone, no rhetorical heat",
    "measured": "measured and analytical, citing evidence or examples",
    "heated": "heated and emphatic, but not personal attacks",
    "inflammatory": "inflammatory rhetoric, sarcasm or moral outrage, but no slurs and no specific persons named",
}


def claude_p(prompt: str, retries: int = 2, timeout: int = 180) -> str:
    """Call `claude -p "<prompt>"` and return stdout."""
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"claude -p failed after {retries+1} attempts")


def parse_posts(raw: str) -> list[str]:
    """Extract text fields from a JSON array possibly wrapped in markdown fences."""
    cleaned = raw.strip()
    # Strip markdown code fences if present.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    # Find first '[' and last ']' to be robust to preamble.
    i, j = cleaned.find("["), cleaned.rfind("]")
    if i == -1 or j == -1:
        raise ValueError(f"Could not find JSON array in claude output: {raw[:200]!r}")
    try:
        arr = json.loads(cleaned[i : j + 1])
    except json.JSONDecodeError:
        # Try lenient repair: replace common smart quotes.
        repaired = cleaned[i : j + 1].replace("“", '"').replace("”", '"')
        arr = json.loads(repaired)
    posts = []
    for item in arr:
        if isinstance(item, str):
            posts.append(item.strip())
        elif isinstance(item, dict) and "text" in item:
            posts.append(str(item["text"]).strip())
    return [p for p in posts if p]


def generate_bucket(topic_key: str, topic_desc: str, stance: int, intensity: str) -> list[str]:
    """Generate 5 posts for one (topic, stance, intensity) cell."""
    stance_desc = STANCE_LABELS[stance]
    intensity_desc = INTENSITIES[intensity]
    prompt = (
        f"Generate exactly 5 short social media posts (Twitter/X style, 1-3 sentences each) "
        f"about the topic: {topic_desc}. "
        f"Each post must reflect a stance that is {stance_desc}. "
        f"Each post must use a {intensity_desc}. "
        f"Avoid hashtag spam, avoid emojis. Each post should be self-contained and realistic, "
        f"as if a real user posted it. Do NOT identify or attack specific named individuals. "
        f"Vary the angle/argument across the 5 posts. "
        f"Return a JSON array of 5 objects, each with a single key \"text\". "
        f"Output ONLY the JSON array. No preamble, no markdown fences."
    )
    raw = claude_p(prompt)
    return parse_posts(raw)


def cosine(a: list[float], b: list[float]) -> float:
    import math
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return s / (na * nb + 1e-9)


def simple_embed(text: str) -> list[float]:
    """Cheap character-trigram TF embedding for dedup. Not great but fast.

    For 300 posts this is fine. If needed later we can swap for a real
    sentence-transformer.
    """
    counts: dict[str, int] = {}
    t = text.lower()
    for i in range(len(t) - 2):
        tri = t[i : i + 3]
        counts[tri] = counts.get(tri, 0) + 1
    keys = sorted(counts)
    vec = [counts[k] for k in keys]
    # Pad to fixed dictionary later; for cosine pairwise we'll align dicts.
    return counts  # actually return dict for now


def cos_from_dicts(a: dict, b: dict) -> float:
    import math
    keys = set(a) | set(b)
    s = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return s / (na * nb + 1e-9)


def dedupe(posts: list[dict], threshold: float = 0.95) -> list[dict]:
    keep: list[dict] = []
    embeds: list[dict] = []
    for p in posts:
        emb = simple_embed(p["text"])
        if any(cos_from_dicts(emb, e) > threshold for e in embeds):
            continue
        keep.append(p)
        embeds.append(emb)
    return keep


def main(outpath: Path):
    rng = random.Random(42)
    all_posts: list[dict] = []
    post_id = 0
    bucket_failures: list[str] = []

    for topic_key, topic_desc in TOPICS.items():
        topic_posts: list[dict] = []
        for stance in [-2, -1, 0, 1, 2]:
            for intensity in INTENSITIES:
                key = f"{topic_key}|s={stance}|i={intensity}"
                t0 = time.perf_counter()
                try:
                    texts = generate_bucket(topic_key, topic_desc, stance, intensity)
                except Exception as e:
                    print(f"  FAIL {key}: {e}", flush=True)
                    bucket_failures.append(key)
                    continue
                elapsed = time.perf_counter() - t0
                print(f"  {key} -> {len(texts)} posts in {elapsed:.1f}s", flush=True)
                for text in texts:
                    post_id += 1
                    topic_posts.append({
                        "id": post_id,
                        "topic": topic_key,
                        "stance": stance,
                        "intensity": intensity,
                        "text": text,
                    })
        # Dedupe within topic
        before = len(topic_posts)
        topic_posts = dedupe(topic_posts, threshold=0.95)
        after = len(topic_posts)
        print(f"  TOPIC {topic_key}: {before} -> {after} after dedup", flush=True)
        all_posts.extend(topic_posts)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        for p in all_posts:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(all_posts)} posts to {outpath}", flush=True)

    # Sanity sample
    print("\n--- Random samples (3 per topic) ---", flush=True)
    by_topic: dict[str, list[dict]] = {}
    for p in all_posts:
        by_topic.setdefault(p["topic"], []).append(p)
    for t, ps in by_topic.items():
        for s in rng.sample(ps, min(3, len(ps))):
            print(f"[{t} s={s['stance']:+d} {s['intensity']}] {s['text'][:200]}", flush=True)

    if bucket_failures:
        print(f"\nBUCKET FAILURES ({len(bucket_failures)}):", flush=True)
        for k in bucket_failures:
            print(f"  - {k}", flush=True)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("posts/pool.jsonl")
    main(out)
