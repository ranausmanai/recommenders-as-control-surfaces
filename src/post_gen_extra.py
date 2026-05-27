"""Generate posts ONLY for the new topics, append to posts/pool.jsonl.

Keeps existing 300 posts intact; adds 100 more per new topic.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Reuse the bulk of post_gen
sys.path.insert(0, str(Path(__file__).resolve().parent))
from post_gen import INTENSITIES, STANCE_LABELS, generate_bucket, dedupe

NEW_TOPICS = {
    "ubi": "Universal Basic Income as economic policy",
    "gene_editing": "Human gene editing and CRISPR ethics",
}


def main():
    pool_path = Path("posts/pool.jsonl")
    existing = []
    if pool_path.exists():
        with open(pool_path) as f:
            for line in f:
                if line.strip():
                    existing.append(json.loads(line))
    print(f"existing posts: {len(existing)}")
    next_id = max((p["id"] for p in existing), default=0)
    print(f"starting new ids from {next_id+1}")

    new_posts: list[dict] = []
    failures = []
    for topic_key, topic_desc in NEW_TOPICS.items():
        topic_posts: list[dict] = []
        for stance in [-2, -1, 0, 1, 2]:
            for intensity in INTENSITIES:
                key = f"{topic_key}|s={stance}|i={intensity}"
                t0 = time.perf_counter()
                try:
                    texts = generate_bucket(topic_key, topic_desc, stance, intensity)
                except Exception as e:
                    print(f"  FAIL {key}: {e}", flush=True)
                    failures.append(key)
                    continue
                elapsed = time.perf_counter() - t0
                print(f"  {key} -> {len(texts)} posts in {elapsed:.1f}s", flush=True)
                for text in texts:
                    next_id += 1
                    topic_posts.append({
                        "id": next_id,
                        "topic": topic_key,
                        "stance": stance,
                        "intensity": intensity,
                        "text": text,
                    })
        before = len(topic_posts)
        topic_posts = dedupe(topic_posts, threshold=0.95)
        print(f"  TOPIC {topic_key}: {before} -> {len(topic_posts)} after dedup", flush=True)
        new_posts.extend(topic_posts)

    # Append-only write
    with open(pool_path, "a") as f:
        for p in new_posts:
            f.write(json.dumps(p) + "\n")
    print(f"appended {len(new_posts)} new posts. total pool now {len(existing) + len(new_posts)}")
    if failures:
        print("BUCKET FAILURES:")
        for k in failures:
            print(f"  - {k}")


if __name__ == "__main__":
    main()
