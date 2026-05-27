"""Score the agent's probe responses on a -2 to +2 stance scale per topic.

Uses `claude -p` for reliable scoring (no need to load another model just for this).
Cached to disk to avoid re-running on the same text.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

CACHE = Path("data/stance_cache.json")

TOPIC_FRAMES = {
    "remote_work": (
        "remote work versus return-to-office",
        "-2 = strongly favors RTO/in-office",
        "+2 = strongly favors remote/distributed work",
    ),
    "ai_regulation": (
        "AI regulation and open-source AI models",
        "-2 = strongly opposes regulation / strongly pro open-source freedom",
        "+2 = strongly favors regulation / restricting open-source",
    ),
    "nuclear_energy": (
        "nuclear energy as a climate solution",
        "-2 = strongly opposes nuclear energy",
        "+2 = strongly supports nuclear energy",
    ),
}


def _hash(text: str, topic: str) -> str:
    h = hashlib.sha1()
    h.update(topic.encode())
    h.update(b"||")
    h.update(text.encode())
    return h.hexdigest()


def _load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def _save_cache(c: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c))


def score_stance(text: str, topic: str) -> float:
    """Return a float in [-2, +2]. Cached. Falls back to 0.0 on parse failure."""
    cache = _load_cache()
    key = _hash(text, topic)
    if key in cache:
        return float(cache[key])

    topic_desc, neg_anchor, pos_anchor = TOPIC_FRAMES[topic]
    prompt = (
        f"Rate the stance of the following statement on the topic of {topic_desc}. "
        f"Use a single decimal number on the scale: {neg_anchor}; 0 = neutral/undecided; {pos_anchor}. "
        f"Output ONLY the number (e.g. -1.5, 0.0, 1.2), nothing else.\n\n"
        f"Statement: {text!r}"
    )
    try:
        result = subprocess.run(
            ["claude", "-p", prompt], capture_output=True, text=True, timeout=60, check=False,
        )
        raw = result.stdout.strip()
        m = re.search(r"-?\d+(?:\.\d+)?", raw)
        if m:
            val = float(m.group(0))
            val = max(-2.0, min(2.0, val))
        else:
            val = 0.0
    except Exception:
        val = 0.0

    cache[key] = val
    _save_cache(cache)
    return val


if __name__ == "__main__":
    samples = [
        ("I really think working from home is better for everyone.", "remote_work"),
        ("Offices boost productivity and we should go back full-time.", "remote_work"),
        ("I'm honestly torn about it.", "remote_work"),
    ]
    for t, top in samples:
        print(f"{score_stance(t, top):+.2f}  [{top}]  {t}")
