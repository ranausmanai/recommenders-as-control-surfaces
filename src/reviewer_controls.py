"""Seeded reviewer-control experiments for the adversarial-feeds paper.

This runner provides:
  * feed-free default measurement;
  * matched stance-composition conditions from one benign topic-specific pool;
  * pure-order conditions that expose the exact same post IDs in different order.

Every completed rollout is append-only and contains its exposure trace. Existing
records are skipped by the queue, so interrupted runs are safely resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agent_loop import PERSONA, build_reaction_prompt
from decision_shift import DECISIONS, parse_choice
from feed_policies import load_pool


ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_CONTEXT_TOKENS = 16384
MATCHED_POOL = ROOT / "posts" / "pool_gemma.jsonl"
MATCHED_CONDITIONS = {
    "matched_balanced",
    "matched_rto",
    "matched_remote",
    "order_interleaved",
    "order_rto_last",
    "order_remote_last",
}
ORDER_CONDITIONS = {
    "order_interleaved",
    "order_rto_last",
    "order_remote_last",
}
RTO_STANCES = {-2, -1}
REMOTE_STANCES = {1, 2}
NEUTRAL_STANCES = {0}
INTENSITIES = ("calm", "measured", "heated", "inflammatory")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_thinking(text: str) -> str:
    return re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()


def ollama_chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    seed: int,
    max_new_tokens: int,
    temperature: float = 0.7,
    retries: int = 3,
) -> tuple[str, dict[str, Any]]:
    """Call Ollama with an explicit seed and bounded retries."""
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": max_new_tokens,
                "seed": seed,
                "num_ctx": OLLAMA_CONTEXT_TOKENS,
            },
            "think": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=360) as response:
                result = json.loads(response.read())
            text = _strip_thinking(result.get("message", {}).get("content", ""))
            meta = {
                "total_duration_ns": result.get("total_duration"),
                "load_duration_ns": result.get("load_duration"),
                "prompt_eval_count": result.get("prompt_eval_count"),
                "eval_count": result.get("eval_count"),
                "eval_duration_ns": result.get("eval_duration"),
            }
            return text, meta
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Ollama failed after {retries} attempts: {last_error}")


def validate_matched_pool(pool: list[dict[str, Any]]) -> None:
    if len(pool) != 100:
        raise ValueError(f"expected 100 matched-pool posts, found {len(pool)}")
    ids = [post["id"] for post in pool]
    if len(set(ids)) != len(ids):
        raise ValueError("matched pool contains duplicate IDs")
    by_cell = Counter((post["stance"], post["intensity"]) for post in pool)
    expected = {
        (stance, intensity)
        for stance in (-2, -1, 0, 1, 2)
        for intensity in INTENSITIES
    }
    if set(by_cell) != expected or any(by_cell[cell] != 5 for cell in expected):
        raise ValueError(f"matched pool is not 5-per-cell balanced: {by_cell}")
    if any(post.get("topic") != "remote_work" for post in pool):
        raise ValueError("matched pool contains a non-remote_work item")


def _bucketed(pool: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for post in pool:
        buckets[(post["stance"], post["intensity"])].append(post)
    for posts in buckets.values():
        posts.sort(key=lambda post: post["id"])
    return buckets


def _rotated(
    posts: list[dict[str, Any]], *, seed: int, salt: int
) -> list[dict[str, Any]]:
    out = list(posts)
    random.Random(seed * 1009 + salt).shuffle(out)
    return out


def _side_posts(
    buckets: dict[tuple[int, str], list[dict[str, Any]]],
    stances: set[int],
    *,
    seed: int,
    salt: int,
) -> list[dict[str, Any]]:
    """Return all 40 posts for one side, balanced by strength and intensity."""
    out: list[dict[str, Any]] = []
    for stance in sorted(stances):
        for intensity_index, intensity in enumerate(INTENSITIES):
            out.extend(
                _rotated(
                    buckets[(stance, intensity)],
                    seed=seed,
                    salt=salt + stance * 31 + intensity_index,
                )
            )
    random.Random(seed * 2029 + salt).shuffle(out)
    return out


def _neutral_posts(
    buckets: dict[tuple[int, str], list[dict[str, Any]]], *, seed: int
) -> list[dict[str, Any]]:
    """Select ten neutral posts with a near-equal intensity distribution."""
    counts = dict(zip(INTENSITIES, (3, 3, 2, 2)))
    out: list[dict[str, Any]] = []
    for index, intensity in enumerate(INTENSITIES):
        candidates = _rotated(
            buckets[(0, intensity)], seed=seed, salt=700 + index
        )
        out.extend(candidates[: counts[intensity]])
    random.Random(seed * 3001 + 701).shuffle(out)
    return out


def _balanced_side_subset(
    side: list[dict[str, Any]], *, seed: int, salt: int
) -> list[dict[str, Any]]:
    """Select 20 of 40 with 10 per stance and 5 per intensity."""
    by_cell: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for post in side:
        by_cell[(post["stance"], post["intensity"])].append(post)
    stances = sorted({post["stance"] for post in side})
    if len(stances) != 2:
        raise ValueError(f"expected two stance levels for a side, found {stances}")
    count_rows = ((3, 3, 2, 2), (2, 2, 3, 3))
    out: list[dict[str, Any]] = []
    for stance_index, stance in enumerate(stances):
        for intensity_index, intensity in enumerate(INTENSITIES):
            candidates = _rotated(
                by_cell[(stance, intensity)],
                seed=seed,
                salt=salt + stance_index * 17 + intensity_index,
            )
            out.extend(candidates[: count_rows[stance_index][intensity_index]])
    random.Random(seed * 4001 + salt).shuffle(out)
    return out


def _make_batches(
    rto: list[dict[str, Any]],
    neutral: list[dict[str, Any]],
    remote: list[dict[str, Any]],
    *,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Distribute supplied groups evenly into ten shuffled five-post batches."""
    if len(rto) + len(neutral) + len(remote) != 50:
        raise ValueError("a rollout must contain exactly 50 posts")
    batches: list[list[dict[str, Any]]] = [[] for _ in range(10)]
    for group, offset in ((rto, 0), (neutral, 3), (remote, 6)):
        for index, post in enumerate(group):
            batches[(index + offset) % 10].append(post)
    if any(len(batch) != 5 for batch in batches):
        raise ValueError(f"batch construction failed: {[len(batch) for batch in batches]}")
    for index, batch in enumerate(batches):
        random.Random(seed * 5003 + index).shuffle(batch)
    return batches


def _order_batches(
    rto: list[dict[str, Any]],
    neutral: list[dict[str, Any]],
    remote: list[dict[str, Any]],
    *,
    condition: str,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Arrange the same 50 IDs differently while preserving five-post turns."""
    if not (len(rto) == 20 and len(neutral) == 10 and len(remote) == 20):
        raise ValueError("order conditions require 20/10/20 posts")
    if condition == "order_interleaved":
        return _make_batches(rto, neutral, remote, seed=seed)

    early, late = (remote, rto) if condition == "order_rto_last" else (rto, remote)
    sequence = list(early) + list(neutral) + list(late)
    batches = [sequence[index : index + 5] for index in range(0, 50, 5)]
    for index, batch in enumerate(batches):
        random.Random(seed * 6007 + index).shuffle(batch)
    return batches


def build_exposure(
    pool: list[dict[str, Any]], condition: str, seed: int
) -> list[list[dict[str, Any]]]:
    """Build a deterministic ten-batch exposure for one matched condition."""
    if condition not in MATCHED_CONDITIONS:
        raise ValueError(f"unknown matched condition: {condition}")
    validate_matched_pool(pool)
    buckets = _bucketed(pool)
    rto_all = _side_posts(buckets, RTO_STANCES, seed=seed, salt=101)
    remote_all = _side_posts(buckets, REMOTE_STANCES, seed=seed, salt=211)
    neutral = _neutral_posts(buckets, seed=seed)

    if condition == "matched_rto":
        return _make_batches(rto_all, neutral, [], seed=seed)
    if condition == "matched_remote":
        return _make_batches([], neutral, remote_all, seed=seed)

    rto = _balanced_side_subset(rto_all, seed=seed, salt=307)
    remote = _balanced_side_subset(remote_all, seed=seed, salt=401)
    if condition == "matched_balanced":
        return _make_batches(rto, neutral, remote, seed=seed)
    return _order_batches(
        rto, neutral, remote, condition=condition, seed=seed
    )


def summarize_exposure(batches: list[list[dict[str, Any]]]) -> dict[str, Any]:
    flat = [post for batch in batches for post in batch]
    return {
        "post_ids": [post["id"] for post in flat],
        "stances": [post["stance"] for post in flat],
        "intensities": [post["intensity"] for post in flat],
        "stance_counts": dict(sorted(Counter(post["stance"] for post in flat).items())),
        "intensity_counts": dict(
            sorted(Counter(post["intensity"] for post in flat).items())
        ),
    }


def run_no_feed(model: str, topic: str, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    messages = [
        {"role": "system", "content": PERSONA},
        {"role": "user", "content": DECISIONS[topic]},
    ]
    raw, meta = ollama_chat(
        model,
        messages,
        seed=seed * 10000 + 9001,
        max_new_tokens=80,
    )
    return {
        "experiment": "no_feed",
        "model": model,
        "topic": topic,
        "condition": "no_feed",
        "seed": seed,
        "choice": parse_choice(raw),
        "raw": raw,
        "elapsed_s": time.perf_counter() - started,
        "ollama_meta": meta,
        "ollama_num_ctx": OLLAMA_CONTEXT_TOKENS,
    }


def run_matched(
    model: str,
    condition: str,
    seed: int,
    *,
    pool: list[dict[str, Any]],
    max_history_turns: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    batches = build_exposure(pool, condition, seed)
    base = [{"role": "system", "content": PERSONA}]
    history: list[dict[str, Any]] = []

    for turn, posts in enumerate(batches):
        messages = list(base)
        for item in history[-max_history_turns:]:
            messages.extend(
                [
                    {"role": "user", "content": item["reaction_prompt"]},
                    {"role": "assistant", "content": item["reaction_raw"]},
                ]
            )
        reaction_prompt = build_reaction_prompt(posts)
        messages.append({"role": "user", "content": reaction_prompt})
        reaction_raw, meta = ollama_chat(
            model,
            messages,
            seed=seed * 10000 + turn + 1,
            max_new_tokens=180,
        )
        history.append(
            {
                "turn": turn,
                "post_ids": [post["id"] for post in posts],
                "stances": [post["stance"] for post in posts],
                "intensities": [post["intensity"] for post in posts],
                "reaction_prompt": reaction_prompt,
                "reaction_raw": reaction_raw,
                "ollama_meta": meta,
            }
        )

    messages = list(base)
    for item in history[-max_history_turns:]:
        messages.extend(
            [
                {"role": "user", "content": item["reaction_prompt"]},
                {"role": "assistant", "content": item["reaction_raw"]},
            ]
        )
    messages.append({"role": "user", "content": DECISIONS["remote_work"]})
    raw, decision_meta = ollama_chat(
        model,
        messages,
        seed=seed * 10000 + 9001,
        max_new_tokens=80,
    )
    experiment = "pure_order" if condition in ORDER_CONDITIONS else "matched_selection"
    return {
        "experiment": experiment,
        "model": model,
        "topic": "remote_work",
        "condition": condition,
        "seed": seed,
        "n_turns": 10,
        "max_history_turns": max_history_turns,
        "choice": parse_choice(raw),
        "raw": raw,
        "elapsed_s": time.perf_counter() - started,
        "pool_path": str(MATCHED_POOL.relative_to(ROOT)),
        "pool_sha256": file_sha256(MATCHED_POOL),
        "exposure": summarize_exposure(batches),
        "turn_trace": history,
        "decision_meta": decision_meta,
        "ollama_num_ctx": OLLAMA_CONTEXT_TOKENS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--topic", default="remote_work")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-history-turns", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.condition == "no_feed":
        record = run_no_feed(args.model, args.topic, args.seed)
    else:
        pool = load_pool(MATCHED_POOL, topic="remote_work")
        record = run_matched(
            args.model,
            args.condition,
            args.seed,
            pool=pool,
            max_history_turns=args.max_history_turns,
        )
    if record["choice"] not in {"A", "B", "C"}:
        raise RuntimeError(
            f"unparsable decision for {record['model']}/{record['condition']}/"
            f"s{record['seed']}: {record['raw']!r}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(
        f"{record['model']}/{record['topic']}/{record['condition']}/"
        f"s{record['seed']} -> {record['choice']} ({record['elapsed_s']:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
