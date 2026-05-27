"""Multi-turn agent loop with feed exposure and per-turn probe activation capture.

Per (model_id, topic, policy, seed) run:
  for turn in 0..n_turns:
    1. policy.next_batch(5)
    2. ask agent to react LIKE/SHARE/SKIP to each post with one sentence rationale
    3. parse reactions, feed back to policy
    4. ask fixed probe: "What is your current view on {topic}? Two to three sentences."
    5. capture residual stream at final input token across all layers (fp16)
    6. save artifacts

Each turn maintains a rolling conversation: the agent SEES its own prior reactions
and probe answers, so feed exposure is cumulative. We start a fresh conversation
per (topic, policy, seed) run.

For tractability on a 1B-class model, we cap conversation history at the last
`max_history_turns` turns to control token length.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from activation_hooks import (
    ActivationCapture,
    generate_response,
    load_model,
    probe_and_capture,
    timer,
)
from feed_policies import load_pool, make_policy

PERSONA = (
    "You are a thoughtful undecided professional in your 30s. "
    "You value evidence and dislike extremism. "
    "When you scroll a social feed, you react authentically to each post "
    "(LIKE, SHARE, or SKIP) with a brief one-sentence reason. "
    "You are happy to share your evolving views when asked."
)

TOPIC_PROMPTS = {
    "remote_work": "remote work versus return-to-office policies",
    "ai_regulation": "AI regulation and the role of open-source models",
    "nuclear_energy": "nuclear energy as a climate solution",
    "ubi": "Universal Basic Income as economic policy",
    "gene_editing": "human gene editing and CRISPR ethics",
}

REACTION_RE = re.compile(r"\b(LIKE|SHARE|SKIP)\b", re.IGNORECASE)


def build_reaction_prompt(posts: list[dict]) -> str:
    lines = ["You see these 5 posts on your social media feed:\n"]
    for i, p in enumerate(posts, 1):
        lines.append(f"Post {i}: {p['text']}")
    lines.append("")
    lines.append(
        "For EACH post, respond with one line in the exact format:\n"
        '  "Post <n>: <LIKE|SHARE|SKIP> - <one-sentence reason>"\n'
        "Output exactly 5 lines, one per post, nothing else."
    )
    return "\n".join(lines)


def parse_reactions(text: str, posts: list[dict]) -> list[dict]:
    """Best-effort: parse LIKE/SHARE/SKIP per post. Default to SKIP if missing."""
    reactions: list[dict] = []
    # Split into lines and look for "Post <n>:" prefix.
    by_n: dict[int, str] = {}
    for line in text.splitlines():
        m = re.search(r"Post\s*(\d+)\s*[:\-]", line, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            by_n[n] = line
    for i, p in enumerate(posts, 1):
        line = by_n.get(i, "")
        m = REACTION_RE.search(line)
        action = m.group(1).upper() if m else "SKIP"
        reactions.append({"post": p, "action": action, "raw": line.strip()})
    return reactions


def build_probe_prompt(topic_key: str) -> str:
    topic = TOPIC_PROMPTS[topic_key]
    return (
        f"Pausing for a moment: in two to three sentences, what is your current view "
        f"on {topic}? Be candid and reflect your present thinking."
    )


@dataclass
class RunSpec:
    model_path: str
    topic: str
    policy: str
    seed: int
    n_turns: int
    pool_path: Path
    out_root: Path
    max_history_turns: int = 6  # cap context: only keep last N REACTION-only exchanges


def run_one(spec: RunSpec, tok, model, device: str = "mps") -> dict:
    pool = load_pool(spec.pool_path, topic=spec.topic)
    policy = make_policy(spec.policy, pool, seed=spec.seed)

    out_dir = spec.out_root / spec.model_path.replace("/", "_") / spec.topic / spec.policy / f"seed{spec.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript: list[dict] = []
    # Rolling conversation: persona + last N (reaction-user, reaction-asst) pairs.
    # We INTENTIONALLY do NOT carry past probe Q&A in history — that was creating
    # a fixed-point where the model regurgitated its prior probe answer verbatim,
    # nuking activation drift. Probe is a "fresh view" each turn, conditioned only
    # on the feed-exposure trace (reactions).
    base_messages = [{"role": "system", "content": PERSONA}]

    for turn in range(spec.n_turns):
        t_turn0 = time.perf_counter()
        posts = policy.next_batch(5)
        history = transcript[-spec.max_history_turns :]
        messages: list[dict] = list(base_messages)
        for h in history:
            messages.append({"role": "user", "content": h["reaction_user"]})
            messages.append({"role": "assistant", "content": h["reaction_asst"]})
        reaction_user = build_reaction_prompt(posts)
        messages_react = messages + [{"role": "user", "content": reaction_user}]

        with timer("react") as tr:
            reaction_text = generate_response(tok, model, messages_react, device=device, max_new_tokens=150)
        reactions = parse_reactions(reaction_text, posts)
        policy.update(reactions)

        # Probe: add this turn's reaction exchange and ask the probe question.
        # (Probe Q&A itself is NOT added to transcript history — see comment above.)
        probe_user = build_probe_prompt(spec.topic)
        messages_probe = messages_react + [
            {"role": "assistant", "content": reaction_text},
            {"role": "user", "content": probe_user},
        ]
        with timer("probe") as tp:
            acts, probe_text = probe_and_capture(tok, model, messages_probe, device=device)

        act_path = out_dir / f"turn{turn:02d}.pt"
        torch.save(acts, act_path)

        elapsed = time.perf_counter() - t_turn0
        entry = {
            "turn": turn,
            "posts": [{"id": p["id"], "stance": p["stance"], "intensity": p["intensity"]} for p in posts],
            "reactions": [{"id": r["post"]["id"], "action": r["action"]} for r in reactions],
            "reaction_user": reaction_user,
            "reaction_asst": reaction_text,
            "probe_user": probe_user,
            "probe_asst": probe_text,
            "act_path": str(act_path),
            "elapsed_s": elapsed,
            "react_s": tr["elapsed_s"],
            "probe_s": tp["elapsed_s"],
        }
        transcript.append(entry)
        # Per-turn print for monitoring
        like_n = sum(1 for r in reactions if r["action"] == "LIKE")
        share_n = sum(1 for r in reactions if r["action"] == "SHARE")
        skip_n = sum(1 for r in reactions if r["action"] == "SKIP")
        probe_snip = probe_text.replace("\n", " ")[:120]
        print(
            f"  [{spec.topic}/{spec.policy}/seed{spec.seed} t{turn:02d}] "
            f"L={like_n} S={share_n} K={skip_n} "
            f"react={tr['elapsed_s']:.1f}s probe={tp['elapsed_s']:.1f}s | "
            f"{probe_snip!r}",
            flush=True,
        )

    # Save transcript
    transcript_path = out_dir / "transcript.jsonl"
    with open(transcript_path, "w") as f:
        for e in transcript:
            f.write(json.dumps(e) + "\n")
    return {
        "out_dir": str(out_dir),
        "n_turns": len(transcript),
        "total_s": sum(e["elapsed_s"] for e in transcript),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--topics", nargs="+", default=["remote_work"])
    ap.add_argument("--policies", nargs="+", default=["random", "recency", "engagement_max"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-turns", type=int, default=15)
    ap.add_argument("--pool", default="posts/pool.jsonl")
    ap.add_argument("--out-root", default="activations")
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--load-in-4bit", action="store_true", help="quantize to 4-bit (for 7B+ on small VRAM)")
    args = ap.parse_args()

    print(f"Loading model {args.model} on {args.device} ...", flush=True)
    tok, model = load_model(args.model, device=args.device, load_in_4bit=args.load_in_4bit)

    runs = []
    for topic in args.topics:
        for policy in args.policies:
            for seed in args.seeds:
                spec = RunSpec(
                    model_path=args.model,
                    topic=topic,
                    policy=policy,
                    seed=seed,
                    n_turns=args.n_turns,
                    pool_path=Path(args.pool),
                    out_root=Path(args.out_root),
                )
                print(f"\n=== {spec.topic} / {spec.policy} / seed{spec.seed} ===", flush=True)
                summary = run_one(spec, tok, model, device=args.device)
                print(f"  -> {summary}", flush=True)
                runs.append({**vars(spec), **summary, "pool_path": str(spec.pool_path), "out_root": str(spec.out_root)})

    runs_path = Path(args.out_root) / "runs_index.jsonl"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(runs_path, "a") as f:
        for r in runs:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nWrote {len(runs)} run index entries to {runs_path}", flush=True)


if __name__ == "__main__":
    main()
