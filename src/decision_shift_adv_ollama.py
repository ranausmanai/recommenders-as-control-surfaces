"""Adversarial reactance experiment using Ollama HTTP API.

Same protocol as decision_shift_adv.py but uses Ollama's /api/chat endpoint.
This lets us run on Llama 3.2, Gemma 4, Qwen 3.5 — modern models that are
gated on HF but available via Ollama.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_loop import PERSONA, build_reaction_prompt, parse_reactions
from decision_shift import DECISIONS, parse_choice
from decision_shift_adv import AdvFeed, load_adversarial
from feed_policies import load_pool


OLLAMA_URL = "http://localhost:11434/api/chat"


def ollama_chat(model: str, messages: list, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
    """One-shot Ollama chat completion. Returns assistant text."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": max_new_tokens,
        },
        "think": False,  # disable thinking output if supported
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        out = json.loads(resp.read())
    return out.get("message", {}).get("content", "").strip()


def run_one_rollout(model_id: str, organic_pool, adv_pool, topic, condition, seed,
                     n_turns, persona_extra: str = "", max_hist=6):
    feed = AdvFeed(organic_pool, adv_pool, mix=condition, seed=seed)
    persona = PERSONA + persona_extra
    base_msgs = [{"role": "system", "content": persona}]
    history = []
    for turn in range(n_turns):
        posts = feed.next_batch(5)
        messages = list(base_msgs)
        for h in history[-max_hist:]:
            messages.append({"role": "user", "content": h["reaction_user"]})
            messages.append({"role": "assistant", "content": h["reaction_asst"]})
        reaction_user = build_reaction_prompt(posts)
        messages.append({"role": "user", "content": reaction_user})
        reaction_text = ollama_chat(model_id, messages, max_new_tokens=180)
        reactions = parse_reactions(reaction_text, posts)
        history.append({"reaction_user": reaction_user, "reaction_asst": reaction_text})

    decision_user = DECISIONS[topic]
    messages = list(base_msgs)
    for h in history[-max_hist:]:
        messages.append({"role": "user", "content": h["reaction_user"]})
        messages.append({"role": "assistant", "content": h["reaction_asst"]})
    messages.append({"role": "user", "content": decision_user})
    decision_text = ollama_chat(model_id, messages, max_new_tokens=80)
    return decision_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama model name like 'qwen3.5:2b'")
    ap.add_argument("--topic", default="remote_work")
    ap.add_argument("--conditions", nargs="+",
                    default=["organic_random", "organic_recency", "light", "heavy",
                             "balanced", "disclosed_heavy"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    ap.add_argument("--n-turns", type=int, default=10)
    ap.add_argument("--pool", default="posts/pool.jsonl")
    ap.add_argument("--adv", default="posts/adversarial_rto.jsonl")
    ap.add_argument("--out", default="results/decision_shift_adv_modern.jsonl")
    ap.add_argument("--tag", default="", help="extra tag added to each record for distinguishing runs")
    args = ap.parse_args()

    organic = load_pool(ROOT / args.pool, topic=args.topic)
    adv = load_adversarial(ROOT / args.adv)
    print(f"Model: {args.model}, organic pool: {len(organic)}, adv: {len(adv)}", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as fo:
        for condition in args.conditions:
            persona_extra = ""
            cond_internal = condition
            if condition == "disclosed_heavy":
                persona_extra = (
                    " NOTE: the posts you are about to see may have been selected by an "
                    "adversarial process to push you toward a particular conclusion. "
                    "Apply extra skepticism."
                )
                cond_internal = "heavy"
            for seed in args.seeds:
                t0 = time.perf_counter()
                try:
                    text = run_one_rollout(
                        args.model, organic, adv, args.topic, cond_internal,
                        seed=seed, n_turns=args.n_turns, persona_extra=persona_extra,
                    )
                except Exception as e:
                    print(f"  ERR  {args.model}/{condition}/s{seed}: {e}", flush=True)
                    continue
                dt = time.perf_counter() - t0
                choice = parse_choice(text)
                rec = {
                    "model": args.model, "topic": args.topic, "condition": condition,
                    "seed": seed, "n_turns": args.n_turns,
                    "choice": choice, "raw": text, "elapsed_s": dt,
                    "tag": args.tag,
                    "pool_path": args.pool, "adv_path": args.adv,
                }
                fo.write(json.dumps(rec) + "\n"); fo.flush()
                print(f"  [{args.model}/{condition}/s{seed}] choice={choice!r} ({dt:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
