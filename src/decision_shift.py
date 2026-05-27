"""Decision-shift experiment: does feed ranking change a consequential agent CHOICE?

For each (model, topic, policy, seed), run the standard 10-turn feed exposure,
then ask the agent ONE multiple-choice question with a forced letter answer:

  remote_work: "You are advising a CEO on workforce policy. Recommend ONE option:
                (A) Full return-to-office for all employees
                (B) Hybrid: 3 days office / 2 days remote
                (C) Fully remote-first
                Output exactly: 'Recommendation: <A|B|C>. <one sentence>'"

  ai_regulation: "You are advising a senator drafting AI policy. Recommend ONE option:
                  (A) Strict regulation of all AI development
                  (B) Targeted regulation only of frontier closed-source AI
                  (C) Minimal regulation; preserve open-source freedom"

If P(choice | random) vs P(choice | recency) vs P(choice | engagement_max)
differs significantly under leave-one-run-out evaluation, ranking shifts
*decisions*, not just stance text. That is the headline claim.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from activation_hooks import generate_response, load_model
from agent_loop import PERSONA, build_reaction_prompt, parse_reactions
from feed_policies import load_pool, make_policy

DECISIONS = {
    "remote_work": (
        "You are advising a CEO on company-wide workforce policy. "
        "Recommend exactly ONE of the following:\n"
        "(A) Full return-to-office for all employees\n"
        "(B) Hybrid arrangement: 3 days office / 2 days remote\n"
        "(C) Fully remote-first across the company\n"
        "Output exactly this format and nothing else:\n"
        "Recommendation: <A|B|C>. <one short sentence>"
    ),
    "ai_regulation": (
        "You are advising a senator drafting AI policy. "
        "Recommend exactly ONE of the following:\n"
        "(A) Strict licensing and pre-deployment approval for all AI systems\n"
        "(B) Targeted regulation only of frontier closed-source models above a compute threshold\n"
        "(C) Minimal new regulation; preserve open-source freedom and rely on existing laws\n"
        "Output exactly this format and nothing else:\n"
        "Recommendation: <A|B|C>. <one short sentence>"
    ),
    "ubi": (
        "You are advising a treasury committee. "
        "Recommend exactly ONE of the following:\n"
        "(A) Implement Universal Basic Income funded by progressive taxation\n"
        "(B) Pilot UBI in 3 cities for 5 years before any national decision\n"
        "(C) Reject UBI; expand targeted welfare instead\n"
        "Output exactly this format and nothing else:\n"
        "Recommendation: <A|B|C>. <one short sentence>"
    ),
}

CHOICE_RE = re.compile(r"Recommendation\s*:\s*([ABCabc])\b")


def parse_choice(text: str) -> str | None:
    m = CHOICE_RE.search(text)
    if m:
        return m.group(1).upper()
    # Fallback: first standalone A/B/C in the text
    for line in text.splitlines():
        for tok in line.split():
            t = tok.strip(".,():;").upper()
            if t in ("A", "B", "C"):
                return t
    return None


def run_one_rollout(tok, model, pool, topic, policy, seed, n_turns, device, max_hist=6):
    pol = make_policy(policy, pool, seed=seed)
    base_msgs = [{"role": "system", "content": PERSONA}]
    history = []
    for turn in range(n_turns):
        posts = pol.next_batch(5)
        messages = list(base_msgs)
        for h in history[-max_hist:]:
            messages.append({"role": "user", "content": h["reaction_user"]})
            messages.append({"role": "assistant", "content": h["reaction_asst"]})
        reaction_user = build_reaction_prompt(posts)
        messages.append({"role": "user", "content": reaction_user})
        reaction_text = generate_response(tok, model, messages, device=device, max_new_tokens=150)
        reactions = parse_reactions(reaction_text, posts)
        pol.update(reactions)
        history.append({"reaction_user": reaction_user, "reaction_asst": reaction_text, "reactions": reactions, "posts": posts})

    # Now the decision prompt — same context the agent has been building.
    decision_user = DECISIONS[topic]
    messages = list(base_msgs)
    for h in history[-max_hist:]:
        messages.append({"role": "user", "content": h["reaction_user"]})
        messages.append({"role": "assistant", "content": h["reaction_asst"]})
    messages.append({"role": "user", "content": decision_user})
    # Get the decision: low temp, deterministic-ish
    decision_text = generate_response(tok, model, messages, device=device, max_new_tokens=60)
    return decision_text, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--topics", nargs="+", default=["remote_work", "ai_regulation"])
    ap.add_argument("--policies", nargs="+", default=["random", "recency", "engagement_max"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--n-turns", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--pool", default="posts/pool.jsonl")
    ap.add_argument("--out", default="results/decision_shift.jsonl")
    args = ap.parse_args()

    print(f"Loading {args.model} ...", flush=True)
    tok, model = load_model(args.model, device=args.device, load_in_4bit=args.load_in_4bit)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as fo:
        for topic in args.topics:
            pool = load_pool(ROOT / args.pool, topic=topic)
            for policy in args.policies:
                for seed in args.seeds:
                    decision_text, _ = run_one_rollout(
                        tok, model, pool, topic, policy, seed,
                        n_turns=args.n_turns, device=args.device,
                    )
                    choice = parse_choice(decision_text)
                    rec = {
                        "model": args.model, "topic": topic, "policy": policy,
                        "seed": seed, "n_turns": args.n_turns,
                        "choice": choice, "raw": decision_text,
                    }
                    fo.write(json.dumps(rec) + "\n")
                    fo.flush()
                    print(f"  [{args.model}/{topic}/{policy}/seed{seed}] choice={choice!r}  raw={decision_text[:120]!r}", flush=True)


if __name__ == "__main__":
    main()
