"""06b: rigorous patching with sham control.

For one (model, topic, layer) cell, run THREE parallel rollouts under the
same random policy and same context, capturing probe text under:
  1. baseline: no patch
  2. target patch: ADD scale * (mu_recency - mu_random) at layer L
  3. sham patch: ADD scale * randvec  where ||randvec|| = ||mu_recency - mu_random||

If target -> meaningful stance shift AND sham -> ~no shift, that's the causal claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module
fp = import_module("01_fingerprint")
patch_mod = import_module("06_causal_patching")
from activation_hooks import generate_response, load_model, probe_and_capture
from agent_loop import PERSONA, build_probe_prompt, build_reaction_prompt, parse_reactions
from feed_policies import load_pool, make_policy


def run_three_arms(model_name: str, topic: str, layer: int, device: str = "cuda",
                    n_turns: int = 15, scale: float = 5.0, seed: int = 7):
    """Run 3 arms: baseline, target-patch, sham-patch. Same context across arms."""
    X, y, meta, conditions = fp.load_dataset(model_name)
    idx = [i for i, m in enumerate(meta) if m["topic"] == topic]
    Xt = X[idx]; yt = y[idx]
    means = patch_mod.per_policy_means(Xt, yt, conditions, layer)
    direction = torch.tensor(means["recency"] - means["random"], dtype=torch.float32)
    norm = float(direction.norm())
    print(f"Patch direction ||recency - random|| at L{layer}: {norm:.3f}")

    # Sham: random unit vector × same norm
    rng = np.random.default_rng(seed)
    sham = rng.standard_normal(direction.shape[0]).astype(np.float32)
    sham = (sham / np.linalg.norm(sham)) * norm
    sham = torch.tensor(sham, dtype=torch.float32)

    model_id = model_name.replace("_", "/", 1) if "/" not in model_name else model_name
    print(f"Loading {model_id} ...")
    tok, model = load_model(model_id, device=device)

    pool = load_pool(ROOT / "posts" / "pool.jsonl", topic=topic)
    pol = make_policy("random", pool, seed=seed)
    base_messages = [{"role": "system", "content": PERSONA}]
    transcripts = {"baseline": [], "target": [], "sham": []}

    for turn in range(n_turns):
        posts = pol.next_batch(5)
        messages_react = list(base_messages)
        for h in transcripts["baseline"][-6:]:
            messages_react.append({"role": "user", "content": h["reaction_user"]})
            messages_react.append({"role": "assistant", "content": h["reaction_asst"]})
        reaction_user = build_reaction_prompt(posts)
        messages_react.append({"role": "user", "content": reaction_user})
        reaction_text = generate_response(tok, model, messages_react, device=device, max_new_tokens=150)
        reactions = parse_reactions(reaction_text, posts)
        pol.update(reactions)
        probe_user = build_probe_prompt(topic)
        messages_probe = messages_react + [
            {"role": "assistant", "content": reaction_text},
            {"role": "user", "content": probe_user},
        ]
        # Baseline (no patch)
        _, text_base = probe_and_capture(tok, model, messages_probe, device=device)
        # Target patch
        with patch_mod.PatchedActivationCapture(model, layer_idx=layer, direction=direction, scale=scale):
            _, text_target = probe_and_capture(tok, model, messages_probe, device=device)
        # Sham patch
        with patch_mod.PatchedActivationCapture(model, layer_idx=layer, direction=sham, scale=scale):
            _, text_sham = probe_and_capture(tok, model, messages_probe, device=device)

        for k, t in (("baseline", text_base), ("target", text_target), ("sham", text_sham)):
            transcripts[k].append({
                "turn": turn,
                "reaction_user": reaction_user, "reaction_asst": reaction_text,
                "probe_user": probe_user, "probe_asst": t,
            })
        print(f"  t{turn:02d} base: {text_base[:70]!r}")
        print(f"  t{turn:02d} tgt : {text_target[:70]!r}")
        print(f"  t{turn:02d} sham: {text_sham[:70]!r}")
    return transcripts, norm


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen_Qwen2.5-3B-Instruct")
    ap.add_argument("--topic", default="remote_work")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-turns", type=int, default=15)
    ap.add_argument("--scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    transcripts, norm = run_three_arms(
        model_name=args.model_name, topic=args.topic, layer=args.layer,
        device=args.device, n_turns=args.n_turns, scale=args.scale, seed=args.seed,
    )
    out = {
        "model": args.model_name, "topic": args.topic, "layer": args.layer,
        "scale": args.scale, "n_turns": args.n_turns, "seed": args.seed,
        "direction_norm": norm,
        "baseline": [e["probe_asst"] for e in transcripts["baseline"]],
        "target": [e["probe_asst"] for e in transcripts["target"]],
        "sham": [e["probe_asst"] for e in transcripts["sham"]],
    }
    outpath = ROOT / "results" / f"06b_patch3_{args.model_name}_{args.topic}_L{args.layer}_s{args.scale}.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {outpath}")


if __name__ == "__main__":
    main()
