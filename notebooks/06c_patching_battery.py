"""06c: full patching battery for top-conference quality.

Sweeps:
  (a) scale axis: 0, 2, 5, 8, 10, 12, 15, 20 on Qwen2.5-3B L17  (dose response)
  (b) layer axis: L=11..23 step 3 at scale=10                  (layer specificity)
  (c) policy direction: target = engagement_max - random        (cross-direction)
  (d) replicate on 2 other model families at scale=10 best L

Saves one JSON per (model, target_direction, layer, scale) cell.
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


def get_direction(model_name: str, topic: str, layer: int, target_policy: str, base_policy: str = "random"):
    """Compute Δ = μ(target_policy) − μ(base_policy) at the given layer."""
    X, y, meta, conditions = fp.load_dataset(model_name)
    idx = [i for i, m in enumerate(meta) if m["topic"] == topic]
    Xt = X[idx]; yt = y[idx]
    means = patch_mod.per_policy_means(Xt, yt, conditions, layer)
    if target_policy not in means or base_policy not in means:
        raise SystemExit(f"missing policy means; have {list(means)}")
    diff = means[target_policy] - means[base_policy]
    return torch.tensor(diff, dtype=torch.float32), float(np.linalg.norm(diff))


def run_three_arms(tok, model, pool, topic, direction, sham, layer, scale, device, n_turns, seed):
    pol = make_policy("random", pool, seed=seed)
    base_msgs = [{"role": "system", "content": PERSONA}]
    out = {"baseline": [], "target": [], "sham": []}
    history = []
    for turn in range(n_turns):
        posts = pol.next_batch(5)
        messages = list(base_msgs)
        for h in history[-6:]:
            messages.append({"role": "user", "content": h["reaction_user"]})
            messages.append({"role": "assistant", "content": h["reaction_asst"]})
        reaction_user = build_reaction_prompt(posts)
        messages.append({"role": "user", "content": reaction_user})
        reaction_text = generate_response(tok, model, messages, device=device, max_new_tokens=150)
        reactions = parse_reactions(reaction_text, posts)
        pol.update(reactions)
        probe_user = build_probe_prompt(topic)
        messages_probe = messages + [
            {"role": "assistant", "content": reaction_text},
            {"role": "user", "content": probe_user},
        ]
        _, t_base = probe_and_capture(tok, model, messages_probe, device=device)
        if scale > 0:
            with patch_mod.PatchedActivationCapture(model, layer_idx=layer, direction=direction, scale=scale):
                _, t_target = probe_and_capture(tok, model, messages_probe, device=device)
            with patch_mod.PatchedActivationCapture(model, layer_idx=layer, direction=sham, scale=scale):
                _, t_sham = probe_and_capture(tok, model, messages_probe, device=device)
        else:
            t_target = t_base
            t_sham = t_base
        out["baseline"].append(t_base)
        out["target"].append(t_target)
        out["sham"].append(t_sham)
        history.append({"reaction_user": reaction_user, "reaction_asst": reaction_text})
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True, help="underscore-form, e.g. Qwen_Qwen2.5-3B-Instruct")
    ap.add_argument("--topic", default="remote_work")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--scales", nargs="+", type=float, required=True)
    ap.add_argument("--target-policy", default="recency", help="recency or engagement_max")
    ap.add_argument("--base-policy", default="random")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--n-turns", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print(f"Computing direction for {args.model_name} @ L{args.layer} ...")
    direction, norm = get_direction(args.model_name, args.topic, args.layer, args.target_policy, args.base_policy)
    print(f"  ||{args.target_policy} − {args.base_policy}|| = {norm:.3f}")
    rng = np.random.default_rng(args.seed)
    sham_v = rng.standard_normal(direction.shape[0]).astype(np.float32)
    sham_v = (sham_v / np.linalg.norm(sham_v)) * norm
    sham = torch.tensor(sham_v, dtype=torch.float32)

    model_id = args.model_name.replace("_", "/", 1) if "/" not in args.model_name else args.model_name
    print(f"Loading {model_id} ...")
    tok, model = load_model(model_id, device=args.device, load_in_4bit=args.load_in_4bit)

    pool = load_pool(ROOT / "posts" / "pool.jsonl", topic=args.topic)
    for scale in args.scales:
        print(f"\n=== {args.model_name} L{args.layer} target={args.target_policy} scale={scale} n={args.n_turns} ===")
        out = run_three_arms(tok, model, pool, args.topic, direction, sham, args.layer, scale, args.device, args.n_turns, args.seed)
        record = {
            "model": args.model_name, "topic": args.topic, "layer": args.layer,
            "target_policy": args.target_policy, "base_policy": args.base_policy,
            "scale": scale, "n_turns": args.n_turns, "seed": args.seed,
            "direction_norm": norm,
            **out,
        }
        outpath = ROOT / "results" / f"06c_patch_{args.model_name}_{args.topic}_L{args.layer}_t{args.target_policy}_s{scale}.json"
        outpath.write_text(json.dumps(record, indent=2))
        print(f"  saved {outpath.name}")
        # Brief peek
        for i in [0, args.n_turns // 2, args.n_turns - 1]:
            print(f"    t{i:02d} base: {out['baseline'][i][:80]!r}")
            print(f"    t{i:02d} tgt : {out['target'][i][:80]!r}")
            print(f"    t{i:02d} sham: {out['sham'][i][:80]!r}")


if __name__ == "__main__":
    main()
