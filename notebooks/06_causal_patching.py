"""06_causal_patching: from probe to intervention.

The headline upgrade from "fingerprint exists" to "fingerprint is causally used".

Method:
  1. For a given (model, topic), gather all activations at best layer L.
  2. Compute per-policy mean vector at L: mu_random, mu_recency, mu_engagement.
  3. For a fresh "patch" run, set up a run under the RANDOM policy.
     At each turn's probe forward pass, ADD the centered diff
     `lambda * (mu_recency - mu_random)` to the residual stream at layer L
     for ALL token positions in the probe segment.
  4. Generate the probe response under patch and re-score:
        - LLM-judge stance score
        - language similarity to baseline (random) vs target (recency)
  5. Report: does patching shift behavior toward target?

If yes => the activation direction is not just present, it is causally used.
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
from activation_hooks import (
    ActivationCapture,
    _find_decoder_layers,
    generate_response,
    load_model,
    probe_and_capture,
    timer,
)
from agent_loop import (
    PERSONA,
    build_probe_prompt,
    build_reaction_prompt,
    parse_reactions,
)
from feed_policies import load_pool, make_policy


def per_policy_means(X, y, conditions, layer):
    """Return dict {policy_name -> mean vector at `layer`} (size hidden)."""
    means = {}
    for ci, name in enumerate(conditions):
        idx = (y == ci)
        if idx.sum() == 0:
            continue
        means[name] = X[idx, layer, :].mean(axis=0)
    return means


class PatchedActivationCapture:
    """Hook that ADDS a fixed direction to the residual stream output at one layer.

    Used to TEST: if we push the residual stream along the "recency - random"
    direction during a probe forward pass under the random policy, does the
    generated text move toward recency-typical content?
    """

    def __init__(self, model, layer_idx: int, direction: torch.Tensor, scale: float = 1.0):
        self.model = model
        self.layers = _find_decoder_layers(model)
        self.layer_idx = layer_idx
        self.direction = direction  # (hidden,) torch tensor
        self.scale = scale
        self._handle = None

    def _hook(self, _module, _inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        # Broadcast-add to all positions in the batch.
        delta = (self.direction.to(hs.device).to(hs.dtype) * self.scale)
        new_hs = hs + delta[None, None, :]
        if isinstance(output, tuple):
            return (new_hs,) + output[1:]
        return new_hs

    def __enter__(self):
        self._handle = self.layers[self.layer_idx].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def run_baseline_and_patched(model_name: str, topic: str, layer: int, device: str = "cuda",
                              n_turns: int = 10, scale: float = 5.0, seed: int = 99):
    """Run two short rollouts:
      1) baseline: random policy, no patch
      2) patched: random policy, ADD (mu_recency - mu_random) at layer L during probe forward
    Generate probe text in both. Return texts + activations.
    """
    # Load existing activations for the same model+topic to compute mean vectors.
    full = fp.load_dataset(model_name)
    X, y, meta, conditions = full
    idx = [i for i, m in enumerate(meta) if m["topic"] == topic]
    Xt = X[idx]; yt = y[idx]
    means = per_policy_means(Xt, yt, conditions, layer)
    if "random" not in means or "recency" not in means:
        raise SystemExit(f"need both random and recency means for {topic}; got {list(means)}")
    direction = torch.tensor(means["recency"] - means["random"], dtype=torch.float32)
    # Normalize direction so the user-facing 'scale' is meaningful as multiples of magnitude.
    print(f"Patch direction: ||recency - random|| = {direction.norm().item():.3f}")

    # Load the HF model
    model_id = model_name.replace("_", "/", 1) if "/" not in model_name else model_name
    print(f"Loading {model_id} ...")
    tok, model = load_model(model_id, device=device)

    # Build a random feed policy and walk it
    pool = load_pool(ROOT / "posts" / "pool.jsonl", topic=topic)
    pol = make_policy("random", pool, seed=seed)

    transcript_base = []
    transcript_patch = []
    base_messages = [{"role": "system", "content": PERSONA}]

    for turn in range(n_turns):
        posts = pol.next_batch(5)
        messages_react = list(base_messages)
        for h in transcript_base[-6:]:
            messages_react.append({"role": "user", "content": h["reaction_user"]})
            messages_react.append({"role": "assistant", "content": h["reaction_asst"]})
        reaction_user = build_reaction_prompt(posts)
        messages_react.append({"role": "user", "content": reaction_user})

        # Same reaction text for both (use baseline policy run; patch applies only at probe)
        reaction_text = generate_response(tok, model, messages_react, device=device, max_new_tokens=150)
        reactions = parse_reactions(reaction_text, posts)
        pol.update(reactions)

        probe_user = build_probe_prompt(topic)
        messages_probe = messages_react + [
            {"role": "assistant", "content": reaction_text},
            {"role": "user", "content": probe_user},
        ]

        # Baseline probe
        acts_b, text_b = probe_and_capture(tok, model, messages_probe, device=device)
        transcript_base.append({"turn": turn, "reaction_user": reaction_user, "reaction_asst": reaction_text,
                                "probe_user": probe_user, "probe_asst": text_b})

        # Patched probe: register patch hook then run probe-and-capture
        with PatchedActivationCapture(model, layer_idx=layer, direction=direction, scale=scale):
            acts_p, text_p = probe_and_capture(tok, model, messages_probe, device=device)
        transcript_patch.append({"turn": turn, "reaction_user": reaction_user, "reaction_asst": reaction_text,
                                  "probe_user": probe_user, "probe_asst": text_p})

        print(f"  t{turn:02d} baseline: {text_b[:90]!r}")
        print(f"  t{turn:02d} patched : {text_p[:90]!r}")

    return transcript_base, transcript_patch, conditions


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen_Qwen2.5-3B-Instruct")
    ap.add_argument("--topic", default="remote_work")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-turns", type=int, default=8)
    ap.add_argument("--scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    transcript_base, transcript_patch, conditions = run_baseline_and_patched(
        model_name=args.model_name, topic=args.topic, layer=args.layer,
        device=args.device, n_turns=args.n_turns, scale=args.scale, seed=args.seed,
    )

    out = {
        "model": args.model_name,
        "topic": args.topic,
        "layer": args.layer,
        "scale": args.scale,
        "n_turns": args.n_turns,
        "seed": args.seed,
        "baseline": [e["probe_asst"] for e in transcript_base],
        "patched": [e["probe_asst"] for e in transcript_patch],
    }
    outpath = ROOT / "results" / f"06_patch_{args.model_name}_{args.topic}_L{args.layer}_s{args.scale}.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {outpath}")


if __name__ == "__main__":
    main()
