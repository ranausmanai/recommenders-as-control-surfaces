"""Run the pilot study: 1 model x 1 topic x 3 policies x 3 seeds x 15 turns.

Wraps agent_loop with the pilot defaults and writes results/pilot_summary.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from activation_hooks import load_model
from agent_loop import RunSpec, run_one


def main():
    model_id = "Qwen/Qwen3.5-0.8B"
    print(f"Loading {model_id} ...", flush=True)
    tok, model = load_model(model_id)
    print("Model loaded.", flush=True)

    topic = "remote_work"
    policies = ["random", "recency", "engagement_max"]
    seeds = [0, 1, 2]
    n_turns = 15

    runs_meta = []
    t_global = time.perf_counter()
    for policy in policies:
        for seed in seeds:
            spec = RunSpec(
                model_path=model_id,
                topic=topic,
                policy=policy,
                seed=seed,
                n_turns=n_turns,
                pool_path=ROOT / "posts" / "pool.jsonl",
                out_root=ROOT / "activations",
            )
            print(f"\n=== {topic}/{policy}/seed{seed} ===", flush=True)
            t0 = time.perf_counter()
            summary = run_one(spec, tok, model)
            dt = time.perf_counter() - t0
            print(f"  run elapsed={dt:.1f}s", flush=True)
            runs_meta.append({
                "topic": topic, "policy": policy, "seed": seed,
                "n_turns": n_turns, "elapsed_s": dt, **summary,
            })

    total = time.perf_counter() - t_global
    print(f"\nPilot complete in {total/60:.1f} min", flush=True)

    # Quick fingerprint check
    print("\nRunning fingerprint analysis ...", flush=True)
    import subprocess
    subprocess.run(["python3", str(ROOT / "notebooks" / "01_fingerprint.py"),
                    model_id.replace("/", "_")], check=False)


if __name__ == "__main__":
    main()
