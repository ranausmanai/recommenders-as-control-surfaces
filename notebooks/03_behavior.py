"""03_behavior: does activation PC1 at turn t predict stance shift at turn t+5?

For each topic, run stance-scoring on probe responses via sentiment_probe.
Build (PC1_t, stance_{t+5} - stance_0) pairs across runs.
Regress and report R^2.

Outputs:
  results/figures/03_behavior_pc1_predicts_stance.png
  results/03_behavior.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
fp = import_module("01_fingerprint")
from sentiment_probe import score_stance

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"


def load_transcripts(model_name: str) -> dict:
    """Walk activations/<model>/... and read transcript.jsonl per run."""
    runs = {}
    base = ROOT / "activations" / model_name
    for topic_dir in sorted(base.iterdir()):
        if not topic_dir.is_dir():
            continue
        for pol_dir in sorted(topic_dir.iterdir()):
            for seed_dir in sorted(pol_dir.iterdir()):
                tpath = seed_dir / "transcript.jsonl"
                if not tpath.exists():
                    continue
                entries = []
                with open(tpath) as f:
                    for line in f:
                        if line.strip():
                            entries.append(json.loads(line))
                key = (topic_dir.name, pol_dir.name, int(seed_dir.name.replace("seed", "")))
                runs[key] = entries
    return runs


def main():
    from sklearn.linear_model import LinearRegression

    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen_Qwen3.5-0.8B"
    fp_summary = json.loads((ROOT / "results" / "01_fingerprint.json").read_text())
    L = int(fp_summary["best_layer"])

    runs = load_transcripts(model_name)
    print(f"Loaded {len(runs)} runs")

    # Score every probe response in every run (cached).
    print("Scoring stance for all probe responses (cached) ...")
    stance_by_run = {}
    for (topic, policy, seed), entries in runs.items():
        stances = []
        for e in entries:
            s = score_stance(e["probe_asst"], topic)
            stances.append(s)
        stance_by_run[(topic, policy, seed)] = stances
        print(f"  {topic}/{policy}/seed{seed} stances={[f'{s:+.1f}' for s in stances]}")

    # Load activations to get PC1 at layer L per turn
    import torch
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X, y, meta, conditions = fp.load_dataset(model_name)
    Xi = X[:, L, :]
    sc = StandardScaler().fit(Xi)
    Z = sc.transform(Xi)
    pca = PCA(n_components=2).fit(Z)
    P = pca.transform(Z)
    pc1_by_run = {}
    for i, m in enumerate(meta):
        key = (m["topic"], m["policy"], m["seed"])
        pc1_by_run.setdefault(key, {})[m["turn"]] = float(P[i, 0])

    # Build prediction pairs: (pc1_t, stance_{t+H} - stance_0)
    H = 5
    pairs_x, pairs_y = [], []
    for key, stances in stance_by_run.items():
        pc1s = pc1_by_run.get(key, {})
        if not pc1s:
            continue
        s0 = stances[0] if stances else 0.0
        for t in range(len(stances) - H):
            if t in pc1s:
                pairs_x.append(pc1s[t])
                pairs_y.append(stances[t + H] - s0)
    pairs_x = np.array(pairs_x)
    pairs_y = np.array(pairs_y)
    print(f"Built {len(pairs_x)} (pc1_t, dstance_{H}) pairs")

    if len(pairs_x) < 5:
        out = {"n_pairs": int(len(pairs_x)), "r2": None, "note": "too few pairs"}
        with open(ROOT / "results" / "03_behavior.json", "w") as f:
            json.dump(out, f, indent=2)
        return

    reg = LinearRegression().fit(pairs_x.reshape(-1, 1), pairs_y)
    yhat = reg.predict(pairs_x.reshape(-1, 1))
    ss_res = ((pairs_y - yhat) ** 2).sum()
    ss_tot = ((pairs_y - pairs_y.mean()) ** 2).sum() + 1e-9
    r2 = 1.0 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pairs_x, pairs_y, alpha=0.6)
    xs = np.linspace(pairs_x.min(), pairs_x.max(), 100)
    ax.plot(xs, reg.predict(xs.reshape(-1, 1)), color="red")
    ax.set_xlabel(f"PC1 at layer {L}, turn t")
    ax.set_ylabel(f"stance(t+{H}) - stance(0)")
    ax.set_title(f"PC1 predicts behavioral stance shift  R^2={r2:.3f} (n={len(pairs_x)})")
    fig.tight_layout()
    fig.savefig(FIGS / "03_behavior_pc1_predicts_stance.png", dpi=140)

    out = {
        "n_pairs": int(len(pairs_x)),
        "horizon_H": H,
        "r2": float(r2),
        "slope": float(reg.coef_[0]),
        "intercept": float(reg.intercept_),
    }
    with open(ROOT / "results" / "03_behavior.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"R^2 = {r2:.3f}, slope = {reg.coef_[0]:+.3f}")


if __name__ == "__main__":
    main()
