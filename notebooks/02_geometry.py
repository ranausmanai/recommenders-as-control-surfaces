"""02_geometry: PCA of activation trajectories at best layer.

For the best layer chosen by 01_fingerprint, run PCA across all samples and
plot trajectories colored by (condition, turn).

Outputs:
  results/figures/02_geometry_pc1_pc2.png
  results/figures/02_geometry_pc1_by_turn.png
  results/02_geometry.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Reuse loader from sibling
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
fp = import_module("01_fingerprint")

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen_Qwen3.5-0.8B"
    X, y, meta, conditions = fp.load_dataset(model_name)
    summary = json.loads((ROOT / "results" / "01_fingerprint.json").read_text())
    L = int(summary["best_layer"])
    print(f"Using best layer L={L}")

    Xi = X[:, L, :]
    sc = StandardScaler().fit(Xi)
    Z = sc.transform(Xi)
    pca = PCA(n_components=5).fit(Z)
    P = pca.transform(Z)
    var = pca.explained_variance_ratio_
    print(f"Explained variance (first 5 PCs): {var}")

    # Plot PC1 vs PC2, color by condition, size by turn (so trajectories visible)
    turns = np.array([m["turn"] for m in meta])
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for ci, cond in enumerate(conditions):
        mask = y == ci
        ax.scatter(P[mask, 0], P[mask, 1], c=[cmap(ci)], s=20 + turns[mask] * 4,
                   alpha=0.65, label=cond, edgecolors="none")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_title(f"layer {L} activations: PC1/PC2 by condition (size = turn)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "02_geometry_pc1_pc2.png", dpi=140)
    print(f"Saved {FIGS / '02_geometry_pc1_pc2.png'}")

    # Plot PC1 trajectory by turn, one line per (condition, seed)
    fig, ax = plt.subplots(figsize=(8, 4))
    seeds = sorted({m["seed"] for m in meta})
    for ci, cond in enumerate(conditions):
        for s in seeds:
            sel = [i for i, m in enumerate(meta) if m["policy"] == cond and m["seed"] == s]
            if not sel:
                continue
            sel = sorted(sel, key=lambda i: meta[i]["turn"])
            tt = [meta[i]["turn"] for i in sel]
            pp = [P[i, 0] for i in sel]
            ax.plot(tt, pp, color=cmap(ci), alpha=0.6, marker="o", markersize=3,
                    label=f"{cond}" if s == seeds[0] else None)
    ax.set_xlabel("turn")
    ax.set_ylabel("PC1")
    ax.set_title(f"layer {L}: PC1 drift over turns (per seed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "02_geometry_pc1_by_turn.png", dpi=140)
    print(f"Saved {FIGS / '02_geometry_pc1_by_turn.png'}")

    out = {
        "best_layer": L,
        "explained_variance_top5": var.tolist(),
        "linear_drift_dominant": bool(var[0] > 0.5),
    }
    with open(ROOT / "results" / "02_geometry.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
