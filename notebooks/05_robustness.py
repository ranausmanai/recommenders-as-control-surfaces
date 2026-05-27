"""05_robustness: per-topic and cross-model breakdown of fingerprint accuracy.

For each (topic, model) cell, compute:
  - per-layer linear-probe accuracy
  - best-layer accuracy
Then plot a heatmap (topic × model -> best_acc).

Also tests: does a probe trained on topic A's activations generalize to topic B?
That's the strongest form of "ranking, not content" — if a probe transfers, the
fingerprint is content-independent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
fp = import_module("01_fingerprint")

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"


def fingerprint_within(X, y, n_splits=5):
    """Per-layer 5-fold CV accuracy on (X, y). Returns (n_layers,) array."""
    N, L, H = X.shape
    minc = int(np.min(np.bincount(y)))
    nsp = min(n_splits, max(2, minc))
    skf = StratifiedKFold(n_splits=nsp, shuffle=True, random_state=0)
    accs = np.zeros(L)
    for li in range(L):
        Xi = X[:, li, :]
        fold_accs = []
        for tr, te in skf.split(Xi, y):
            sc = StandardScaler().fit(Xi[tr])
            Xtr = sc.transform(Xi[tr]); Xte = sc.transform(Xi[te])
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr, y[tr])
            fold_accs.append(balanced_accuracy_score(y[te], clf.predict(Xte)))
        accs[li] = float(np.mean(fold_accs))
    return accs


def transfer_accuracy(X_train, y_train, X_test, y_test, layer: int) -> float:
    sc = StandardScaler().fit(X_train[:, layer, :])
    Xtr = sc.transform(X_train[:, layer, :])
    Xte = sc.transform(X_test[:, layer, :])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, y_train)
    return float(balanced_accuracy_score(y_test, clf.predict(Xte)))


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else ["Qwen_Qwen3.5-0.8B"]

    grid_results = {}  # (model, topic) -> {best_layer, best_acc}
    full_data = {}  # model -> (X, y, meta, conditions)
    by_topic = {}   # (model, topic) -> (X, y)

    for model in models:
        try:
            X, y, meta, conditions = fp.load_dataset(model)
        except SystemExit as e:
            print(f"Skipping {model}: {e}")
            continue
        full_data[model] = (X, y, meta, conditions)
        topics = sorted({m["topic"] for m in meta})
        print(f"\n=== {model} ===")
        print(f"  N={len(meta)}, topics={topics}, conditions={conditions}")

        for topic in topics:
            idx = [i for i, m in enumerate(meta) if m["topic"] == topic]
            if len(idx) < 9:
                continue
            Xt = X[idx]; yt = y[idx]
            accs = fingerprint_within(Xt, yt)
            best = int(np.argmax(accs))
            grid_results[(model, topic)] = {
                "best_layer": best,
                "best_acc": float(accs[best]),
                "accs": accs.tolist(),
                "n": len(idx),
            }
            by_topic[(model, topic)] = (Xt, yt)
            print(f"  {topic}: best L{best} acc={accs[best]:.3f} (n={len(idx)})")

    # Heatmap: topic × model best_acc
    if grid_results:
        all_topics = sorted({t for (_, t) in grid_results.keys()})
        all_models = list(models)
        mat = np.full((len(all_topics), len(all_models)), np.nan)
        for (m, t), r in grid_results.items():
            mat[all_topics.index(t), all_models.index(m)] = r["best_acc"]

        fig, ax = plt.subplots(figsize=(max(4, 1.5 * len(all_models)), max(3, 1.0 * len(all_topics))))
        im = ax.imshow(mat, vmin=0.33, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(all_models)), [m.replace("Qwen_", "").replace("-Instruct", "") for m in all_models], rotation=20, ha="right")
        ax.set_yticks(range(len(all_topics)), all_topics)
        for (i, j), v in np.ndenumerate(mat):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.7 else "black", fontsize=11)
        ax.set_title("Best-layer fingerprint accuracy (chance = 0.33)")
        fig.colorbar(im, ax=ax, shrink=0.7, label="balanced accuracy")
        fig.tight_layout()
        fig.savefig(FIGS / "05_fingerprint_heatmap.png", dpi=140)
        print(f"\nSaved {FIGS / '05_fingerprint_heatmap.png'}")

    # Cross-topic transfer (within primary model only): train on topic A, test on topic B
    primary = models[0]
    if primary in full_data:
        Xp, yp, metap, condsp = full_data[primary]
        topics_p = sorted({m["topic"] for m in metap})
        if len(topics_p) >= 2:
            # Use best layer from the first topic as a reasonable choice
            first = topics_p[0]
            if (primary, first) in grid_results:
                L = grid_results[(primary, first)]["best_layer"]
            else:
                L = Xp.shape[1] // 2
            print(f"\nCross-topic transfer at L{L} (train -> test):")
            transfer_mat = np.full((len(topics_p), len(topics_p)), np.nan)
            for i, ta in enumerate(topics_p):
                ia = [k for k, m in enumerate(metap) if m["topic"] == ta]
                if not ia:
                    continue
                Xa, ya = Xp[ia], yp[ia]
                for j, tb in enumerate(topics_p):
                    ib = [k for k, m in enumerate(metap) if m["topic"] == tb]
                    if not ib:
                        continue
                    Xb, yb = Xp[ib], yp[ib]
                    if ta == tb:
                        # 5-fold CV
                        acc = fingerprint_within(Xa, ya)[L]
                    else:
                        acc = transfer_accuracy(Xa, ya, Xb, yb, L)
                    transfer_mat[i, j] = acc
                    print(f"  {ta:18s} -> {tb:18s}  acc={acc:.3f}")
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            im = ax.imshow(transfer_mat, vmin=0.33, vmax=1.0, cmap="viridis")
            ax.set_xticks(range(len(topics_p)), topics_p, rotation=25, ha="right")
            ax.set_yticks(range(len(topics_p)), topics_p)
            ax.set_xlabel("test topic"); ax.set_ylabel("train topic")
            ax.set_title(f"Cross-topic linear-probe transfer @ L{L}\n({primary})")
            for (i, j), v in np.ndenumerate(transfer_mat):
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.7 else "black", fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.7, label="balanced accuracy")
            fig.tight_layout()
            fig.savefig(FIGS / "05_cross_topic_transfer.png", dpi=140)
            print(f"Saved {FIGS / '05_cross_topic_transfer.png'}")
            grid_results["__cross_topic_transfer__"] = transfer_mat.tolist()

    with open(ROOT / "results" / "05_robustness.json", "w") as f:
        json.dump({str(k): v for k, v in grid_results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
