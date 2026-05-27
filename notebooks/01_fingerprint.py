"""01_fingerprint: can a linear probe at layer L recover the FEED CONDITION?

For each layer L (0..n_layers-1):
  - X = activation vector at layer L, one per (run, turn)
  - y = condition (random / recency / engagement_max)
  - 5-fold stratified CV logistic regression
  - record balanced accuracy

Above-chance (>0.333) at any layer => the feed algorithm leaves a fingerprint.

Outputs:
  results/figures/01_fingerprint_accuracy_vs_layer.png
  results/figures/01_fingerprint_confusion.png
  results/01_fingerprint.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import os
ROOT = Path(__file__).resolve().parent.parent
ACTS = ROOT / os.environ.get("ACT_ROOT", "activations")
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def load_dataset(model_name: str, topics: list[str] | None = None):
    """Walk <ACT_ROOT>/<model>/<topic>/<policy>/seed<k>/turn<NN>.pt and stack.

    Returns:
        X: (N, n_layers, hidden) np.float32
        y: (N,) condition labels (int)
        meta: list of dicts with topic/policy/seed/turn
        conditions: ordered list of policy names (for label mapping)
    """
    base = ACTS / model_name
    if not base.exists():
        raise SystemExit(f"No activations under {base}")

    samples = []
    for topic_dir in sorted(base.iterdir()):
        if not topic_dir.is_dir():
            continue
        if topics and topic_dir.name not in topics:
            continue
        for policy_dir in sorted(topic_dir.iterdir()):
            if not policy_dir.is_dir():
                continue
            for seed_dir in sorted(policy_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                seed = int(seed_dir.name.replace("seed", ""))
                for pt in sorted(seed_dir.glob("turn*.pt")):
                    turn = int(pt.stem.replace("turn", ""))
                    samples.append({
                        "topic": topic_dir.name,
                        "policy": policy_dir.name,
                        "seed": seed,
                        "turn": turn,
                        "path": pt,
                    })

    if not samples:
        raise SystemExit(f"No activation .pt files under {base}")

    conditions = sorted({s["policy"] for s in samples})
    cond_to_y = {c: i for i, c in enumerate(conditions)}

    acts_list = []
    y_list = []
    meta = []
    for s in samples:
        t = torch.load(s["path"], map_location="cpu", weights_only=True)
        acts_list.append(t.numpy().astype(np.float32))
        y_list.append(cond_to_y[s["policy"]])
        meta.append(s)

    X = np.stack(acts_list, axis=0)  # (N, L, H)
    y = np.array(y_list, dtype=np.int64)
    return X, y, meta, conditions


def per_layer_accuracy(X, y, n_splits=5):
    N, L, H = X.shape
    skf = StratifiedKFold(n_splits=min(n_splits, np.min(np.bincount(y))), shuffle=True, random_state=0)
    accs = np.zeros(L)
    for li in range(L):
        Xi = X[:, li, :]
        fold_accs = []
        for tr, te in skf.split(Xi, y):
            sc = StandardScaler().fit(Xi[tr])
            Xtr = sc.transform(Xi[tr]); Xte = sc.transform(Xi[te])
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr, y[tr])
            preds = clf.predict(Xte)
            fold_accs.append(balanced_accuracy_score(y[te], preds))
        accs[li] = float(np.mean(fold_accs))
    return accs


def best_layer_confusion(X, y, layer, conditions):
    sc = StandardScaler().fit(X[:, layer, :])
    Xs = sc.transform(X[:, layer, :])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    skf = StratifiedKFold(n_splits=min(5, np.min(np.bincount(y))), shuffle=True, random_state=0)
    yhat = np.zeros_like(y)
    for tr, te in skf.split(Xs, y):
        clf.fit(Xs[tr], y[tr]); yhat[te] = clf.predict(Xs[te])
    cm = confusion_matrix(y, yhat, labels=list(range(len(conditions))))
    return cm


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen_Qwen3.5-0.8B"
    X, y, meta, conditions = load_dataset(model_name)
    n_samples, n_layers, hidden = X.shape
    print(f"Loaded N={n_samples}, layers={n_layers}, hidden={hidden}, conditions={conditions}")

    accs = per_layer_accuracy(X, y)
    chance = 1.0 / len(conditions)
    best_layer = int(np.argmax(accs))
    best_acc = float(accs[best_layer])

    cm = best_layer_confusion(X, y, best_layer, conditions)

    # Plot accuracy vs layer
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(accs, marker="o")
    ax.axhline(chance, ls="--", color="grey", label=f"chance = {chance:.2f}")
    ax.set_xlabel("layer index")
    ax.set_ylabel("balanced accuracy (5-fold CV)")
    ax.set_title(f"{model_name}: feed-condition linear-probe accuracy per layer\nN={n_samples}, best L{best_layer} acc={best_acc:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "01_fingerprint_accuracy_vs_layer.png", dpi=140)
    print(f"Saved {FIGS / '01_fingerprint_accuracy_vs_layer.png'}")

    # Confusion at best layer
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(conditions)), conditions, rotation=30, ha="right")
    ax.set_yticks(range(len(conditions)), conditions)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"confusion @ L{best_layer}  (acc {best_acc:.3f})")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="black" if v < cm.max() / 2 else "white")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(FIGS / "01_fingerprint_confusion.png", dpi=140)
    print(f"Saved {FIGS / '01_fingerprint_confusion.png'}")

    summary = {
        "model": model_name,
        "n_samples": n_samples,
        "n_layers": n_layers,
        "hidden": hidden,
        "conditions": conditions,
        "chance_acc": chance,
        "accs_per_layer": accs.tolist(),
        "best_layer": best_layer,
        "best_acc": best_acc,
        "confusion_at_best": cm.tolist(),
    }
    with open(ROOT / "results" / "01_fingerprint.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Best layer = {best_layer}, acc = {best_acc:.3f} (chance {chance:.3f})")


if __name__ == "__main__":
    main()
