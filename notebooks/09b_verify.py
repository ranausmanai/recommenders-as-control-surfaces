"""Quick verification: random k-fold vs LORO on the same data, to confirm the
LORO numbers are real and not a code bug."""
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
mod = __import__("09_visible_history_baseline")

ACT_ROOT = os.environ.get("ACT_ROOT", "activations")


def random_kfold_acc(Xi, y, n_splits=5):
    skf = StratifiedKFold(n_splits=min(n_splits, int(np.min(np.bincount(y)))), shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(Xi, y):
        sc = StandardScaler().fit(Xi[tr])
        Xtr = sc.transform(Xi[tr]); Xte = sc.transform(Xi[te])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(Xte)))
    return float(np.mean(accs))


def loso_acc(Xi, y, seeds):
    """Leave one seed out — train on one set of seeds, test on the other."""
    unique = np.unique(seeds)
    accs = []
    for held in unique:
        tr = seeds != held; te = seeds == held
        sc = StandardScaler().fit(Xi[tr])
        Xtr = sc.transform(Xi[tr]); Xte = sc.transform(Xi[te])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(Xte)))
    return float(np.mean(accs))


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen_Qwen2.5-0.5B-Instruct"
    runs = mod.load_runs(model_name, act_root=ACT_ROOT)
    topics = sorted({r["topic"] for r in runs})
    print(f"=== {model_name}, ACT_ROOT={ACT_ROOT} ===")
    for topic in topics:
        topic_runs = [r for r in runs if r["topic"] == topic]
        Xa, ya, ga, ma, pols = mod.build_activation_dataset(topic_runs, model_name, topic=topic, act_root=ACT_ROOT)
        if Xa is None or len(np.unique(ya)) < 2:
            print(f"  {topic}: no data")
            continue
        # Pick the middle layer for a quick check
        L = Xa.shape[1] // 2
        seeds = np.array([m["seed"] for m in ma])
        # Sanity: are activations non-zero?
        sample_norm = float(np.linalg.norm(Xa[0, L, :]))
        # Random 5-fold (leaky)
        rk = random_kfold_acc(Xa[:, L, :], ya, n_splits=5)
        # LORO
        loro = mod.leave_one_group_out_acc(Xa[:, L, :], ya, ga)
        # LOSO (leave-one-seed-out)
        loso = loso_acc(Xa[:, L, :], ya, seeds)
        print(f"  {topic}: N={len(Xa)}, L={L}, ||sample||={sample_norm:.2f}, n_runs={len(np.unique(ga))}, n_seeds={len(np.unique(seeds))}")
        print(f"    random 5-fold = {rk:.3f}   |   LOSO = {loso:.3f}   |   LORO = {loro:.3f}")
        # Also: per-layer LORO trace
        N, NL, H = Xa.shape
        accs = []
        for li in range(0, NL, max(1, NL // 6)):
            accs.append((li, mod.leave_one_group_out_acc(Xa[:, li, :], ya, ga)))
        print("    LORO per layer sweep:", "  ".join(f"L{li}={a:.2f}" for li, a in accs))


if __name__ == "__main__":
    main()
