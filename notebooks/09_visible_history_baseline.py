"""09_visible_history_baseline: the critical control experiment.

For each (model, topic) cell, train a classifier on ONLY visible-history
features (no activations) to predict feed condition. Compare to the
activation probe accuracy from notebook 01.

If activation probe ≈ visible baseline → the "mechanistic" claim collapses
to "the probe is reading what's already in the visible conversation".

If activation probe ≫ visible baseline → the activations carry policy
information BEYOND what the visible history reveals — the mechanistic claim
survives.

Visible-history features per turn:
  - mean post stance shown this turn (5 posts)
  - std post stance shown this turn
  - count of each intensity bucket {calm, measured, heated, inflammatory}
  - count of each reaction {LIKE, SHARE, SKIP}
  - cumulative running counts of likes, shares, skips since turn 0
  - cumulative running mean stance shown
  - turn index
  - approximate token count of the rolling history

Also evaluate the ACTIVATION probe under hard splits:
  - leave-one-run-out (LORO): test never sees any turn from a training run
  - leave-one-seed-out (LOSO): like LORO but coarser
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

INTENSITIES = ["calm", "measured", "heated", "inflammatory"]
ACTIONS = ["LIKE", "SHARE", "SKIP"]


def load_runs(model_name: str, act_root: str = None):
    """Walk activations/<model>/... and yield run dicts with transcript + acts."""
    if act_root is None:
        act_root = os.environ.get("ACT_ROOT", "activations")
    base = ROOT / act_root / model_name
    runs = []
    for topic_dir in sorted(base.iterdir()):
        if not topic_dir.is_dir():
            continue
        for pol_dir in sorted(topic_dir.iterdir()):
            for seed_dir in sorted(pol_dir.iterdir()):
                tpath = seed_dir / "transcript.jsonl"
                if not tpath.exists():
                    continue
                entries = [json.loads(l) for l in open(tpath) if l.strip()]
                runs.append({
                    "topic": topic_dir.name,
                    "policy": pol_dir.name,
                    "seed": int(seed_dir.name.replace("seed", "")),
                    "transcript": entries,
                })
    return runs


def visible_features(transcript_so_far: list[dict], current_turn_entry: dict) -> np.ndarray:
    """Compute visible-history features for the current turn.

    transcript_so_far: list of prior turn entries.
    current_turn_entry: this turn's entry.

    Returns a 1D feature vector that ONLY uses information the agent itself
    could see in its visible context (posts, reactions, turn index, length).
    """
    feats = []
    # This turn's posts
    posts = current_turn_entry.get("posts", [])
    stances = [p.get("stance", 0) for p in posts]
    feats.append(np.mean(stances) if stances else 0)
    feats.append(np.std(stances) if stances else 0)
    intensity_counts = {k: 0 for k in INTENSITIES}
    for p in posts:
        if p.get("intensity") in intensity_counts:
            intensity_counts[p["intensity"]] += 1
    for k in INTENSITIES:
        feats.append(intensity_counts[k])
    # This turn's reactions
    reactions = current_turn_entry.get("reactions", [])
    action_counts = {k: 0 for k in ACTIONS}
    for r in reactions:
        if r.get("action") in action_counts:
            action_counts[r["action"]] += 1
    for k in ACTIONS:
        feats.append(action_counts[k])
    # Cumulative running counts (history visible to model)
    cum_like = cum_share = cum_skip = 0
    cum_stance_sum = 0.0
    cum_n_posts = 0
    for h in transcript_so_far:
        for r in h.get("reactions", []):
            if r.get("action") == "LIKE":
                cum_like += 1
            elif r.get("action") == "SHARE":
                cum_share += 1
            elif r.get("action") == "SKIP":
                cum_skip += 1
        for p in h.get("posts", []):
            cum_stance_sum += p.get("stance", 0)
            cum_n_posts += 1
    feats += [cum_like, cum_share, cum_skip]
    feats.append(cum_stance_sum / max(1, cum_n_posts))
    feats.append(cum_n_posts)
    # Turn index
    feats.append(current_turn_entry.get("turn", len(transcript_so_far)))
    # Crude history-length proxy: characters of reaction_user + reaction_asst
    history_chars = sum(len(h.get("reaction_user", "")) + len(h.get("reaction_asst", ""))
                        for h in transcript_so_far)
    feats.append(history_chars / 1000.0)
    return np.array(feats, dtype=np.float32)


def build_visible_dataset(runs: list[dict]):
    """Returns X (N, D), y (N,), groups (N,) — group = run index for LORO."""
    X, y, groups, meta = [], [], [], []
    policies = sorted({r["policy"] for r in runs})
    pol_to_y = {p: i for i, p in enumerate(policies)}
    for ri, run in enumerate(runs):
        for ti, entry in enumerate(run["transcript"]):
            feats = visible_features(run["transcript"][:ti], entry)
            X.append(feats); y.append(pol_to_y[run["policy"]])
            groups.append(ri)
            meta.append({"topic": run["topic"], "policy": run["policy"],
                         "seed": run["seed"], "turn": ti, "run_idx": ri})
    return np.array(X), np.array(y), np.array(groups), meta, policies


def build_activation_dataset(runs: list[dict], model_name: str, topic: str | None = None,
                              act_root: str | None = None):
    """Stack saved .pt activations into (N, L, H). Returns same groups labels for LORO."""
    if act_root is None:
        act_root = os.environ.get("ACT_ROOT", "activations")
    X, y, groups, meta = [], [], [], []
    policies = sorted({r["policy"] for r in runs if topic is None or r["topic"] == topic})
    pol_to_y = {p: i for i, p in enumerate(policies)}
    for ri, run in enumerate(runs):
        if topic and run["topic"] != topic:
            continue
        # Reconstruct the activation directory from (model, topic, policy, seed)
        seed_dir = ROOT / act_root / model_name / run["topic"] / run["policy"] / f"seed{run['seed']}"
        if not seed_dir.exists():
            continue
        pt_files = sorted(seed_dir.glob("turn*.pt"))
        # Cap at the number of transcript entries
        n_turns = len(run["transcript"])
        for ti, pt in enumerate(pt_files[:n_turns]):
            t = torch.load(pt, map_location="cpu", weights_only=True)
            X.append(t.numpy().astype(np.float32))
            y.append(pol_to_y[run["policy"]])
            groups.append(ri)
            meta.append({"topic": run["topic"], "policy": run["policy"],
                         "seed": run["seed"], "turn": ti, "run_idx": ri})
    if not X:
        return None, None, None, None, policies
    return np.stack(X), np.array(y), np.array(groups), meta, policies


def leave_one_group_out_acc(Xi, y, groups):
    """LOGO: train on all but one group, test on that group. Returns mean balanced acc."""
    accs = []
    for g in np.unique(groups):
        tr = groups != g; te = groups == g
        if tr.sum() == 0 or te.sum() == 0:
            continue
        sc = StandardScaler().fit(Xi[tr])
        Xtr = sc.transform(Xi[tr]); Xte = sc.transform(Xi[te])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        try:
            clf.fit(Xtr, y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(Xte)))
        except Exception:
            continue
    return float(np.mean(accs)) if accs else float("nan")


def best_layer_loro(X, y, groups, every: int = 2):
    """For activation tensor of shape (N, L, H), return (best_layer, best_LORO acc).

    `every` strides through layers to bound runtime on 32-layer models.
    """
    N, L, H = X.shape
    accs = np.full(L, np.nan)
    for li in range(0, L, every):
        accs[li] = leave_one_group_out_acc(X[:, li, :], y, groups)
    # Also do the final layer
    if np.isnan(accs[L - 1]):
        accs[L - 1] = leave_one_group_out_acc(X[:, L - 1, :], y, groups)
    best = int(np.nanargmax(accs))
    return best, float(accs[best]), accs


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else ["Qwen_Qwen2.5-0.5B-Instruct"]
    rows = []

    for model_name in models:
        runs = load_runs(model_name)
        if not runs:
            print(f"[skip] no runs for {model_name}")
            continue
        topics = sorted({r["topic"] for r in runs})
        print(f"\n=== {model_name} ({len(runs)} runs, topics={topics}) ===")

        for topic in topics:
            topic_runs = [r for r in runs if r["topic"] == topic]
            if len(topic_runs) < 3:
                continue
            # Visible-history baseline (LORO)
            Xv, yv, gv, mv, pols = build_visible_dataset(topic_runs)
            if len(np.unique(yv)) < 2:
                continue
            vis_acc = leave_one_group_out_acc(Xv, yv, gv)

            # Activation probe (LORO + LOSO)
            Xa, ya, ga, ma, pols_a = build_activation_dataset(topic_runs, model_name, topic=topic)
            act_loro = float("nan")
            best_layer = -1
            if Xa is not None and len(np.unique(ya)) >= 2:
                best_layer, act_loro, _accs = best_layer_loro(Xa, ya, ga)

            row = {
                "model": model_name, "topic": topic,
                "visible_LORO_acc": vis_acc,
                "activation_LORO_acc": act_loro,
                "best_layer": best_layer,
                "n_runs": len(topic_runs),
                "n_samples_visible": int(len(Xv)),
                "n_samples_activation": int(0 if Xa is None else len(Xa)),
                "n_policies": int(len(np.unique(yv))),
                "chance": 1.0 / max(1, len(np.unique(yv))),
            }
            rows.append(row)
            print(f"  {topic}: visible={vis_acc:.3f}, activation@L{best_layer}={act_loro:.3f}, "
                  f"diff={act_loro - vis_acc:+.3f}, n_runs={row['n_runs']}, chance={row['chance']:.2f}")

    # Save and plot
    out = {"rows": rows}
    (ROOT / "results" / "09_visible_baseline.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote results/09_visible_baseline.json")

    if rows:
        # Bar plot: for each (model, topic), grouped bars (visible vs activation)
        labels = [f"{r['model'].split('_', 1)[-1][:20]}\n{r['topic'][:8]}" for r in rows]
        vis = [r["visible_LORO_acc"] for r in rows]
        act = [r["activation_LORO_acc"] for r in rows]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 4.5))
        w = 0.4
        ax.bar(x - w / 2, vis, w, label="visible-history baseline (LORO)", color="grey")
        ax.bar(x + w / 2, act, w, label="activation probe (LORO)", color="C0")
        ax.axhline(1.0 / 3, ls="--", color="black", linewidth=0.6, label="chance = 0.33")
        ax.set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("balanced accuracy (leave-one-run-out)")
        ax.set_title("Activation probe vs visible-history baseline (LORO)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIGS / "09_visible_vs_activation.png", dpi=140)
        print(f"Saved {FIGS / '09_visible_vs_activation.png'}")


if __name__ == "__main__":
    main()
