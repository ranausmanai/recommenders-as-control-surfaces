"""08_patching_analysis: score all 06c_*.json files and produce dose-response,
layer-specificity, and cross-direction plots.

Inputs:
  results/06c_patch_<model>_<topic>_L<layer>_t<target_policy>_s<scale>.json

Outputs:
  results/figures/08_dose_response.png
  results/figures/08_layer_specificity.png
  results/figures/08_cross_model.png
  results/08_patching_summary.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sentiment_probe import score_stance

FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def parse_patch_file(p: Path):
    d = json.loads(p.read_text())
    rec = {
        "model": d["model"],
        "topic": d["topic"],
        "layer": d["layer"],
        "scale": d["scale"],
        "target_policy": d.get("target_policy", "recency"),
        "n_turns": d["n_turns"],
        "direction_norm": d.get("direction_norm"),
    }
    sb, st_, ss = [], [], []
    for i in range(d["n_turns"]):
        sb.append(score_stance(d["baseline"][i], d["topic"]))
        st_.append(score_stance(d["target"][i], d["topic"]))
        ss.append(score_stance(d["sham"][i], d["topic"]))
    rec["baseline_mean"] = float(np.mean(sb))
    rec["target_mean"] = float(np.mean(st_))
    rec["sham_mean"] = float(np.mean(ss))
    rec["target_minus_baseline"] = rec["target_mean"] - rec["baseline_mean"]
    rec["sham_minus_baseline"] = rec["sham_mean"] - rec["baseline_mean"]
    rec["target_minus_sham"] = rec["target_minus_baseline"] - rec["sham_minus_baseline"]
    # per-turn diffs for std error
    diffs_target = [st_[i] - sb[i] for i in range(d["n_turns"])]
    diffs_sham = [ss[i] - sb[i] for i in range(d["n_turns"])]
    rec["target_diff_std"] = float(np.std(diffs_target, ddof=1)) if len(diffs_target) > 1 else 0.0
    rec["sham_diff_std"] = float(np.std(diffs_sham, ddof=1)) if len(diffs_sham) > 1 else 0.0
    rec["target_diff_sem"] = rec["target_diff_std"] / max(1, len(diffs_target)) ** 0.5
    rec["sham_diff_sem"] = rec["sham_diff_std"] / max(1, len(diffs_sham)) ** 0.5
    rec["target_diff_per_turn"] = diffs_target
    rec["sham_diff_per_turn"] = diffs_sham
    return rec


def main():
    files = sorted((ROOT / "results").glob("06c_patch_*.json"))
    print(f"Found {len(files)} patch files")
    records = []
    for p in files:
        try:
            records.append(parse_patch_file(p))
            r = records[-1]
            print(f"  {p.name}  T-B={r['target_minus_baseline']:+.3f}  S-B={r['sham_minus_baseline']:+.3f}  T-S={r['target_minus_sham']:+.3f}")
        except Exception as e:
            print(f"  FAIL {p.name}: {e}")

    # (1) Dose response: model=Qwen2.5-3B, layer=17, target=recency, vary scale.
    dose = [r for r in records if r["model"] == "Qwen_Qwen2.5-3B-Instruct"
            and r["layer"] == 17 and r["target_policy"] == "recency"]
    dose.sort(key=lambda r: r["scale"])
    if len(dose) >= 3:
        xs = [r["scale"] for r in dose]
        yt = [r["target_minus_baseline"] for r in dose]
        ys = [r["sham_minus_baseline"] for r in dose]
        et = [r["target_diff_sem"] for r in dose]
        es = [r["sham_diff_sem"] for r in dose]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(xs, yt, yerr=et, marker="o", label="target patch  (μ(recency) − μ(random))", color="C3", linewidth=2)
        ax.errorbar(xs, ys, yerr=es, marker="s", label="sham patch (random direction, same magnitude)", color="grey", linewidth=2)
        ax.axhline(0, ls="--", color="black", linewidth=0.5)
        ax.set_xlabel("patch scale  (multiple of ||direction||)")
        ax.set_ylabel("stance shift  (target/sham − baseline)\n-2 = full RTO, +2 = full remote")
        ax.set_title("Dose-response: causal patching at L17 on Qwen2.5-3B-Instruct, remote_work")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / "08_dose_response.png", dpi=140)
        print(f"\nSaved {FIGS / '08_dose_response.png'}")

    # (2) Layer specificity: model=Qwen2.5-3B, scale=10, vary layer.
    layer = [r for r in records if r["model"] == "Qwen_Qwen2.5-3B-Instruct"
             and r["scale"] == 10.0 and r["target_policy"] == "recency"]
    layer.sort(key=lambda r: r["layer"])
    if len(layer) >= 3:
        xs = [r["layer"] for r in layer]
        yt = [r["target_minus_baseline"] for r in layer]
        ys = [r["sham_minus_baseline"] for r in layer]
        et = [r["target_diff_sem"] for r in layer]
        es = [r["sham_diff_sem"] for r in layer]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(xs, yt, yerr=et, marker="o", label="target patch", color="C3", linewidth=2)
        ax.errorbar(xs, ys, yerr=es, marker="s", label="sham patch", color="grey", linewidth=2)
        ax.axhline(0, ls="--", color="black", linewidth=0.5)
        ax.set_xlabel("patch layer")
        ax.set_ylabel("stance shift  (target/sham − baseline)")
        ax.set_title("Layer specificity: patch at scale=10 on Qwen2.5-3B-Instruct, remote_work")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / "08_layer_specificity.png", dpi=140)
        print(f"Saved {FIGS / '08_layer_specificity.png'}")

    # (3) Cross-model: scale=10, target=recency, at each model's best layer.
    cross = [r for r in records if r["scale"] == 10.0 and r["target_policy"] == "recency"]
    cross_by_model = {}
    for r in cross:
        cross_by_model.setdefault(r["model"], r)
    if len(cross_by_model) >= 2:
        names = sorted(cross_by_model)
        yt = [cross_by_model[n]["target_minus_baseline"] for n in names]
        ys = [cross_by_model[n]["sham_minus_baseline"] for n in names]
        et = [cross_by_model[n]["target_diff_sem"] for n in names]
        es = [cross_by_model[n]["sham_diff_sem"] for n in names]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.6), 4.5))
        w = 0.35
        ax.bar(x - w / 2, yt, w, yerr=et, label="target patch", color="C3", capsize=4)
        ax.bar(x + w / 2, ys, w, yerr=es, label="sham patch", color="grey", capsize=4)
        ax.axhline(0, ls="--", color="black", linewidth=0.5)
        ax.set_xticks(x, [n.replace("_", "/", 1).replace("Instruct", "Inst") for n in names], rotation=20, ha="right")
        ax.set_ylabel("stance shift  (mean over 12 turns)")
        ax.set_title("Cross-model causal patching (scale=10, target=recency)")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIGS / "08_cross_model.png", dpi=140)
        print(f"Saved {FIGS / '08_cross_model.png'}")

    # Save summary JSON
    summary = []
    for r in records:
        summary.append({k: r[k] for k in r if not k.endswith("per_turn")})
    (ROOT / "results" / "08_patching_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary: {len(summary)} records")


if __name__ == "__main__":
    main()
