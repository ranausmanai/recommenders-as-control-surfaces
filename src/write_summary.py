"""Read all results/*.json and produce results/pilot_summary.md or full_summary.md.

Headline numbers: best layer, fingerprint accuracy, drift PC1 variance, behavior R^2,
cosine(PC1, static_stance) per topic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else "pilot_summary.md"
    fp = load_json(ROOT / "results" / "01_fingerprint.json")
    geo = load_json(ROOT / "results" / "02_geometry.json")
    beh = load_json(ROOT / "results" / "03_behavior.json")
    jc = load_json(ROOT / "results" / "04_jima_compare.json")

    lines: list[str] = []
    lines.append(f"# {out_name.replace('.md','').replace('_',' ').title()}\n")
    if fp is None:
        lines.append("**No fingerprint result found. Did 01_fingerprint.py run?**\n")
    else:
        lines.append("## 01 — Algorithm fingerprint\n")
        lines.append(f"- N samples: **{fp['n_samples']}**, n_layers: {fp['n_layers']}, hidden: {fp['hidden']}\n")
        lines.append(f"- Conditions: {fp['conditions']}\n")
        lines.append(f"- Chance accuracy: {fp['chance_acc']:.3f}\n")
        lines.append(f"- **Best layer: L{fp['best_layer']}**, balanced accuracy = **{fp['best_acc']:.3f}**\n")
        margin = fp["best_acc"] - fp["chance_acc"]
        lines.append(f"- Margin above chance: {margin:+.3f}\n")
        lines.append(f"- Confusion matrix (rows = true, cols = pred): `{fp['confusion_at_best']}`\n")
        lines.append(f"- Figure: `results/figures/01_fingerprint_accuracy_vs_layer.png`\n\n")

    if geo:
        lines.append("## 02 — Drift geometry\n")
        lines.append(f"- Best layer (inherited): L{geo['best_layer']}\n")
        lines.append(f"- Variance explained, top 5 PCs: {[f'{v:.3f}' for v in geo['explained_variance_top5']]}\n")
        lines.append(f"- Linear drift (PC1 > 50%): {geo['linear_drift_dominant']}\n")
        lines.append(f"- Figures: `02_geometry_pc1_pc2.png`, `02_geometry_pc1_by_turn.png`\n\n")

    if beh:
        lines.append("## 03 — Behavioral prediction\n")
        lines.append(f"- N (PC1_t, dstance) pairs: **{beh['n_pairs']}**\n")
        if beh.get("r2") is not None:
            lines.append(f"- Horizon H = {beh['horizon_H']} turns\n")
            lines.append(f"- **R^2 = {beh['r2']:.3f}**, slope = {beh['slope']:+.3f}, intercept = {beh['intercept']:+.3f}\n")
        else:
            lines.append(f"- {beh.get('note', '(no r2)')}\n")
        lines.append(f"- Figure: `03_behavior_pc1_predicts_stance.png`\n\n")

    if jc:
        lines.append("## 04 — Ji-Ma comparison\n")
        lines.append(f"- Best layer (inherited): L{jc['best_layer']}\n")
        for topic, r in jc.get("per_topic", {}).items():
            lines.append(f"- {topic}: cos(empirical drift PC1, static stance vec) = {r['cosine']:+.3f}  (|.|={r['cosine_abs']:.3f})\n")
        lines.append(f"- Figure: `04_jima_alignment.png`\n\n")

    # Verdict
    lines.append("## Verdict\n")
    if fp and fp["best_acc"] > fp["chance_acc"] + 0.07:
        lines.append("- **Pilot shows separation.** Feed-condition is decodable above chance margin. Scale up.\n")
    elif fp:
        lines.append("- **Pilot inconclusive.** Margin above chance is small. See null_result.md.\n")
    else:
        lines.append("- **No data analyzed yet.**\n")

    (ROOT / "results" / out_name).write_text("".join(lines))
    print(f"Wrote results/{out_name}")


if __name__ == "__main__":
    main()
