"""07_scaling: how does fingerprint accuracy scale with parameter count?

Reads results/01_fingerprint_<model>.json for the Qwen2.5 family at 4 sizes
plus SmolLM2-1.7B as a non-Qwen anchor; produces a single accuracy-vs-params
plot per topic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# (model dir name, family, param count in millions)
MODELS = [
    ("Qwen_Qwen2.5-0.5B-Instruct", "Qwen2.5", 494),
    ("Qwen_Qwen2.5-1.5B-Instruct", "Qwen2.5", 1543),
    ("Qwen_Qwen2.5-3B-Instruct", "Qwen2.5", 3086),
    ("Qwen_Qwen2.5-7B-Instruct", "Qwen2.5 (4-bit)", 7615),
    ("HuggingFaceTB_SmolLM2-1.7B-Instruct", "SmolLM2 (HF)", 1707),
    ("01-ai_Yi-1.5-6B-Chat", "Yi (4-bit)", 6061),
    ("HuggingFaceH4_zephyr-7b-beta", "Zephyr/Mistral (4-bit)", 7242),
    ("stabilityai_stablelm-2-1_6b-chat", "StableLM-2", 1644),
    ("tiiuae_Falcon3-3B-Instruct", "Falcon3", 3232),
]


def main():
    rows = []
    for d, family, params in MODELS:
        p = ROOT / "results" / f"01_fingerprint_{d}.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        rows.append({"model": d, "family": family, "params_M": params,
                     "best_layer": s["best_layer"], "best_acc": s["best_acc"],
                     "n_layers": s["n_layers"], "hidden": s["hidden"], "n": s["n_samples"]})

    # Plot: x = params (log), y = best-layer accuracy.
    fig, ax = plt.subplots(figsize=(8.5, 5))
    qwen_rows = [r for r in rows if "Qwen" in r["family"]]
    other_rows = [r for r in rows if "Qwen" not in r["family"]]

    qx = [r["params_M"] for r in qwen_rows]
    qy = [r["best_acc"] for r in qwen_rows]
    ax.plot(qx, qy, "o-", label="Qwen2.5-Instruct family", color="C0", markersize=11, linewidth=2)
    for r in qwen_rows:
        ax.annotate(f'{r["params_M"]/1000:.1f}B', (r["params_M"], r["best_acc"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=8)

    family_colors = {"SmolLM2 (HF)": "C1", "Yi (4-bit)": "C2", "Zephyr/Mistral (4-bit)": "C3",
                      "StableLM-2": "C4", "Falcon3": "C5"}
    family_markers = {"SmolLM2 (HF)": "s", "Yi (4-bit)": "D", "Zephyr/Mistral (4-bit)": "^",
                       "StableLM-2": "v", "Falcon3": "P"}
    for r in other_rows:
        ax.scatter([r["params_M"]], [r["best_acc"]],
                   marker=family_markers.get(r["family"], "*"),
                   color=family_colors.get(r["family"], "black"),
                   s=180, label=r["family"], edgecolors="black", linewidths=0.6, zorder=3)
        ax.annotate(f'{r["params_M"]/1000:.1f}B', (r["params_M"], r["best_acc"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=8)

    ax.axhline(1.0 / 3, ls="--", color="grey", label="chance = 0.33")
    ax.set_xscale("log")
    ax.set_xlabel("parameter count (millions, log scale)")
    ax.set_ylabel("best-layer balanced accuracy\n(3-class probe, 3 topics combined)")
    ax.set_title("Fingerprint accuracy vs model scale")
    ax.set_ylim(0.3, 1.02)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "07_accuracy_vs_scale.png", dpi=140)
    print(f"Saved {FIGS / '07_accuracy_vs_scale.png'}")
    print("\nTable:")
    for r in rows:
        print(f"  {r['model']:50s} {r['params_M']:>6d}M  L{r['best_layer']:>2d}/{r['n_layers']}  acc={r['best_acc']:.3f}")


if __name__ == "__main__":
    main()
