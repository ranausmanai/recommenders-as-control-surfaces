"""Cross-task generalization figure: per topic, P(attack target) at baseline vs
heavy attack, for the susceptible models. Saves paper_aisec/figures/paper_fig5_generalization.png.

Run: python3 notebooks/13_task_figure.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "decision_shift_tasks.jsonl"
OUT = ROOT / "paper_aisec" / "figures" / "paper_fig5_generalization.png"

TARGET = {"ubi": "A", "deploy_security": "C", "access_policy": "C",
          "vendor_security": "C", "ai_regulation": "C"}
LABEL = {"ubi": "UBI", "deploy_security": "deploy gate",
         "access_policy": "access policy", "vendor_security": "vendor adopt",
         "ai_regulation": "AI regulation"}
# Only the tasks run on BOTH models, so the two panels are a fair like-for-like
# comparison (no missing-cell gaps). The Llama-only boundary-probe tasks
# (vendor_security, ai_regulation) live in the table, not this head-to-head figure.
TOPIC_ORDER = ["ubi", "deploy_security", "access_policy"]
MODELS = [("llama3.2:3b", "Llama 3.2-3B"), ("gemma4:e4b", "Gemma 4-e4b")]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return p, max(0, c - half), min(1, c + half)


def main():
    rows = [json.loads(l) for l in open(DATA) if l.strip()] if DATA.exists() else []
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[(r["model"], r["topic"])][r["condition"]].append(r["choice"])

    BLUE, RED = "#4C72B0", "#C44E52"
    import matplotlib.lines as mlines
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 3.6), sharex=True)
    for ax, (mid, mname) in zip(axes, MODELS):
        topics = [t for t in TOPIC_ORDER
                  if cells.get((mid, t), {}).get("organic_random") and cells.get((mid, t), {}).get("heavy")]
        for i, t in enumerate(topics):
            tgt = TARGET[t]
            b = cells[(mid, t)]["organic_random"]; h = cells[(mid, t)]["heavy"]
            pb = sum(1 for c in b if c == tgt) / len(b)
            ph = sum(1 for c in h if c == tgt) / len(h)
            y = len(topics) - 1 - i  # first task on top
            moved = abs(ph - pb) >= 0.25
            ax.plot([pb, ph], [y, y], "-", color=(RED if moved else "#b9bec7"),
                    lw=3.2, zorder=1, solid_capstyle="round")
            ax.scatter(pb, y, s=95, color=BLUE, zorder=3)
            ax.scatter(ph, y, s=110, color=RED, zorder=3)
            # endpoint labels; merge if the two points coincide (e.g. 0% and 0%)
            if abs(ph - pb) < 0.04:
                ax.text(pb, y + 0.16, f"{round(pb*100)}%", ha="center", va="bottom",
                        fontsize=9, color="#2b3342")
            else:
                ax.text(pb, y + 0.17, f"{round(pb*100)}%", ha="center", va="bottom",
                        fontsize=9, color=BLUE)
                ax.text(ph, y + 0.17, f"{round(ph*100)}%", ha="center", va="bottom",
                        fontsize=9, color=RED, fontweight="bold" if moved else "normal")
        ax.set_yticks(range(len(topics)))
        ax.set_yticklabels([LABEL[t] for t in reversed(topics)], fontsize=11)
        ax.set_ylim(-0.5, len(topics) - 0.4)
        ax.set_xlim(-0.06, 1.08)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0%", "50%", "100%"])
        ax.set_xlabel("P(attack-target choice)")
        ax.set_title(mname, fontsize=13)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
    legb = mlines.Line2D([], [], color=BLUE, marker="o", linestyle="None", markersize=9, label="organic baseline")
    legr = mlines.Line2D([], [], color=RED, marker="o", linestyle="None", markersize=9, label="heavy attack")
    fig.legend(handles=[legb, legr], loc="lower center", ncol=2, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Attack generalization across decision tasks (baseline vs heavy injection)", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
