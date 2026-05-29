"""Generate the 4 paper-ready figures for the adversarial-feed-injection paper.

Outputs:
  results/figures/paper_fig1_cross_model_attack.png
  results/figures/paper_fig2_dose_response.png
  results/figures/paper_fig3_generator_swap.png
  results/figures/paper_fig4_defenses.png
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Common style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
})

CHOICE_RE = re.compile(r"Recommendation\s*:\s*[^A-Za-z]*([ABCabc])\b")


def pc(t: str):
    m = CHOICE_RE.search(t)
    if m:
        return m.group(1).upper()
    for ln in t.splitlines():
        for tk in ln.split():
            s = tk.strip(".,():;<>[]*\"'").upper()
            if s in ("A", "B", "C"):
                return s
    return None


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def sig_marker(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ---------------------------------------------------------------------------
# Figure 1: Cross-model attack — %C (remote-first) under organic vs heavy
# ---------------------------------------------------------------------------
def fig1_cross_model():
    rows = load(ROOT / "results" / "decision_shift_adv_modern.jsonl")
    grid = defaultdict(Counter)
    for r in rows:
        grid[(r["model"], r["condition"])][pc(r["raw"])] += 1

    models = [
        ("llama3.2:3b", "Llama 3.2-3B\n(Meta)"),
        ("gemma4:e4b", "Gemma 4-e4b\n(Google)"),
        ("qwen3.5:2b", "Qwen 3.5-2B\n(Alibaba)"),
        ("qwen3.5:9b", "Qwen 3.5-9B\n(Alibaba)"),
    ]
    labels = [m[1] for m in models]

    base_p, base_lo, base_hi = [], [], []
    atk_p, atk_lo, atk_hi = [], [], []
    sig_marks = []
    for mid, _ in models:
        cb = grid[(mid, "organic_random")]; ch = grid[(mid, "heavy")]
        nb = sum(cb[x] for x in "ABC"); nh = sum(ch[x] for x in "ABC")
        bp = cb["C"] / nb if nb else 0
        ap = ch["C"] / nh if nh else 0
        blo, bhi = wilson_ci(cb["C"], nb)
        alo, ahi = wilson_ci(ch["C"], nh)
        _, p = fisher_exact([[cb["C"], nb - cb["C"]], [ch["C"], nh - ch["C"]]])
        base_p.append(bp); base_lo.append(blo); base_hi.append(bhi)
        atk_p.append(ap); atk_lo.append(alo); atk_hi.append(ahi)
        sig_marks.append(sig_marker(p))

    # Dumbbell encoding: a 0% value is a visible dot on the zero line, not an
    # invisible bar. Each model is a row; baseline dot -> heavy dot.
    import matplotlib.lines as mlines
    BLUE, RED = "#5b8def", "#e25c5c"
    fig, ax = plt.subplots(figsize=(9, 4.3))
    n = len(labels)
    for i in range(n):
        y = n - 1 - i  # first model on top
        bp, ap = base_p[i], atk_p[i]
        moved = sig_marks[i] != "n.s."
        ax.plot([bp, ap], [y, y], "-", color=(RED if moved else "#b9bec7"),
                lw=3.2, zorder=1, solid_capstyle="round")
        ax.scatter(bp, y, s=110, color=BLUE, zorder=3)
        ax.scatter(ap, y, s=120, color=RED, zorder=3)
        if abs(ap - bp) < 0.04:
            ax.text(bp, y + 0.18, f"{round(bp*100)}%", ha="center", va="bottom",
                    fontsize=9, color="#2b3342")
        else:
            ax.text(bp, y + 0.19, f"{round(bp*100)}%", ha="center", va="bottom", fontsize=9, color=BLUE)
            ax.text(ap, y + 0.19, f"{round(ap*100)}%", ha="center", va="bottom",
                    fontsize=9, color=RED, fontweight="bold")
        ax.text(1.04, y, sig_marks[i], ha="left", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(range(n)); ax.set_yticklabels(list(reversed(labels)))
    ax.set_ylim(-0.5, n - 0.4); ax.set_xlim(-0.05, 1.16)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0%", "50%", "100%"])
    ax.set_xlabel("Probability of recommending 'fully remote'")
    ax.set_title("Adversarial feed injection shifts decisions in 2 of 4 modern LLMs (n=20 per cell)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    legb = mlines.Line2D([], [], color=BLUE, marker="o", linestyle="None", markersize=10, label="Organic random (baseline)")
    legr = mlines.Line2D([], [], color=RED, marker="o", linestyle="None", markersize=10, label="Heavy adversarial injection")
    ax.legend(handles=[legb, legr], loc="center", bbox_to_anchor=(0.62, 0.30), frameon=True)
    fig.tight_layout()
    fig.savefig(FIGS / "paper_fig1_cross_model_attack.png")
    plt.close(fig)
    print("Saved paper_fig1_cross_model_attack.png")


# ---------------------------------------------------------------------------
# Figure 2: Dose-response curve on Llama 3.2 (E3)
# ---------------------------------------------------------------------------
def fig2_dose_response():
    rows = load(ROOT / "results" / "decision_shift_followup.jsonl")
    dose = defaultdict(Counter)
    for r in rows:
        if r["tag"] == "E3_dose_response":
            dose[r["condition"]][pc(r["raw"])] += 1

    levels = [0, 1, 2, 3, 4, 5]
    xs, ys, los, his = [], [], [], []
    for d in levels:
        c = dose[f"dose{d}"]
        n = sum(c[x] for x in "ABC")
        p = c["C"] / n if n else 0
        lo, hi = wilson_ci(c["C"], n)
        xs.append(d); ys.append(p); los.append(lo); his.append(hi)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, marker="o", markersize=10, linewidth=2.4, color="#e25c5c", zorder=3, label="P(remote-first)")
    ax.fill_between(xs, los, his, color="#e25c5c", alpha=0.2, label="95% CI")
    ax.axhline(1.0, ls="--", color="#5b8def", alpha=0.7, label="Baseline (organic, no adversarial)")
    ax.set_xlabel("Adversarial posts injected per 5-post batch")
    ax.set_ylabel("Probability of recommending 'fully remote'")
    ax.set_title("Dose-response: Llama 3.2-3B, n=20 per dose (chi² p = 0.006)")
    ax.set_xticks(levels)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    # Annotate threshold
    ax.annotate("threshold ≈ 2 adv/batch",
                xy=(2, ys[2]), xytext=(2.6, 0.78),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "paper_fig2_dose_response.png")
    plt.close(fig)
    print("Saved paper_fig2_dose_response.png")


# ---------------------------------------------------------------------------
# Figure 3: Generator-swap — attack replicates & strengthens with different writer
# ---------------------------------------------------------------------------
def fig3_generator_swap():
    rows_modern = load(ROOT / "results" / "decision_shift_adv_modern.jsonl")
    rows_fu = load(ROOT / "results" / "decision_shift_followup.jsonl")
    claude = defaultdict(Counter)
    gemma = defaultdict(Counter)
    for r in rows_modern:
        if r["model"] == "llama3.2:3b":
            claude[r["condition"]][pc(r["raw"])] += 1
    for r in rows_fu:
        if r["tag"] == "E2_gemma_pool":
            gemma[r["condition"]][pc(r["raw"])] += 1

    conds = ["organic_random", "heavy", "balanced", "disclosed_heavy"]
    nice = ["Organic random\n(baseline)", "Heavy attack", "Balanced\ndefense", "Disclosed\ndefense"]
    cp, cp_lo, cp_hi = [], [], []
    gp, gp_lo, gp_hi = [], [], []
    for cond in conds:
        c1 = claude[cond]; c2 = gemma[cond]
        n1 = sum(c1[x] for x in "ABC"); n2 = sum(c2[x] for x in "ABC")
        p1 = c1["C"] / n1 if n1 else 0
        p2 = c2["C"] / n2 if n2 else 0
        l1, h1 = wilson_ci(c1["C"], n1); l2, h2 = wilson_ci(c2["C"], n2)
        cp.append(p1); cp_lo.append(l1); cp_hi.append(h1)
        gp.append(p2); gp_lo.append(l2); gp_hi.append(h2)
    x = np.arange(len(conds)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, cp, w, yerr=[np.array(cp) - np.array(cp_lo), np.array(cp_hi) - np.array(cp)],
           capsize=4, color="#5b8def", edgecolor="black", linewidth=0.4, label="Claude-written posts")
    ax.bar(x + w / 2, gp, w, yerr=[np.array(gp) - np.array(gp_lo), np.array(gp_hi) - np.array(gp)],
           capsize=4, color="#e6a01a", edgecolor="black", linewidth=0.4, label="Gemma-written posts (generator swap)")
    ax.set_xticks(x); ax.set_xticklabels(nice)
    ax.set_ylabel("P(remote-first) on Llama 3.2-3B")
    ax.set_ylim(0, 1.15)
    ax.set_title("Generator-swap: attack replicates and strengthens with different post writer\n"
                 "(heavy attack on Gemma-written posts: Fisher p = 3 × 10⁻¹⁰)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "paper_fig3_generator_swap.png")
    plt.close(fig)
    print("Saved paper_fig3_generator_swap.png")


# ---------------------------------------------------------------------------
# Figure 4: Defenses — restoration relative to baseline
# ---------------------------------------------------------------------------
def fig4_defenses():
    rows_modern = load(ROOT / "results" / "decision_shift_adv_modern.jsonl")
    rows_fu = load(ROOT / "results" / "decision_shift_followup.jsonl")
    claude = defaultdict(Counter)
    gemma = defaultdict(Counter)
    for r in rows_modern:
        if r["model"] == "llama3.2:3b":
            claude[r["condition"]][pc(r["raw"])] += 1
    for r in rows_fu:
        if r["tag"] == "E2_gemma_pool":
            gemma[r["condition"]][pc(r["raw"])] += 1

    conditions = [("Heavy attack", "heavy"), ("+ Balanced defense", "balanced"), ("+ Disclosed defense", "disclosed_heavy")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, pool, name in [(axes[0], claude, "Claude-written posts"), (axes[1], gemma, "Gemma-written posts")]:
        # Baseline reference
        cb = pool["organic_random"]; nb = sum(cb[x] for x in "ABC")
        base_p = cb["C"] / nb if nb else 0
        ax.axhline(base_p, ls="--", color="#5b8def", linewidth=1.5, label=f"Baseline P(remote) = {base_p:.2f}")
        ys, lo, hi, ps = [], [], [], []
        for label, cond in conditions:
            c = pool[cond]; n = sum(c[x] for x in "ABC")
            p_remote = c["C"] / n if n else 0
            l, h = wilson_ci(c["C"], n)
            ys.append(p_remote); lo.append(l); hi.append(h)
            # Fisher vs heavy (defense effectiveness)
            if cond != "heavy":
                ch = pool["heavy"]; nh = sum(ch[x] for x in "ABC")
                _, p = fisher_exact([[ch["C"], nh - ch["C"]], [c["C"], n - c["C"]]])
                ps.append(p)
            else:
                ps.append(None)
        x = np.arange(len(conditions))
        colors = ["#e25c5c", "#7cba6c", "#8a6bbf"]
        ax.bar(x, ys, 0.55,
               yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)],
               capsize=4, color=colors, edgecolor="black", linewidth=0.4)
        for i, p in enumerate(ps):
            if p is not None:
                ax.text(i, max(hi[i] + 0.04, 0.08), sig_marker(p), ha="center", fontsize=11, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([c[0] for c in conditions])
        ax.set_title(name)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("P(remote-first) on Llama 3.2-3B")
    fig.suptitle("Defenses restore baseline decisions on the susceptible model\n"
                 "(significance = defense vs heavy attack, Fisher's exact)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "paper_fig4_defenses.png")
    plt.close(fig)
    print("Saved paper_fig4_defenses.png")


# ---------------------------------------------------------------------------
def main():
    fig1_cross_model()
    fig2_dose_response()
    fig3_generator_swap()
    fig4_defenses()
    print("\nAll 4 paper figures generated in results/figures/")


if __name__ == "__main__":
    main()
