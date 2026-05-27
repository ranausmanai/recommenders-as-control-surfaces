"""Rigorous decision-shift statistics for the paper.

For each (model, condition) and pairwise model-vs-baseline:
  - exact counts (A, B, C, parse-fail)
  - proportions with Wilson 95% CI
  - Fisher's exact p-value (baseline=organic_random)
  - Bonferroni-corrected p-values across the comparison set
  - Cramér's V (3x2 effect size)

Outputs:
  results/10_rigorous_stats.json
  results/10_decision_tables.md   (paper-ready tables)
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import fisher_exact, chi2_contingency
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

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
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (center - half, center + half)


def fisher_p(a_x: int, a_n: int, b_x: int, b_n: int, alternative: str = "two-sided"):
    """Fisher's exact test, returning (odds_ratio, p)."""
    table = [[a_x, a_n - a_x], [b_x, b_n - b_x]]
    return fisher_exact(table, alternative=alternative)


def cramers_v(table: np.ndarray) -> float:
    if table.sum() == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2, _p, _dof, _exp = chi2_contingency(table)
    n = table.sum()
    return math.sqrt(chi2 / (n * (min(table.shape) - 1)))


def load_all_decisions():
    """Load ALL decision-shift JSONLs. Returns list of records with normalized fields."""
    files = [
        ("v1_simple", "decision_shift.jsonl"),
        ("v1_n20", "decision_shift_v1_n20.jsonl"),
        ("v2_continuous", "decision_shift_v2.jsonl"),
        ("lock_round1", "decision_shift_lock.jsonl"),
        ("adv_qwen0.5b", "decision_shift_adv.jsonl"),
        ("adv_modern", "decision_shift_adv_modern.jsonl"),
    ]
    out = []
    for source, fname in files:
        p = RESULTS / fname
        if not p.exists():
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["source"] = source
            # Normalize condition vs policy
            r["condition_norm"] = r.get("condition", r.get("policy"))
            # Normalize choice with the unified parser
            r["choice2"] = pc(r.get("raw", ""))
            out.append(r)
    return out


def summarize_cell(rows):
    """For a set of rollouts (same model, topic, condition), return summary dict."""
    n_total = len(rows)
    counts = Counter(r["choice2"] for r in rows)
    res = {
        "n_total": n_total,
        "A": counts.get("A", 0),
        "B": counts.get("B", 0),
        "C": counts.get("C", 0),
        "None": counts.get(None, 0),
    }
    n_valid = res["A"] + res["B"] + res["C"]
    res["n_valid"] = n_valid
    for k in ("A", "B", "C"):
        ci_lo, ci_hi = wilson_ci(res[k], n_valid) if n_valid > 0 else (float("nan"), float("nan"))
        res[f"{k}_prop"] = res[k] / n_valid if n_valid > 0 else float("nan")
        res[f"{k}_ci_lo"] = ci_lo
        res[f"{k}_ci_hi"] = ci_hi
    return res


def model_topic_grid(records):
    """Group records into (source, model, topic, condition_norm) cells, summarize."""
    by_cell = defaultdict(list)
    for r in records:
        key = (r["source"], r["model"], r["topic"], r["condition_norm"])
        by_cell[key].append(r)
    grid = {}
    for key, rows in by_cell.items():
        grid[key] = summarize_cell(rows)
    return grid


def adversarial_analysis(records):
    """Adversarial study (decision_shift_adv + decision_shift_adv_modern).

    For each (source, model, topic), compute:
      - cell summaries with Wilson CIs
      - pairwise Fisher vs organic_random for {A, B, C}
      - Bonferroni-corrected p-values (each model gets its own family of 5 comparisons)
    """
    adv = [r for r in records if r["source"] in ("adv_qwen0.5b", "adv_modern")]
    by_model = defaultdict(list)
    for r in adv:
        by_model[(r["source"], r["model"], r["topic"])].append(r)

    out = {}
    for (src, model, topic), rows in by_model.items():
        cells = {}
        for cond in ["organic_random", "organic_recency", "light", "heavy", "balanced", "disclosed_heavy"]:
            cr = [r for r in rows if r["condition_norm"] == cond]
            cells[cond] = summarize_cell(cr)
        # Pairwise Fisher
        base = cells.get("organic_random", {})
        comparisons = {}
        for cond in ["organic_recency", "light", "heavy", "balanced", "disclosed_heavy"]:
            cur = cells.get(cond, {})
            if base.get("n_valid", 0) < 5 or cur.get("n_valid", 0) < 5:
                continue
            for target in ["A", "B", "C"]:
                _, p = fisher_p(base[target], base["n_valid"], cur[target], cur["n_valid"])
                comparisons[(cond, target)] = p
        # Bonferroni: number of comparisons = 5 conditions × 3 targets = 15
        if comparisons:
            m = len(comparisons)
            corrected = {k: min(1.0, v * m) for k, v in comparisons.items()}
        else:
            corrected = {}
        out[(src, model, topic)] = {
            "cells": cells,
            "raw_p": comparisons,
            "bonferroni_p": corrected,
            "n_comparisons": len(comparisons),
        }
    return out


def trivial_defense_check(records):
    """Check whether 'defenses' (balanced, disclosed) just push every model to a fixed default.

    If a defense's mode answer is the same as the baseline's mode answer AND the proportions
    are close, the defense is mostly a no-op — it didn't FORCE a default, but it also didn't
    do much against the attack either.

    The dangerous case is: baseline = X, heavy attack moves model OFF X, but defense pushes
    everything to X regardless of attack content (i.e., defense overrides everything).
    That would be a 'trivial' defense.

    We approximate this by:
      - For each model where heavy injection shifts behavior significantly,
        check if the defense's distribution matches the baseline distribution (good) or
        instead matches a fixed default that ignores the attack content (suspicious).
    """
    # The cleanest test we can do with existing data:
    # For models where baseline != heavy, are balanced/disclosed closer to baseline or to a fixed value?
    adv = [r for r in records if r["source"] in ("adv_qwen0.5b", "adv_modern")]
    by_model = defaultdict(list)
    for r in adv:
        by_model[(r["source"], r["model"], r["topic"])].append(r)

    out = {}
    for (src, model, topic), rows in by_model.items():
        cells = {cond: summarize_cell([r for r in rows if r["condition_norm"] == cond])
                 for cond in ["organic_random", "heavy", "balanced", "disclosed_heavy"]}
        base = cells["organic_random"]; heavy = cells["heavy"]
        bal = cells["balanced"]; disc = cells["disclosed_heavy"]
        if any(c.get("n_valid", 0) < 5 for c in [base, heavy, bal, disc]):
            continue
        # Distance to baseline distribution (L1)
        def dist(a, b):
            return sum(abs(a.get(f"{k}_prop", 0) - b.get(f"{k}_prop", 0)) for k in ["A", "B", "C"]) / 2.0
        d_heavy_base = dist(heavy, base)
        d_bal_base = dist(bal, base)
        d_disc_base = dist(disc, base)
        out[(src, model, topic)] = {
            "L1_heavy_vs_baseline": d_heavy_base,
            "L1_balanced_vs_baseline": d_bal_base,
            "L1_disclosed_vs_baseline": d_disc_base,
            "verdict_balanced": "restores" if d_bal_base < 0.15 < d_heavy_base else
                                "partial" if d_bal_base < d_heavy_base else "no_effect",
            "verdict_disclosed": "restores" if d_disc_base < 0.15 < d_heavy_base else
                                 "partial" if d_disc_base < d_heavy_base else "no_effect",
        }
    return out


def render_paper_tables(adv_results, defense_check):
    """Markdown tables for the paper."""
    lines = []
    lines.append("# Adversarial Reactance: Paper Tables (auto-generated)\n")
    lines.append("Source files:")
    lines.append("- decision_shift_adv.jsonl (Qwen2.5-0.5B-Instruct, n=20/cond)")
    lines.append("- decision_shift_adv_modern.jsonl (Llama 3.2-3B, Qwen 3.5-{2B,9B}, Gemma 4-e4b, n=20/cond)\n")

    # Per-model decision tables
    for (src, model, topic), result in sorted(adv_results.items()):
        lines.append(f"\n## {model} / {topic}  (source: {src})")
        lines.append(f"\n| condition | A (95% CI) | B (95% CI) | C (95% CI) | n |")
        lines.append(f"|-----------|-----------|-----------|-----------|---|")
        for cond in ["organic_random", "organic_recency", "light", "heavy", "balanced", "disclosed_heavy"]:
            c = result["cells"].get(cond, {})
            if c.get("n_valid", 0) == 0:
                lines.append(f"| {cond} | — | — | — | 0 |")
                continue
            def fmt(k):
                p = c[f"{k}_prop"] * 100
                lo = c[f"{k}_ci_lo"] * 100; hi = c[f"{k}_ci_hi"] * 100
                return f"{p:.1f}% [{lo:.0f},{hi:.0f}]"
            lines.append(f"| {cond} | {fmt('A')} | {fmt('B')} | {fmt('C')} | {c['n_valid']} |")

        # Pairwise vs random table with Bonferroni
        lines.append(f"\n### Fisher's exact vs `organic_random`  (Bonferroni-corrected across {result['n_comparisons']} tests)")
        lines.append("| condition | A p_raw / p_corr | B p_raw / p_corr | C p_raw / p_corr |")
        lines.append("|---|---|---|---|")
        for cond in ["organic_recency", "light", "heavy", "balanced", "disclosed_heavy"]:
            row = [cond]
            for tgt in ["A", "B", "C"]:
                raw = result["raw_p"].get((cond, tgt))
                cor = result["bonferroni_p"].get((cond, tgt))
                if raw is None:
                    row.append("—")
                else:
                    sig = "***" if cor < 0.001 else "**" if cor < 0.01 else "*" if cor < 0.05 else ""
                    row.append(f"{raw:.4f} / {cor:.4f} {sig}")
            lines.append("| " + " | ".join(row) + " |")

        # Defense triviality check
        if (src, model, topic) in defense_check:
            d = defense_check[(src, model, topic)]
            lines.append(f"\n**Defense check** (L1 distance vs baseline, lower = closer to baseline):")
            lines.append(f"- heavy attack drift: {d['L1_heavy_vs_baseline']:.3f}")
            lines.append(f"- balanced defense vs baseline: {d['L1_balanced_vs_baseline']:.3f}  ({d['verdict_balanced']})")
            lines.append(f"- disclosed defense vs baseline: {d['L1_disclosed_vs_baseline']:.3f}  ({d['verdict_disclosed']})")
    return "\n".join(lines)


def main():
    records = load_all_decisions()
    print(f"Loaded {len(records)} decision-shift rollouts from {len(set(r['source'] for r in records))} sources.")

    adv = adversarial_analysis(records)
    defc = trivial_defense_check(records)

    # Save structured output
    out = {
        "n_total_rollouts": len(records),
        "sources": list({r["source"] for r in records}),
        "adversarial_results": {
            f"{k[0]}|{k[1]}|{k[2]}": {
                "cells": v["cells"],
                "raw_p": {f"{a}|{b}": p for (a, b), p in v["raw_p"].items()},
                "bonferroni_p": {f"{a}|{b}": p for (a, b), p in v["bonferroni_p"].items()},
                "n_comparisons": v["n_comparisons"],
            }
            for k, v in adv.items()
        },
        "defense_check": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in defc.items()},
    }
    (RESULTS / "10_rigorous_stats.json").write_text(json.dumps(out, indent=2, default=str))

    md = render_paper_tables(adv, defc)
    (RESULTS / "10_decision_tables.md").write_text(md)
    print(f"\nWrote {RESULTS / '10_rigorous_stats.json'}")
    print(f"Wrote {RESULTS / '10_decision_tables.md'}")
    # Brief stdout summary
    print("\n=== Headline cells (n=40+ or modern) ===")
    for (src, model, topic), result in sorted(adv.items()):
        h = result["raw_p"].get(("heavy", "C"))
        if h is not None:
            cb = result["bonferroni_p"].get(("heavy", "C"))
            print(f"  {model:30s} | heavy vs random on C: p_raw={h:.4f}  p_bonf={cb:.4f}")


if __name__ == "__main__":
    main()
