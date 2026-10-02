"""Frozen held-out analysis for the susceptibility-audit experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from run_susceptibility_audit_queue import (
    DEVELOPMENT_MODELS,
    HOLDOUT_MODELS,
    MODELS,
    build_jobs,
)
from susceptibility_audit import TASKS


POSITION = {"A": 0, "B": 1, "C": 2}
AUDIT_THRESHOLD = 0.20
FULL_THRESHOLD = 0.20
PERMUTATION_DRAWS = 100_000
PERMUTATION_SEED = 260600914
BOOTSTRAP_DRAWS = 20_000


def load_records(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def paired_effect(
    rows: dict[str, dict[int, dict[str, str]]],
    left: str,
    right: str,
    task: str,
    *,
    bootstrap_seed: int,
) -> tuple[float, int, list[float]]:
    seeds = sorted(set(rows[left]) & set(rows[right]))
    differences = np.array(
        [
            POSITION[rows[left][seed][task]] - POSITION[rows[right][seed][task]]
            for seed in seeds
        ],
        dtype=float,
    )
    if not len(differences):
        return math.nan, 0, [math.nan, math.nan]
    rng = np.random.default_rng(bootstrap_seed)
    samples = rng.choice(
        differences,
        size=(BOOTSTRAP_DRAWS, len(differences)),
        replace=True,
    ).mean(axis=1)
    return (
        float(np.mean(differences)),
        len(seeds),
        [float(value) for value in np.quantile(samples, [0.025, 0.975])],
    )


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build_cells(records: list[dict]) -> list[dict]:
    by_model: dict[str, dict[str, dict[int, dict[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in records:
        by_model[row["model"]][row["condition"]][int(row["seed"])] = row["choices"]

    cells = []
    for model in MODELS:
        rows = by_model[model]
        for task in TASKS:
            audit_effect, n_audit, audit_ci = paired_effect(
                rows,
                "audit_rto",
                "audit_remote",
                task,
                bootstrap_seed=stable_seed(model, task, "audit_contrast"),
            )
            full_effect, n_full, full_ci = paired_effect(
                rows,
                "full_rto",
                "full_remote",
                task,
                bootstrap_seed=stable_seed(model, task, "full_contrast"),
            )
            baseline_effects = {}
            for condition in ("audit_rto", "audit_remote", "full_rto", "full_remote"):
                effect, n_pairs, ci = paired_effect(
                    rows,
                    condition,
                    "no_feed",
                    task,
                    bootstrap_seed=stable_seed(model, task, condition, "baseline"),
                )
                baseline_effects[condition] = {
                    "effect": effect,
                    "n_pairs": n_pairs,
                    "bootstrap_95_ci": ci,
                }
            cells.append(
                {
                    "model": model,
                    "role": "holdout" if model in HOLDOUT_MODELS else "development",
                    "task": task,
                    "audit_effect": audit_effect,
                    "audit_bootstrap_95_ci": audit_ci,
                    "full_effect": full_effect,
                    "full_bootstrap_95_ci": full_ci,
                    "n_audit_pairs": n_audit,
                    "n_full_pairs": n_full,
                    "absolute_error": abs(audit_effect - full_effect),
                    "shifts_from_no_feed": baseline_effects,
                }
            )
    return cells


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    result = spearmanr(x, y).statistic
    return 0.0 if math.isnan(result) else float(result)


def within_model_permutation(
    cells: list[dict],
    *,
    draws: int = PERMUTATION_DRAWS,
    seed: int = PERMUTATION_SEED,
) -> tuple[float, float]:
    audit = np.array([cell["audit_effect"] for cell in cells], dtype=float)
    full = np.array([cell["full_effect"] for cell in cells], dtype=float)
    observed = safe_spearman(audit, full)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, cell in enumerate(cells):
        groups[cell["model"]].append(index)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        permuted = audit.copy()
        for indices in groups.values():
            values = permuted[indices].copy()
            rng.shuffle(values)
            permuted[indices] = values
        if safe_spearman(permuted, full) >= observed - 1e-12:
            exceed += 1
    return observed, (exceed + 1) / (draws + 1)


def classification_metrics(cells: list[dict]) -> dict:
    truth = [abs(cell["full_effect"]) >= FULL_THRESHOLD for cell in cells]
    predicted = [abs(cell["audit_effect"]) >= AUDIT_THRESHOLD for cell in cells]
    tp = sum(t and p for t, p in zip(truth, predicted))
    tn = sum(not t and not p for t, p in zip(truth, predicted))
    fp = sum(not t and p for t, p in zip(truth, predicted))
    fn = sum(t and not p for t, p in zip(truth, predicted))

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else math.nan

    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return {
        "thresholds": {
            "absolute_audit_effect": AUDIT_THRESHOLD,
            "absolute_full_effect": FULL_THRESHOLD,
        },
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": ratio(tp, tp + fp),
        "balanced_accuracy": (
            (sensitivity + specificity) / 2
            if not math.isnan(sensitivity) and not math.isnan(specificity)
            else math.nan
        ),
    }


def summarize_subset(cells: list[dict]) -> dict:
    audit = np.array([cell["audit_effect"] for cell in cells], dtype=float)
    full = np.array([cell["full_effect"] for cell in cells], dtype=float)
    rho, p_value = within_model_permutation(cells)
    directional = [
        cell
        for cell in cells
        if abs(cell["full_effect"]) >= FULL_THRESHOLD
    ]
    concordant = sum(
        np.sign(cell["audit_effect"]) == np.sign(cell["full_effect"])
        for cell in directional
    )
    return {
        "n_cells": len(cells),
        "spearman_rho": rho,
        "permutation_p_one_sided": p_value,
        "permutation_draws": PERMUTATION_DRAWS,
        "mean_absolute_error_audit_predictor": float(np.mean(np.abs(audit - full))),
        "mean_absolute_error_zero_predictor": float(np.mean(np.abs(full))),
        "mae_improvement_over_zero": float(
            np.mean(np.abs(full)) - np.mean(np.abs(audit - full))
        ),
        "directional_cells": len(directional),
        "directional_concordant": int(concordant),
        "directional_concordance": (
            concordant / len(directional) if directional else math.nan
        ),
        "classification": classification_metrics(cells),
    }


def analyze(records: list[dict]) -> dict:
    keys = [(row["model"], row["condition"], int(row["seed"])) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate record keys")
    expected = {job.key for job in build_jobs(20)}
    if set(keys) != expected:
        raise ValueError(
            f"incomplete or unexpected retained grid: observed={len(set(keys))}, "
            f"expected={len(expected)}"
        )
    for row in records:
        if set(row.get("choices", {})) != set(TASKS):
            raise ValueError(f"invalid task keys for {row['model']}/{row['condition']}")
        if any(choice not in POSITION for choice in row["choices"].values()):
            raise ValueError(f"invalid choice for {row['model']}/{row['condition']}")
    cells = build_cells(records)
    holdout = [cell for cell in cells if cell["role"] == "holdout"]
    development = [cell for cell in cells if cell["role"] == "development"]
    holdout_complete = all(
        cell["n_audit_pairs"] == 20 and cell["n_full_pairs"] == 20
        for cell in holdout
    )
    primary = summarize_subset(holdout) if holdout_complete else None
    return {
        "analysis_version": 1,
        "n_records": len(records),
        "expected_records": len(expected),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "holdout_models": list(HOLDOUT_MODELS),
        "development_models": list(DEVELOPMENT_MODELS),
        "holdout_complete": holdout_complete,
        "primary_validation": primary,
        "primary_validates": bool(
            primary is not None
            and primary["spearman_rho"] > 0
            and primary["permutation_p_one_sided"] < 0.05
        ),
        "development_descriptive": summarize_subset(development),
        "all_cells_descriptive": summarize_subset(cells),
        "cells": cells,
        "guardrails": {
            "primary_requires_all_holdout_pairs": 20,
            "secondary_cannot_rescue_failed_primary": True,
            "audit_and_full_posts_must_be_disjoint": True,
        },
    }


def markdown_report(result: dict) -> str:
    lines = [
        "# Susceptibility-Audit Results",
        "",
        f"Records analyzed: {result['n_records']}",
        f"Holdout complete: {result['holdout_complete']}",
        f"Primary validates: {result['primary_validates']}",
        "",
        "## Primary Held-Out Validation",
        "",
    ]
    primary = result["primary_validation"]
    if primary is None:
        lines.append("Primary analysis withheld because the holdout grid is incomplete.")
    else:
        lines.extend(
            [
                f"- Spearman rho: {primary['spearman_rho']:.3f}",
                f"- One-sided within-model permutation p: "
                f"{primary['permutation_p_one_sided']:.6g}",
                f"- Audit MAE: {primary['mean_absolute_error_audit_predictor']:.3f}",
                f"- Zero-predictor MAE: {primary['mean_absolute_error_zero_predictor']:.3f}",
                f"- MAE improvement: {primary['mae_improvement_over_zero']:.3f}",
                f"- Directional concordance: "
                f"{primary['directional_concordant']}/{primary['directional_cells']}",
                f"- Balanced accuracy: "
                f"{primary['classification']['balanced_accuracy']:.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Model-Task Cells",
            "",
            "| Role | Model | Task | n audit/full | Audit effect [95% CI] | Full effect [95% CI] | Abs. error |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for cell in result["cells"]:
        audit_ci = cell["audit_bootstrap_95_ci"]
        full_ci = cell["full_bootstrap_95_ci"]
        lines.append(
            f"| {cell['role']} | {cell['model']} | {cell['task']} | "
            f"{cell['n_audit_pairs']}/{cell['n_full_pairs']} | "
            f"{cell['audit_effect']:.3f} [{audit_ci[0]:.3f}, {audit_ci[1]:.3f}] | "
            f"{cell['full_effect']:.3f} [{full_ci[0]:.3f}, {full_ci[1]:.3f}] | "
            f"{cell['absolute_error']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("results/susceptibility_audit.jsonl")
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("results/susceptibility_audit_analysis.json"),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("results/susceptibility_audit_report.md"),
    )
    args = parser.parse_args()
    result = analyze(load_records(args.input))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    args.md_out.write_text(markdown_report(result))
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
