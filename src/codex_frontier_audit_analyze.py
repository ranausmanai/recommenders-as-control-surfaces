"""Frozen analysis for the Codex frontier-agent audit."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from run_codex_frontier_audit_queue import MODELS, jobs
from susceptibility_audit import TASKS


POSITION = {"A": 0, "B": 1, "C": 2}
DRAW_COUNT = 100_000
RNG_SEED = 560028


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def safe_rho(x: np.ndarray, y: np.ndarray) -> float:
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    value = spearmanr(x, y).statistic
    return 0.0 if math.isnan(value) else float(value)


def paired_differences(data: dict, left: str, right: str, task: str) -> np.ndarray:
    replicates = sorted(set(data[left]) & set(data[right]))
    if replicates != list(range(20)):
        raise ValueError(f"incomplete paired replicates for {left}/{right}/{task}")
    return np.array(
        [
            POSITION[data[left][replicate][task]]
            - POSITION[data[right][replicate][task]]
            for replicate in replicates
        ],
        dtype=float,
    )


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    samples = rng.choice(
        values, size=(20_000, len(values)), replace=True
    ).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def sign_flip_p(values: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(float(np.mean(values)))
    signs = rng.choice((-1, 1), size=(DRAW_COUNT, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.sum(null >= observed - 1e-12) + 1) / (DRAW_COUNT + 1))


def analyze(rows: list[dict]) -> dict:
    expected = {job.key for job in jobs()}
    keys = [
        (row["model"], row["condition"], int(row["replicate"])) for row in rows
    ]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("frontier dataset does not match frozen 300-call grid")
    if any(
        set(row.get("choices", {})) != set(TASKS)
        or any(choice not in POSITION for choice in row["choices"].values())
        for row in rows
    ):
        raise ValueError("invalid frontier choices")

    indexed = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        indexed[row["model"]][row["condition"]][int(row["replicate"])] = row[
            "choices"
        ]
    rng = np.random.default_rng(RNG_SEED)
    cells = []
    for model in MODELS:
        for task in TASKS:
            audit = paired_differences(
                indexed[model], "audit_rto", "audit_remote", task
            )
            full = paired_differences(
                indexed[model], "full_rto", "full_remote", task
            )
            cells.append(
                {
                    "model": model,
                    "task": task,
                    "audit_effect": float(np.mean(audit)),
                    "audit_bootstrap_95_ci": bootstrap_ci(audit, rng),
                    "full_effect": float(np.mean(full)),
                    "full_bootstrap_95_ci": bootstrap_ci(full, rng),
                    "full_p_raw": sign_flip_p(full, rng),
                    "full_rto_from_no_evidence": float(
                        np.mean(
                            paired_differences(
                                {
                                    "left": indexed[model]["full_rto"],
                                    "right": indexed[model]["no_evidence"],
                                },
                                "left",
                                "right",
                                task,
                            )
                        )
                    ),
                    "full_remote_from_no_evidence": float(
                        np.mean(
                            paired_differences(
                                {
                                    "left": indexed[model]["full_remote"],
                                    "right": indexed[model]["no_evidence"],
                                },
                                "left",
                                "right",
                                task,
                            )
                        )
                    ),
                }
            )
    adjusted = multipletests(
        [cell["full_p_raw"] for cell in cells], method="holm"
    )[1]
    for cell, value in zip(cells, adjusted):
        cell["full_p_holm"] = float(value)
        cell["full_reject_holm_005"] = bool(value < 0.05)

    audit = np.array([cell["audit_effect"] for cell in cells])
    full = np.array([cell["full_effect"] for cell in cells])
    observed = safe_rho(audit, full)
    groups = {
        model: [index for index, cell in enumerate(cells) if cell["model"] == model]
        for model in MODELS
    }
    exceed = 0
    for _ in range(DRAW_COUNT):
        permuted = audit.copy()
        for indices in groups.values():
            values = permuted[indices].copy()
            rng.shuffle(values)
            permuted[indices] = values
        if safe_rho(permuted, full) >= observed - 1e-12:
            exceed += 1
    prediction_p = (exceed + 1) / (DRAW_COUNT + 1)
    return {
        "analysis_version": 1,
        "records": len(rows),
        "decision_labels": len(rows) * 6,
        "audit_full_prediction": {
            "spearman_rho": observed,
            "permutation_p_one_sided": prediction_p,
            "validates": observed > 0 and prediction_p < 0.05,
        },
        "significant_full_cells_holm": sum(
            cell["full_reject_holm_005"] for cell in cells
        ),
        "cells": cells,
    }


def markdown(result: dict) -> str:
    prediction = result["audit_full_prediction"]
    lines = [
        "# Codex Frontier-Agent Audit",
        "",
        f"Records: {result['records']}",
        f"Decision labels: {result['decision_labels']}",
        f"Audit/full rho: {prediction['spearman_rho']:.3f}",
        f"One-sided permutation p: {prediction['permutation_p_one_sided']:.6g}",
        f"Audit validates: {prediction['validates']}",
        f"Holm-significant full-effect cells: "
        f"{result['significant_full_cells_holm']}/18",
        "",
        "| Model | Task | Audit effect | Full effect | 95% CI | Holm p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cell in result["cells"]:
        ci = cell["full_bootstrap_95_ci"]
        lines.append(
            f"| {cell['model']} | {cell['task']} | {cell['audit_effect']:.3f} | "
            f"{cell['full_effect']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{cell['full_p_holm']:.4g} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("results/codex_frontier_audit.jsonl")
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("results/codex_frontier_audit_analysis.json"),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("results/codex_frontier_audit_report.md"),
    )
    args = parser.parse_args()
    result = analyze(load(args.input))
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    args.md_out.write_text(markdown(result))
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
