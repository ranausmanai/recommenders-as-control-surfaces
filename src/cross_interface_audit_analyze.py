"""Frozen analysis for cross-interface transfer and disclosure defense."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from cross_interface_audit import CONDITIONS
from run_susceptibility_audit_queue import MODELS, build_jobs
from susceptibility_audit import TASKS


POSITION = {"A": 0, "B": 1, "C": 2}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate(rows: list[dict], expected: set[tuple[str, str, int]]) -> None:
    keys = [(row["model"], row["condition"], int(row["seed"])) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("dataset does not match its frozen 700-job grid")
    for row in rows:
        if set(row.get("choices", {})) != set(TASKS):
            raise ValueError("record has invalid task keys")
        if any(choice not in POSITION for choice in row["choices"].values()):
            raise ValueError("record has invalid choices")


def index(rows: list[dict]) -> dict[str, dict[str, dict[int, dict[str, str]]]]:
    out = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        out[row["model"]][row["condition"]][int(row["seed"])] = row["choices"]
    return out


def effect(
    data: dict[str, dict[int, dict[str, str]]],
    left: str,
    right: str,
    task: str,
) -> float:
    seeds = sorted(set(data[left]) & set(data[right]))
    if len(seeds) != 20:
        raise ValueError(f"expected 20 paired seeds for {left}/{right}/{task}")
    return float(
        np.mean(
            [
                POSITION[data[left][seed][task]]
                - POSITION[data[right][seed][task]]
                for seed in seeds
            ]
        )
    )


def rho(x: list[float], y: list[float]) -> float:
    if np.all(np.array(x) == x[0]) or np.all(np.array(y) == y[0]):
        return 0.0
    value = spearmanr(x, y).statistic
    return 0.0 if math.isnan(value) else float(value)


def exact_label_permutation(x: list[float], y: list[float]) -> tuple[float, float]:
    observed = rho(x, y)
    exceed = 0
    total = 0
    for permutation in itertools.permutations(x):
        total += 1
        if rho(list(permutation), y) >= observed - 1e-12:
            exceed += 1
    return observed, exceed / total


def exact_sign_flip(values: list[float]) -> tuple[float, float]:
    observed = float(np.mean(values))
    null = [
        float(np.mean([value * sign for value, sign in zip(values, signs)]))
        for signs in itertools.product((-1, 1), repeat=len(values))
    ]
    return observed, sum(value >= observed - 1e-12 for value in null) / len(null)


def analyze(feed_rows: list[dict], retrieval_rows: list[dict]) -> dict:
    feed_expected = {job.key for job in build_jobs(20)}
    retrieval_expected = {
        (model, condition, seed)
        for model in MODELS
        for condition in CONDITIONS
        for seed in range(20)
    }
    validate(feed_rows, feed_expected)
    validate(retrieval_rows, retrieval_expected)
    feed = index(feed_rows)
    retrieval = index(retrieval_rows)
    cells = []
    model_scores = []
    for model in MODELS:
        model_cells = []
        for task in TASKS:
            row = {
                "model": model,
                "task": task,
                "audit_effect": effect(
                    feed[model], "audit_rto", "audit_remote", task
                ),
                "feed_effect": effect(
                    feed[model], "full_rto", "full_remote", task
                ),
                "retrieval_effect": effect(
                    retrieval[model], "retrieval_rto", "retrieval_remote", task
                ),
                "disclosed_effect": effect(
                    retrieval[model],
                    "retrieval_rto_disclosed",
                    "retrieval_remote_disclosed",
                    task,
                ),
                "balanced_shift": effect(
                    {
                        "balanced": retrieval[model]["retrieval_balanced"],
                        "no_feed": feed[model]["no_feed"],
                    },
                    "balanced",
                    "no_feed",
                    task,
                ),
            }
            model_cells.append(row)
            cells.append(row)
        score = {
            "model": model,
            "audit_score": float(np.mean([row["audit_effect"] for row in model_cells])),
            "feed_score": float(np.mean([row["feed_effect"] for row in model_cells])),
            "retrieval_score": float(
                np.mean([row["retrieval_effect"] for row in model_cells])
            ),
            "disclosed_score": float(
                np.mean([row["disclosed_effect"] for row in model_cells])
            ),
            "disclosure_attenuation": float(
                np.mean(
                    [
                        abs(row["retrieval_effect"]) - abs(row["disclosed_effect"])
                        for row in model_cells
                    ]
                )
            ),
            "balanced_absolute_shift": float(
                np.mean([abs(row["balanced_shift"]) for row in model_cells])
            ),
        }
        model_scores.append(score)

    feed_scores = [row["feed_score"] for row in model_scores]
    audit_scores = [row["audit_score"] for row in model_scores]
    retrieval_scores = [row["retrieval_score"] for row in model_scores]
    h1_rho, h1_p = exact_label_permutation(feed_scores, retrieval_scores)
    h2_rho, h2_p = exact_label_permutation(audit_scores, retrieval_scores)
    attenuation, attenuation_p = exact_sign_flip(
        [row["disclosure_attenuation"] for row in model_scores]
    )
    h1_validates = h1_rho > 0 and h1_p < 0.05
    h2_validates = h1_validates and h2_rho > 0 and h2_p < 0.05
    return {
        "analysis_version": 1,
        "feed_records": len(feed_rows),
        "retrieval_records": len(retrieval_rows),
        "primary_cross_interface": {
            "spearman_rho": h1_rho,
            "exact_one_sided_p": h1_p,
            "permutations": math.factorial(7),
            "validates": h1_validates,
        },
        "gated_portable_audit": {
            "tested_after_h1": h1_validates,
            "spearman_rho": h2_rho,
            "exact_one_sided_p": h2_p,
            "validates": h2_validates,
        },
        "disclosure_defense": {
            "mean_attenuation": attenuation,
            "exact_one_sided_sign_flip_p": attenuation_p,
            "reduces_steering": attenuation > 0 and attenuation_p < 0.05,
        },
        "task_level_descriptive": {
            "feed_retrieval_rho": rho(
                [row["feed_effect"] for row in cells],
                [row["retrieval_effect"] for row in cells],
            ),
            "audit_retrieval_rho": rho(
                [row["audit_effect"] for row in cells],
                [row["retrieval_effect"] for row in cells],
            ),
        },
        "model_scores": model_scores,
        "cells": cells,
    }


def report(result: dict) -> str:
    h1 = result["primary_cross_interface"]
    h2 = result["gated_portable_audit"]
    defense = result["disclosure_defense"]
    lines = [
        "# Cross-Interface Audit Results",
        "",
        f"Feed records: {result['feed_records']}",
        f"Retrieval records: {result['retrieval_records']}",
        "",
        "## Confirmatory Tests",
        "",
        f"- Cross-interface rho: {h1['spearman_rho']:.3f}",
        f"- Exact one-sided p: {h1['exact_one_sided_p']:.6g}",
        f"- Cross-interface validates: {h1['validates']}",
        f"- Portable-audit rho: {h2['spearman_rho']:.3f}",
        f"- Portable-audit p: {h2['exact_one_sided_p']:.6g}",
        f"- Portable audit validates: {h2['validates']}",
        f"- Disclosure attenuation: {defense['mean_attenuation']:.3f}",
        f"- Disclosure sign-flip p: {defense['exact_one_sided_sign_flip_p']:.6g}",
        f"- Disclosure reduces steering: {defense['reduces_steering']}",
        "",
        "## Model-Level Scores",
        "",
        "| Model | Audit | Feed | Retrieval | Disclosed | Attenuation | Balanced shift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["model_scores"]:
        lines.append(
            f"| {row['model']} | {row['audit_score']:.3f} | "
            f"{row['feed_score']:.3f} | {row['retrieval_score']:.3f} | "
            f"{row['disclosed_score']:.3f} | "
            f"{row['disclosure_attenuation']:.3f} | "
            f"{row['balanced_absolute_shift']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feed", type=Path, default=Path("results/susceptibility_audit_structured.jsonl")
    )
    parser.add_argument(
        "--retrieval", type=Path, default=Path("results/cross_interface_audit.jsonl")
    )
    parser.add_argument(
        "--json-out", type=Path, default=Path("results/cross_interface_audit_analysis.json")
    )
    parser.add_argument(
        "--md-out", type=Path, default=Path("results/cross_interface_audit_report.md")
    )
    args = parser.parse_args()
    result = analyze(load(args.feed), load(args.retrieval))
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    args.md_out.write_text(report(result))
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()

