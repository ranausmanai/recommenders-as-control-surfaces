"""Post-hoc task-mean baseline requested during FLMSec review.

This analysis is deliberately separate from the frozen confirmatory analyzer.
It tests whether held-out full-context effects can be predicted from each
task's average full-context effect in the four development model families.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "results" / "susceptibility_audit_structured_analysis.json"
DEFAULT_OUTPUT = ROOT / "results" / "reviewer_requested_baseline_analysis.json"


def analyze(input_path: Path) -> dict:
    analysis = json.loads(input_path.read_text())
    cells = analysis["cells"]

    development_by_task: dict[str, list[float]] = defaultdict(list)
    for cell in cells:
        if cell["role"] == "development":
            development_by_task[cell["task"]].append(float(cell["full_effect"]))

    task_means = {
        task: float(np.mean(values))
        for task, values in sorted(development_by_task.items())
    }
    holdout = [cell for cell in cells if cell["role"] == "holdout"]
    observed = np.asarray([float(cell["full_effect"]) for cell in holdout])
    audit_predictions = np.asarray(
        [float(cell["audit_effect"]) for cell in holdout]
    )
    task_mean_predictions = np.asarray(
        [task_means[cell["task"]] for cell in holdout]
    )

    audit_mae = float(np.mean(np.abs(observed - audit_predictions)))
    task_mean_mae = float(np.mean(np.abs(observed - task_mean_predictions)))
    audit_rho = float(spearmanr(audit_predictions, observed).statistic)
    task_mean_rho = float(spearmanr(task_mean_predictions, observed).statistic)

    return {
        "analysis_type": "post_hoc_reviewer_requested_baseline",
        "source": str(input_path.relative_to(ROOT)),
        "n_development_models": len(
            {cell["model"] for cell in cells if cell["role"] == "development"}
        ),
        "n_holdout_cells": len(holdout),
        "development_full_effect_mean_by_task": task_means,
        "task_mean_predictor": {
            "mean_absolute_error": task_mean_mae,
            "spearman_rho": task_mean_rho,
        },
        "counterfactual_audit_predictor": {
            "mean_absolute_error": audit_mae,
            "spearman_rho": audit_rho,
        },
        "audit_mae_reduction_vs_task_mean": (
            (task_mean_mae - audit_mae) / task_mean_mae
        ),
        "interpretation": (
            "The held-out audit result is not explained by a shared per-task "
            "susceptibility profile estimated from development models. This "
            "comparison is post-hoc and descriptive."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
