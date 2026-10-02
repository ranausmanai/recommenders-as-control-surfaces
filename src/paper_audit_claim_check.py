#!/usr/bin/env python3
"""Fail if the audit manuscript's headline claims drift from frozen analyses."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "paper_audit.tex"
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    with (RESULTS / name).open() as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_text(text: str, fragment: str) -> None:
    normalized_text = re.sub(r"\s+", " ", text)
    normalized_fragment = re.sub(r"\s+", " ", fragment)
    require(
        normalized_fragment in normalized_text,
        f"Manuscript is missing frozen claim: {fragment}",
    )


def main() -> None:
    text = PAPER.read_text()
    controls = load("reviewer_controls_analysis.json")
    audit = load("susceptibility_audit_structured_analysis.json")
    cross = load("cross_interface_audit_analysis.json")
    frontier = load("codex_frontier_audit_analysis.json")

    llama_matched = next(
        row
        for row in controls["primary_comparisons"]
        if row["model"] == "llama3.2:3b"
        and row["experiment"] == "matched_selection"
    )
    llama_order = next(
        row
        for row in controls["primary_comparisons"]
        if row["model"] == "llama3.2:3b" and row["experiment"] == "pure_order"
    )
    llama_asymmetry = next(
        row
        for row in controls["default_direction_asymmetry"]
        if row["model"] == "llama3.2:3b"
    )

    primary = audit["primary_validation"]
    cross_primary = cross["primary_cross_interface"]
    portable = cross["gated_portable_audit"]
    disclosure = cross["disclosure_defense"]
    frontier_prediction = frontier["audit_full_prediction"]

    require(controls["n_records"] == 1800, "Reviewer-control record count changed")
    require(audit["n_records"] == 700, "Structured-audit record count changed")
    require(cross["retrieval_records"] == 700, "Cross-interface record count changed")
    require(frontier["records"] == 300, "Frontier record count changed")
    require(frontier["decision_labels"] == 1800, "Frontier label count changed")

    require(llama_matched["mean_ordinal_difference_left_minus_right"] == -0.68,
            "Matched Llama effect changed")
    require(llama_order["mean_ordinal_difference_left_minus_right"] == -0.06,
            "Pure-order Llama effect changed")
    require(round(llama_asymmetry["p_holm"], 3) == 0.079,
            "Asymmetry correction changed")

    require(round(primary["spearman_rho"], 3) == 0.855,
            "Held-out audit rho changed")
    require(round(primary["mae_improvement_over_zero"], 3) == 0.269,
            "Held-out audit MAE improvement changed")
    require(primary["directional_concordant"] == 12
            and primary["directional_cells"] == 13,
            "Held-out directional concordance changed")
    require(primary["classification"]["confusion"]
            == {"tp": 9, "tn": 5, "fp": 0, "fn": 4},
            "Held-out classification changed")

    require(round(cross_primary["spearman_rho"], 3) == 0.750,
            "Cross-interface rho changed")
    require(round(cross_primary["exact_one_sided_p"], 3) == 0.033,
            "Cross-interface p-value changed")
    require(round(portable["exact_one_sided_p"], 3) == 0.071,
            "Portable-audit p-value changed")
    require(round(disclosure["exact_one_sided_sign_flip_p"], 3) == 0.281,
            "Disclosure p-value changed")

    require(round(frontier_prediction["spearman_rho"], 3) == 0.951,
            "Frontier audit/full rho changed")
    require(frontier["significant_full_cells_holm"] == 0,
            "Frontier Holm-significant count changed")

    for fragment in (
        "3,500 agent jobs and 12,000 decision labels",
        "Spearman $\\rho=.855$",
        "12 of 13 material full-context effects",
        "1.00 precision and .846 balanced accuracy",
        "$\\rho=.750$, exact $p=.033$",
        "($p=.071$)",
        "($p=.281$)",
        "$\\rho=.951$",
        "no individually significant 45-document effect after Holm correction",
        "The incomplete current-model replication",
    ):
        require_text(text, fragment)

    print("CLAIM_CHECK_OK")


if __name__ == "__main__":
    main()
