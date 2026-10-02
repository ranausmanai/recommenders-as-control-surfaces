#!/usr/bin/env python3
"""Validate the NeurIPS workshop manuscript against frozen analysis outputs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_neurips_workshop" / "paper.tex"
STYLE = ROOT / "paper_neurips_workshop" / "neurips_2026.sty"
RESULTS = ROOT / "results"
OFFICIAL_STYLE_SHA256 = (
    "c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a"
)


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
        f"Workshop manuscript is missing frozen claim: {fragment}",
    )


def main() -> None:
    text = PAPER.read_text()
    controls = load("reviewer_controls_analysis.json")
    audit = load("susceptibility_audit_structured_analysis.json")
    cross = load("cross_interface_audit_analysis.json")
    frontier = load("codex_frontier_audit_analysis.json")

    style_hash = hashlib.sha256(STYLE.read_bytes()).hexdigest()
    require(style_hash == OFFICIAL_STYLE_SHA256, "NeurIPS style file changed")
    require(
        r"\usepackage[dblblindworkshop]{neurips_2026}" in text,
        "Workshop paper is not using the double-blind workshop style",
    )
    require(
        r"\workshoptitle{Foundations of Language Model Security}" in text,
        "FLMSec workshop title is missing",
    )

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
    primary = audit["primary_validation"]
    cross_primary = cross["primary_cross_interface"]
    portable = cross["gated_portable_audit"]
    disclosure = cross["disclosure_defense"]
    frontier_prediction = frontier["audit_full_prediction"]

    require(controls["n_records"] == 1800, "Control record count changed")
    require(audit["n_records"] == 700, "Audit record count changed")
    require(cross["retrieval_records"] == 700, "RAG record count changed")
    require(frontier["records"] == 300, "Codex record count changed")
    require(frontier["decision_labels"] == 1800, "Codex label count changed")
    require(
        llama_matched["mean_ordinal_difference_left_minus_right"] == -0.68,
        "Matched-composition effect changed",
    )
    require(
        llama_order["mean_ordinal_difference_left_minus_right"] == -0.06,
        "Pure-order effect changed",
    )
    require(round(primary["spearman_rho"], 3) == 0.855, "Audit rho changed")
    require(
        round(primary["mae_improvement_over_zero"], 3) == 0.269,
        "Audit MAE improvement changed",
    )
    require(
        primary["classification"]["confusion"]
        == {"tp": 9, "tn": 5, "fp": 0, "fn": 4},
        "Audit classification changed",
    )
    require(
        round(cross_primary["spearman_rho"], 3) == 0.750,
        "Feed-to-RAG rho changed",
    )
    require(
        round(portable["exact_one_sided_p"], 3) == 0.071,
        "Direct audit-to-RAG p-value changed",
    )
    require(
        round(disclosure["exact_one_sided_sign_flip_p"], 3) == 0.281,
        "Disclosure p-value changed",
    )
    require(
        round(frontier_prediction["spearman_rho"], 3) == 0.951,
        "Codex audit/full rho changed",
    )
    require(
        frontier["significant_full_cells_holm"] == 0,
        "Codex Holm-significant count changed",
    )

    for fragment in (
        "3,500 agent jobs and 12,000 decision labels",
        "Spearman $\\rho=.855$",
        "12 of 13 material",
        "precision and specificity are 1.00",
        "$\\rho=.750$ (exact one-sided $p=.033$",
        "exact $p=.071$",
        "exact $p=.281$",
        "$\\rho=.951$",
        "none is significant after Holm correction",
        "not a safety certificate",
    ):
        require_text(text, fragment)

    print("NEURIPS_WORKSHOP_CLAIM_CHECK_OK")


if __name__ == "__main__":
    main()
