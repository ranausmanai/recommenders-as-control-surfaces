#!/usr/bin/env python3
"""Validate the self-contained FLMSec/arXiv artifact release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


STUDIES = {
    "reviewer_controls": {
        "path": "results/reviewer_controls.jsonl",
        "count": 1800,
        "labels": 1800,
        "key": ("model", "topic", "condition", "seed"),
    },
    "susceptibility_audit": {
        "path": "results/susceptibility_audit_structured.jsonl",
        "count": 700,
        "labels": 4200,
        "key": ("model", "condition", "seed"),
    },
    "cross_interface": {
        "path": "results/cross_interface_audit.jsonl",
        "count": 700,
        "labels": 4200,
        "key": ("model", "condition", "seed"),
    },
    "codex_frontier": {
        "path": "results/codex_frontier_audit.jsonl",
        "count": 300,
        "labels": 1800,
        "key": ("model", "condition", "replicate"),
    },
}

HASH_MAPS = {
    "results/reviewer_controls_manifest.json": {
        "preregistration": "experiments/reviewer_controls_preregistration.md",
        "runner": "src/reviewer_controls.py",
        "queue": "src/run_reviewer_controls_queue.py",
        "matched_pool": "posts/pool_gemma.jsonl",
    },
    "results/susceptibility_audit_structured_manifest.json": {
        "amendment": "experiments/susceptibility_audit_structured_amendment.md",
        "structured_runner": "src/susceptibility_audit_structured.py",
        "queue": "src/run_susceptibility_audit_structured_queue.py",
        "original_runner": "src/susceptibility_audit.py",
        "analysis": "src/susceptibility_audit_analyze.py",
        "base_snapshot": "results/susceptibility_audit_frozen_incomplete_629.jsonl",
    },
    "results/cross_interface_audit_manifest.json": {
        "preregistration": "experiments/cross_interface_audit_preregistration.md",
        "runner": "src/cross_interface_audit.py",
        "queue": "src/run_cross_interface_audit_queue.py",
        "analysis": "src/cross_interface_audit_analyze.py",
        "structured_runtime": "src/susceptibility_audit_structured.py",
        "matched_pool": "posts/pool_gemma.jsonl",
    },
    "results/codex_frontier_audit_manifest.json": {
        "preregistration": "experiments/codex_frontier_audit_preregistration.md",
        "schema": "experiments/codex_frontier_choices.schema.json",
        "runner": "src/codex_frontier_audit.py",
        "queue": "src/run_codex_frontier_audit_queue.py",
        "analysis": "src/codex_frontier_audit_analyze.py",
        "matched_pool": "posts/pool_gemma.jsonl",
    },
}

REQUIRED = (
    "ARTIFACTS.md",
    "LICENSE",
    "CITATION.cff",
    "requirements.txt",
    "experiments/susceptibility_audit_preregistration.md",
    "experiments/codex_frontier_choices.schema.json",
    "results/reviewer_controls_analysis_amendment.md",
    "results/susceptibility_audit_structured_report.md",
    "results/cross_interface_audit_report.md",
    "results/codex_frontier_audit_report.md",
    "notebooks/15_audit_paper_figures.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_checksums(root: Path) -> None:
    checksum_file = root / "SHA256SUMS"
    if not checksum_file.exists():
        return
    for line in checksum_file.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        require(relative != "SHA256SUMS", "checksum file cannot hash itself")
        path = root / relative
        require(path.is_file(), f"checksum target missing: {relative}")
        require(sha256(path) == expected, f"checksum mismatch: {relative}")


def validate_studies(root: Path) -> None:
    total_jobs = 0
    total_labels = 0
    for name, spec in STUDIES.items():
        rows = jsonl(root / spec["path"])
        require(len(rows) == spec["count"], f"{name}: wrong record count")
        keys = {tuple(row[field] for field in spec["key"]) for row in rows}
        require(len(keys) == len(rows), f"{name}: duplicate job keys")
        labels = 0
        for row in rows:
            if "choices" in row:
                require(len(row["choices"]) == 6, f"{name}: expected six choices")
                require(set(row["choices"].values()) <= {"A", "B", "C"},
                        f"{name}: invalid choice")
                labels += 6
            else:
                require(row.get("choice") in {"A", "B", "C"},
                        f"{name}: invalid choice")
                labels += 1
        require(labels == spec["labels"], f"{name}: wrong label count")
        total_jobs += len(rows)
        total_labels += labels
    require(total_jobs == 3500, "confirmatory job total is not 3,500")
    require(total_labels == 12000, "decision-label total is not 12,000")


def validate_launch_hashes(root: Path) -> None:
    for manifest_path, mapping in HASH_MAPS.items():
        manifest = json.loads((root / manifest_path).read_text())
        expected = manifest["design"]["hashes"]
        for name, relative in mapping.items():
            require(sha256(root / relative) == expected[name],
                    f"launch-hash mismatch: {relative}")

    reviewer = json.loads(
        (root / "results/reviewer_controls_manifest.json").read_text()
    )
    frozen_hash = reviewer["design"]["hashes"]["analysis"]
    corrected_hash = sha256(root / "src/reviewer_controls_analyze.py")
    require(corrected_hash != frozen_hash,
            "expected documented reviewer-control analyzer amendment")
    require((root / "results/reviewer_controls_analysis_amendment.md").exists(),
            "matched-control analysis amendment missing")


def validate_headlines(root: Path) -> None:
    audit = json.loads(
        (root / "results/susceptibility_audit_structured_analysis.json").read_text()
    )
    cross = json.loads(
        (root / "results/cross_interface_audit_analysis.json").read_text()
    )
    frontier = json.loads(
        (root / "results/codex_frontier_audit_analysis.json").read_text()
    )
    baseline = json.loads(
        (root / "results/reviewer_requested_baseline_analysis.json").read_text()
    )
    require(round(audit["primary_validation"]["spearman_rho"], 3) == .855,
            "audit headline changed")
    require(round(cross["primary_cross_interface"]["spearman_rho"], 3) == .750,
            "cross-interface headline changed")
    require(round(frontier["audit_full_prediction"]["spearman_rho"], 3) == .951,
            "frontier headline changed")
    require(round(baseline["task_mean_predictor"]["mean_absolute_error"], 3) == .369,
            "post-hoc baseline changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    for relative in REQUIRED:
        require((root / relative).is_file(), f"required file missing: {relative}")
    pool = jsonl(root / "posts/pool_gemma.jsonl")
    require(len(pool) == 100 and len({row["id"] for row in pool}) == 100,
            "matched pool must contain 100 unique posts")
    incomplete = jsonl(root / "results/susceptibility_audit_frozen_incomplete_629.jsonl")
    require(len(incomplete) == 629, "frozen incomplete audit must contain 629 jobs")
    validate_checksums(root)
    validate_studies(root)
    validate_launch_hashes(root)
    validate_headlines(root)
    print("ARTIFACT_RELEASE_OK jobs=3500 labels=12000 incomplete=629 pool=100")


if __name__ == "__main__":
    main()
