"""Outcome-blind status for the Codex frontier audit."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from run_codex_frontier_audit_queue import MODELS, jobs


def main() -> None:
    output = Path("results/codex_frontier_audit.jsonl")
    failures_path = Path("results/codex_frontier_audit_failures.jsonl")
    expected = {job.key for job in jobs()}
    rows = (
        [json.loads(line) for line in output.read_text().splitlines() if line]
        if output.exists()
        else []
    )
    keys = [
        (row["model"], row["condition"], int(row["replicate"])) for row in rows
    ]
    counts = Counter(keys)
    duplicates = sum(value - 1 for value in counts.values() if value > 1)
    complete = set(keys) & expected
    failures = (
        len(failures_path.read_text().splitlines())
        if failures_path.exists()
        else 0
    )
    print(
        f"complete={len(complete)}/300 pending={len(expected-complete)} "
        f"failures={failures} duplicates={duplicates}"
    )
    for model in MODELS:
        print(f"{model}: {sum(key[0] == model for key in complete)}/100")
    if duplicates or set(keys) - expected:
        raise SystemExit(2)
    if complete == expected:
        print("CODEX_FRONTIER_READY_FOR_FROZEN_ANALYSIS")


if __name__ == "__main__":
    main()
