"""Outcome-blind completion status for the cross-interface audit."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from run_cross_interface_audit_queue import MODELS, build_jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("results/cross_interface_audit.jsonl")
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=Path("results/cross_interface_audit_failures.jsonl"),
    )
    args = parser.parse_args()
    expected = {job.key for job in build_jobs()}
    observed = []
    invalid = 0
    if args.input.exists():
        for line in args.input.read_text().splitlines():
            try:
                row = json.loads(line)
                observed.append((row["model"], row["condition"], int(row["seed"])))
            except (json.JSONDecodeError, KeyError, ValueError):
                invalid += 1
    counts = Counter(observed)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    complete = set(observed) & expected
    failures = (
        len(args.failures.read_text().splitlines()) if args.failures.exists() else 0
    )
    print(
        f"complete={len(complete)}/700 pending={len(expected - complete)} "
        f"failures={failures} invalid={invalid} duplicates={duplicates}"
    )
    for model in MODELS:
        print(f"{model}: {sum(key[0] == model for key in complete)}/100")
    if invalid or duplicates or set(observed) - expected:
        raise SystemExit(2)
    if complete == expected:
        print("CROSS_INTERFACE_READY_FOR_FROZEN_ANALYSIS")


if __name__ == "__main__":
    main()

