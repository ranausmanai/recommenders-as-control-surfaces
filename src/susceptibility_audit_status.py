"""Outcome-blind completion audit for the susceptibility experiment."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from run_susceptibility_audit_queue import MODELS, build_jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument(
        "--input", type=Path, default=Path("results/susceptibility_audit.jsonl")
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=Path("results/susceptibility_audit_failures.jsonl"),
    )
    args = parser.parse_args()

    expected = {job.key for job in build_jobs(args.n)}
    observed = []
    invalid = 0
    if args.input.exists():
        with args.input.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    observed.append(
                        (row["model"], row["condition"], int(row["seed"]))
                    )
                except (json.JSONDecodeError, KeyError, ValueError):
                    invalid += 1
    counts = Counter(observed)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    observed_set = set(observed)
    complete = observed_set & expected
    unexpected = observed_set - expected
    failures = 0
    if args.failures.exists():
        with args.failures.open() as handle:
            failures = sum(1 for line in handle if line.strip())
    print(
        f"complete={len(complete)}/{len(expected)} "
        f"pending={len(expected - complete)} failures={failures} "
        f"invalid={invalid} duplicates={duplicates} unexpected={len(unexpected)}"
    )
    for model in MODELS:
        model_expected = {key for key in expected if key[0] == model}
        print(f"{model}: {len(complete & model_expected)}/{len(model_expected)}")
    if invalid or duplicates or unexpected:
        raise SystemExit(2)
    if complete == expected:
        print("AUDIT_READY_FOR_FROZEN_ANALYSIS")


if __name__ == "__main__":
    main()
