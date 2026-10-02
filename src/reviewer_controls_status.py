"""Report completion and ETA for the local reviewer-control queue."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from run_reviewer_controls_queue import DEFAULT_OUT, build_jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    jobs = build_jobs(args.n)
    target = {job.key for job in jobs}
    records = []
    if args.out.exists():
        with args.out.open() as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    done = {
        (r["model"], r["topic"], r["condition"], int(r["seed"]))
        for r in records
    }
    valid = [r for r in records if (r["model"], r["topic"], r["condition"], int(r["seed"])) in target]
    times = [float(r.get("elapsed_s", 0)) for r in valid if r.get("elapsed_s")]
    remaining = len(target - done)
    mean_s = sum(times) / len(times) if times else 0.0

    print(f"completed: {len(target & done)}/{len(target)} ({len(target & done) / len(target):.1%})")
    print(f"remaining: {remaining}")
    print(f"unparsable choices: {sum(r.get('choice') not in {'A', 'B', 'C'} for r in valid)}")
    if mean_s:
        print(f"observed mean: {mean_s:.1f}s/job")
        print(f"rough ETA: {remaining * mean_s / 3600:.1f}h")

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in valid:
        grouped[(record["model"], record["topic"], record["condition"])].append(record)
    for key in sorted(grouped):
        rows = grouped[key]
        choices = Counter(row.get("choice") for row in rows)
        print(f"{key[0]:14} {key[1]:18} {key[2]:20} n={len(rows):2} {dict(choices)}")


if __name__ == "__main__":
    main()
