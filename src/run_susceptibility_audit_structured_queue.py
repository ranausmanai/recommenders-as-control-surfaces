"""Frozen queue for the structured-decoding susceptibility amendment."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from feed_policies import load_pool
from reviewer_controls import MATCHED_POOL, validate_matched_pool
from run_susceptibility_audit_queue import build_jobs, load_done
from susceptibility_audit_structured import load_base_records, run_structured_job


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = ROOT / "results" / "susceptibility_audit_frozen_incomplete_629.jsonl"
DEFAULT_OUT = ROOT / "results" / "susceptibility_audit_structured.jsonl"
DEFAULT_FAILURES = ROOT / "results" / "susceptibility_audit_structured_failures.jsonl"
DEFAULT_MANIFEST = ROOT / "results" / "susceptibility_audit_structured_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_or_validate_manifest(path: Path, base: Path) -> None:
    tracked = {
        "amendment": ROOT
        / "experiments"
        / "susceptibility_audit_structured_amendment.md",
        "structured_runner": ROOT / "src" / "susceptibility_audit_structured.py",
        "queue": Path(__file__).resolve(),
        "original_runner": ROOT / "src" / "susceptibility_audit.py",
        "analysis": ROOT / "src" / "susceptibility_audit_analyze.py",
        "base_snapshot": base,
    }
    design = {
        "job_count": 700,
        "n_per_cell": 20,
        "decision_count": 4200,
        "uniform_structured_decoding": True,
        "base_record_count": 629,
        "hashes": {name: sha256(value) for name, value in tracked.items()},
    }
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["design"] != design:
            raise RuntimeError("structured manifest differs from frozen design")
        return
    path.write_text(
        json.dumps({"created_unix": time.time(), "design": design}, indent=2) + "\n"
    )


def append_failure(path: Path, key: tuple[str, str, int], error: str) -> None:
    with path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "model": key[0],
                    "condition": key[1],
                    "seed": key[2],
                    "error": error[-4000:],
                    "timestamp": time.time(),
                }
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--failure-log", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(20)
    write_or_validate_manifest(args.manifest, args.base)
    base_records = load_base_records(args.base)
    if len(base_records) != 629:
        raise RuntimeError(f"expected 629 base records, found {len(base_records)}")
    done = load_done(args.out)
    pending = [job for job in jobs if job.key not in done]
    print(
        f"[structured] target=700 done={len(done)} pending={len(pending)} "
        f"base={len(base_records)}",
        flush=True,
    )
    if args.dry_run:
        return

    pool = load_pool(MATCHED_POOL, topic="remote_work")
    validate_matched_pool(pool)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    failures = 0
    for index, job in enumerate(pending, start=1):
        ok = False
        error = ""
        job_started = time.time()
        for attempt in range(3):
            try:
                record = run_structured_job(
                    job.model,
                    job.condition,
                    job.seed,
                    pool=pool,
                    base_record=base_records.get(job.key),
                )
                with args.out.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                ok = True
                break
            except Exception as caught:
                error = repr(caught)
                if attempt < 2:
                    time.sleep(2**attempt)
        if not ok:
            failures += 1
            append_failure(args.failure_log, job.key, error)
        mean = (time.time() - started) / index
        eta = mean * (len(pending) - index)
        print(
            f"[structured {index}/{len(pending)}] {'ok' if ok else 'FAILED'} "
            f"{job.key} job={time.time() - job_started:.0f}s eta={eta / 3600:.1f}h",
            flush=True,
        )

    final_done = load_done(args.out)
    print(
        f"[structured] finished retained={len(final_done)}/700 failures={failures}",
        flush=True,
    )
    if len(final_done) != 700:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

