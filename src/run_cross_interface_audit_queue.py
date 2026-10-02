"""Frozen append-only queue for the cross-interface audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from cross_interface_audit import CONDITIONS, run_retrieval_job
from feed_policies import load_pool
from reviewer_controls import MATCHED_POOL, validate_matched_pool
from run_susceptibility_audit_queue import MODELS, Job, load_done


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "cross_interface_audit.jsonl"
DEFAULT_FAILURES = ROOT / "results" / "cross_interface_audit_failures.jsonl"
DEFAULT_MANIFEST = ROOT / "results" / "cross_interface_audit_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_jobs() -> list[Job]:
    return [
        Job(model, condition, seed)
        for model in MODELS
        for seed in range(20)
        for condition in CONDITIONS
    ]


def write_or_validate_manifest(path: Path) -> None:
    tracked = {
        "preregistration": ROOT
        / "experiments"
        / "cross_interface_audit_preregistration.md",
        "runner": ROOT / "src" / "cross_interface_audit.py",
        "queue": Path(__file__).resolve(),
        "analysis": ROOT / "src" / "cross_interface_audit_analyze.py",
        "structured_runtime": ROOT / "src" / "susceptibility_audit_structured.py",
        "matched_pool": MATCHED_POOL,
    }
    design = {
        "job_count": 700,
        "decision_count": 4200,
        "models": list(MODELS),
        "conditions": list(CONDITIONS),
        "n_per_cell": 20,
        "hashes": {name: sha256(value) for name, value in tracked.items()},
    }
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["design"] != design:
            raise RuntimeError("cross-interface manifest differs from frozen design")
        return
    path.write_text(
        json.dumps({"created_unix": time.time(), "design": design}, indent=2) + "\n"
    )


def append_failure(path: Path, job: Job, error: str) -> None:
    with path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "model": job.model,
                    "condition": job.condition,
                    "seed": job.seed,
                    "error": error[-4000:],
                    "timestamp": time.time(),
                }
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--failure-log", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs()
    write_or_validate_manifest(args.manifest)
    done = load_done(args.out)
    pending = [job for job in jobs if job.key not in done]
    print(
        f"[cross-interface] target=700 done={len(done)} pending={len(pending)}",
        flush=True,
    )
    if args.dry_run:
        return

    pool = load_pool(MATCHED_POOL, topic="remote_work")
    validate_matched_pool(pool)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    started = time.time()
    for index, job in enumerate(pending, start=1):
        ok = False
        error = ""
        job_started = time.time()
        for attempt in range(3):
            try:
                record = run_retrieval_job(
                    job.model,
                    job.condition,
                    job.seed,
                    pool=pool,
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
            append_failure(args.failure_log, job, error)
        mean = (time.time() - started) / index
        eta = mean * (len(pending) - index)
        print(
            f"[cross {index}/{len(pending)}] {'ok' if ok else 'FAILED'} "
            f"{job.key} job={time.time() - job_started:.0f}s eta={eta / 3600:.1f}h",
            flush=True,
        )

    final_done = load_done(args.out)
    print(
        f"[cross-interface] finished retained={len(final_done)}/700 "
        f"failures={failures}",
        flush=True,
    )
    if len(final_done) != 700:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

