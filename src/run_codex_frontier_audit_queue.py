"""Frozen resumable queue for Codex frontier-agent evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from codex_frontier_audit import CODEX_BIN, CONDITIONS, run_job
from feed_policies import load_pool
from reviewer_controls import MATCHED_POOL, validate_matched_pool


ROOT = Path(__file__).resolve().parent.parent
MODELS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
DEFAULT_OUT = ROOT / "results" / "codex_frontier_audit.jsonl"
DEFAULT_FAILURES = ROOT / "results" / "codex_frontier_audit_failures.jsonl"
DEFAULT_MANIFEST = ROOT / "results" / "codex_frontier_audit_manifest.json"


@dataclass(frozen=True)
class Job:
    model: str
    condition: str
    replicate: int

    @property
    def key(self) -> tuple[str, str, int]:
        return self.model, self.condition, self.replicate


def jobs() -> list[Job]:
    return [
        Job(model, condition, replicate)
        for model in MODELS
        for replicate in range(20)
        for condition in CONDITIONS
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_done(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    keys = {
        (row["model"], row["condition"], int(row["replicate"])) for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("duplicate frontier records")
    return keys


def manifest(path: Path) -> None:
    tracked = {
        "preregistration": ROOT
        / "experiments"
        / "codex_frontier_audit_preregistration.md",
        "schema": ROOT / "experiments" / "codex_frontier_choices.schema.json",
        "runner": ROOT / "src" / "codex_frontier_audit.py",
        "queue": Path(__file__).resolve(),
        "analysis": ROOT / "src" / "codex_frontier_audit_analyze.py",
        "matched_pool": MATCHED_POOL,
    }
    design = {
        "models": list(MODELS),
        "conditions": list(CONDITIONS),
        "replicates": 20,
        "calls": 300,
        "decision_labels": 1800,
        "codex_version": subprocess.run(
            [str(CODEX_BIN), "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "codex_binary": str(CODEX_BIN),
        "hashes": {name: sha256(value) for name, value in tracked.items()},
    }
    if path.exists():
        if json.loads(path.read_text())["design"] != design:
            raise RuntimeError("frontier manifest differs from frozen design")
        return
    path.write_text(
        json.dumps({"created_unix": time.time(), "design": design}, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest(args.manifest)
    all_jobs = jobs()
    done = load_done(args.out)
    pending = [job for job in all_jobs if job.key not in done]
    print(f"[codex-frontier] done={len(done)}/300 pending={len(pending)}", flush=True)
    if args.dry_run:
        return
    pool = load_pool(MATCHED_POOL, topic="remote_work")
    validate_matched_pool(pool)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for index, job in enumerate(pending, start=1):
        ok = False
        error = ""
        for attempt in range(3):
            try:
                row = run_job(
                    job.model,
                    job.condition,
                    job.replicate,
                    pool=pool,
                )
                with args.out.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                ok = True
                break
            except Exception as caught:
                error = repr(caught)
                if attempt < 2:
                    time.sleep(30 * (attempt + 1))
        if not ok:
            with args.failures.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "model": job.model,
                            "condition": job.condition,
                            "replicate": job.replicate,
                            "error": error,
                            "timestamp": time.time(),
                        }
                    )
                    + "\n"
                )
        mean = (time.time() - started) / index
        eta = mean * (len(pending) - index)
        print(
            f"[frontier {index}/{len(pending)}] {'ok' if ok else 'FAILED'} "
            f"{job.key} eta={eta / 3600:.1f}h",
            flush=True,
        )
    final = load_done(args.out)
    print(f"[codex-frontier] finished={len(final)}/300", flush=True)
    if len(final) != 300:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
