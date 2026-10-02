"""Frozen, append-only queue for the susceptibility-audit experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from susceptibility_audit import CONDITIONS


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "src" / "susceptibility_audit.py"
DEFAULT_OUT = ROOT / "results" / "susceptibility_audit.jsonl"
DEFAULT_FAILURES = ROOT / "results" / "susceptibility_audit_failures.jsonl"
DEFAULT_MANIFEST = ROOT / "results" / "susceptibility_audit_manifest.json"
HOLDOUT_MODELS = (
    "lfm2.5:8b-a1b-q4_K_M",
    "granite4:7b-a1b-h",
    "MichelRosselli/apertus:8b-instruct-2509-q4_k_m",
)
DEVELOPMENT_MODELS = (
    "glm-4.7-flash:q4_K_M",
    "gpt-oss:20b",
    "olmo-3:7b-instruct-q4_K_M",
    "nemotron-3-nano:4b",
)
MODELS = HOLDOUT_MODELS + DEVELOPMENT_MODELS


@dataclass(frozen=True)
class Job:
    model: str
    condition: str
    seed: int

    @property
    def key(self) -> tuple[str, str, int]:
        return self.model, self.condition, self.seed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except FileNotFoundError:
        return f"unavailable: {command[0]}"


def build_jobs(n: int) -> list[Job]:
    return [
        Job(model, condition, seed)
        for model in MODELS
        for seed in range(n)
        for condition in CONDITIONS
    ]


def load_done(path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not path.exists():
        return done
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (row["model"], row["condition"], int(row["seed"]))
            except (json.JSONDecodeError, KeyError, ValueError) as error:
                raise ValueError(f"invalid record at {path}:{line_number}") from error
            if key in done:
                raise ValueError(f"duplicate record at {path}:{line_number}: {key}")
            done.add(key)
    return done


def write_or_validate_manifest(path: Path, n: int, jobs: list[Job]) -> None:
    tracked = {
        "preregistration": ROOT
        / "experiments"
        / "susceptibility_audit_preregistration.md",
        "runner": RUNNER,
        "queue": Path(__file__).resolve(),
        "analysis": ROOT / "src" / "susceptibility_audit_analyze.py",
        "compatibility_runtime": ROOT / "src" / "current_models_controls.py",
        "matched_pool": ROOT / "posts" / "pool_gemma.jsonl",
    }
    design = {
        "n_per_cell": n,
        "job_count": len(jobs),
        "holdout_models": list(HOLDOUT_MODELS),
        "development_models": list(DEVELOPMENT_MODELS),
        "conditions": list(CONDITIONS),
        "decoding": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_ctx": 16384,
            "reaction_token_ceiling": 360,
            "decision_token_ceiling": 1024,
        },
        "hashes": {name: sha256(file_path) for name, file_path in tracked.items()},
    }
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("design") != design:
            raise RuntimeError(f"frozen manifest differs from current design: {path}")
        return
    manifest = {
        "created_unix": time.time(),
        "excluded_smoke_seeds": list(range(990, 1000)),
        "design": design,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit_at_launch": command_output(["git", "rev-parse", "HEAD"]),
            "ollama_version": command_output(["ollama", "--version"]),
            "ollama_models_at_launch": command_output(["ollama", "list"]),
            "gpu": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ]
            ),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def append_failure(path: Path, job: Job, error: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model": job.model,
        "condition": job.condition,
        "seed": job.seed,
        "error": error[-4000:],
        "timestamp": time.time(),
    }
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def run_job(job: Job, out: Path, retries: int) -> tuple[bool, str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--model",
        job.model,
        "--condition",
        job.condition,
        "--seed",
        str(job.seed),
        "--out",
        str(out),
    ]
    last_error = ""
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            return True, ""
        last_error = result.stderr.strip() or f"runner exited {result.returncode}"
        if attempt < retries:
            time.sleep(2 ** (attempt - 1))
    return False, last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--failure-log", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.n != 20:
        parser.error("the frozen susceptibility audit requires --n 20")

    jobs = build_jobs(args.n)
    write_or_validate_manifest(args.manifest, args.n, jobs)
    done = load_done(args.out)
    expected = {job.key for job in jobs}
    if done - expected:
        raise RuntimeError("output contains records outside the frozen design")
    pending = [job for job in jobs if job.key not in done]
    print(f"[audit] target={len(jobs)} done={len(done)} pending={len(pending)}", flush=True)
    if args.dry_run:
        for job in pending[:20]:
            print(job.key)
        return

    started = time.time()
    failures = 0
    for index, job in enumerate(pending, 1):
        job_started = time.time()
        ok, error = run_job(job, args.out, args.retries)
        if not ok:
            failures += 1
            append_failure(args.failure_log, job, error)
        mean = (time.time() - started) / index
        eta = mean * (len(pending) - index)
        print(
            f"[audit {index}/{len(pending)}] {'ok' if ok else 'FAILED'} "
            f"{job.key} job={time.time() - job_started:.0f}s eta={eta / 3600:.1f}h",
            flush=True,
        )

    final_done = load_done(args.out)
    print(
        f"[audit] finished retained={len(final_done & expected)}/{len(jobs)} "
        f"failures={failures} elapsed={(time.time() - started) / 3600:.2f}h",
        flush=True,
    )
    if final_done & expected != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
