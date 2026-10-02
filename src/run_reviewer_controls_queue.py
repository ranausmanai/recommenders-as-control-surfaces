"""Priority-ordered, append-only queue for reviewer-control experiments."""
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


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "src" / "reviewer_controls.py"
DEFAULT_OUT = ROOT / "results" / "reviewer_controls.jsonl"
DEFAULT_LOG = ROOT / "results" / "reviewer_controls_failures.jsonl"
DEFAULT_MANIFEST = ROOT / "results" / "reviewer_controls_manifest.json"
MODELS = ("llama3.2:3b", "gemma4:e4b", "qwen3.5:2b", "qwen3.5:9b")
CORE_MODELS = ("llama3.2:3b", "gemma4:e4b")
TOPICS = (
    "remote_work",
    "ai_regulation",
    "ubi",
    "deploy_security",
    "vendor_security",
    "access_policy",
)
SELECTION_CONDITIONS = ("matched_balanced", "matched_rto", "matched_remote")
ORDER_CONDITIONS = ("order_interleaved", "order_rto_last", "order_remote_last")


@dataclass(frozen=True)
class Job:
    model: str
    topic: str
    condition: str
    seed: int
    max_history_turns: int = 10

    @property
    def key(self) -> tuple[str, str, str, int]:
        return self.model, self.topic, self.condition, self.seed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_jobs(n: int) -> list[Job]:
    jobs: list[Job] = []

    # First answer the headline default/ceiling concern for all modern models.
    for model in MODELS:
        for seed in range(n):
            jobs.append(Job(model, "remote_work", "no_feed", seed))

    # Then run the make-or-break matched and order-only controls, Llama first.
    for model in CORE_MODELS:
        for condition in SELECTION_CONDITIONS:
            for seed in range(n):
                jobs.append(Job(model, "remote_work", condition, seed))
        for condition in ORDER_CONDITIONS:
            for seed in range(n):
                jobs.append(Job(model, "remote_work", condition, seed))

    # Finally measure defaults for every other task used in the paper.
    for model in MODELS:
        for topic in TOPICS[1:]:
            for seed in range(n):
                jobs.append(Job(model, topic, "no_feed", seed))
    return jobs


def write_or_validate_manifest(path: Path, n: int, jobs: list[Job]) -> None:
    tracked = {
        "preregistration": ROOT / "experiments" / "reviewer_controls_preregistration.md",
        "runner": ROOT / "src" / "reviewer_controls.py",
        "queue": ROOT / "src" / "run_reviewer_controls_queue.py",
        "analysis": ROOT / "src" / "reviewer_controls_analyze.py",
        "matched_pool": ROOT / "posts" / "pool_gemma.jsonl",
    }
    design = {
        "n_per_cell": n,
        "job_count": len(jobs),
        "models": list(MODELS),
        "core_models": list(CORE_MODELS),
        "topics": list(TOPICS),
        "selection_conditions": list(SELECTION_CONDITIONS),
        "order_conditions": list(ORDER_CONDITIONS),
        "hashes": {name: sha256(file_path) for name, file_path in tracked.items()},
    }
    if path.exists():
        existing = json.loads(path.read_text())
        existing_design = existing.get("design")
        if existing_design != design:
            raise RuntimeError(
                f"existing manifest design differs from current code/data: {path}"
            )
        return

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ollama_version = subprocess.run(
        ["ollama", "--version"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "created_unix": time.time(),
        "design": design,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit_at_launch": git_commit,
            "ollama_version": ollama_version,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def load_done(path: Path) -> set[tuple[str, str, str, int]]:
    done: set[tuple[str, str, str, int]] = set()
    if not path.exists():
        return done
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                done.add(
                    (
                        record["model"],
                        record["topic"],
                        record["condition"],
                        int(record["seed"]),
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError) as error:
                raise ValueError(f"invalid record at {path}:{line_number}: {error}") from error
    return done


def append_failure(path: Path, job: Job, attempts: int, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model": job.model,
        "topic": job.topic,
        "condition": job.condition,
        "seed": job.seed,
        "attempts": attempts,
        "error": message,
        "timestamp": time.time(),
    }
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def run_job(job: Job, out: Path, retries: int) -> bool:
    command = [
        sys.executable,
        str(RUNNER),
        "--model",
        job.model,
        "--topic",
        job.topic,
        "--condition",
        job.condition,
        "--seed",
        str(job.seed),
        "--max-history-turns",
        str(job.max_history_turns),
        "--out",
        str(out),
    ]
    for attempt in range(1, retries + 1):
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode == 0:
            return True
        print(
            f"[queue] failed attempt {attempt}/{retries}: {job.key}",
            file=sys.stderr,
            flush=True,
        )
        if attempt < retries:
            time.sleep(2 ** (attempt - 1))
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--failure-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run at most this many pending jobs; useful for smoke tests",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.n <= 0:
        parser.error("--n must be positive")
    jobs = build_jobs(args.n)
    write_or_validate_manifest(args.manifest, args.n, jobs)
    done = load_done(args.out)
    pending = [job for job in jobs if job.key not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        f"[queue] target={len(jobs)} done={len(done & {job.key for job in jobs})} "
        f"pending_now={len(pending)} output={args.out}",
        flush=True,
    )
    if args.dry_run:
        for job in pending[:20]:
            print(f"  {job.key}")
        return

    started = time.time()
    successes = 0
    failures = 0
    for index, job in enumerate(pending, 1):
        job_started = time.time()
        ok = run_job(job, args.out, args.retries)
        elapsed = time.time() - job_started
        if ok:
            successes += 1
        else:
            failures += 1
            append_failure(args.failure_log, job, args.retries, "runner failed")
        total_elapsed = time.time() - started
        mean = total_elapsed / index
        eta = mean * (len(pending) - index)
        print(
            f"[queue {index}/{len(pending)}] {'ok' if ok else 'FAILED'} "
            f"{job.key} job={elapsed:.0f}s eta={eta / 3600:.1f}h",
            flush=True,
        )

    print(
        f"[queue] complete successes={successes} failures={failures} "
        f"elapsed={(time.time() - started) / 3600:.1f}h",
        flush=True,
    )


if __name__ == "__main__":
    main()
