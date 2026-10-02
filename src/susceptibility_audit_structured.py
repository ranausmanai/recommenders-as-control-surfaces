"""Uniform structured-decision amendment for the susceptibility audit."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from agent_loop import PERSONA, build_reaction_prompt
from current_models_controls import current_ollama_chat
from decision_shift import parse_choice
from reviewer_controls import OLLAMA_CONTEXT_TOKENS, OLLAMA_URL, summarize_exposure
from susceptibility_audit import TASKS, decision_messages, exposure_for


DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendation": {
            "type": "string",
            "enum": ["A", "B", "C"],
        }
    },
    "required": ["recommendation"],
    "additionalProperties": False,
}
TEXT_SCHEMA_REQUEST = (
    "Return exactly one answer using this schema and no other text: "
    "Recommendation: <A|B|C>"
)
JSON_SCHEMA_REQUEST = (
    'Return exactly one JSON object and no other text: '
    '{"recommendation":"<A|B|C>"}'
)


def structured_messages(
    history: list[dict[str, Any]], question: str
) -> list[dict[str, str]]:
    messages = decision_messages(history, question)
    content = messages[-1]["content"]
    if TEXT_SCHEMA_REQUEST not in content:
        raise ValueError("decision prompt schema request was not found")
    messages[-1] = {
        "role": "user",
        "content": content.replace(TEXT_SCHEMA_REQUEST, JSON_SCHEMA_REQUEST),
    }
    return messages


def structured_ollama_chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    seed: int,
    max_new_tokens: int = 256,
    retries: int = 3,
) -> tuple[str, dict[str, Any]]:
    think_mode: bool | str = "low" if model == "gpt-oss:20b" else False
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "format": DECISION_SCHEMA,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": max_new_tokens,
                "seed": seed,
                "num_ctx": OLLAMA_CONTEXT_TOKENS,
            },
            "think": think_mode,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=360) as response:
                result = json.loads(response.read())
            raw = result.get("message", {}).get("content", "")
            parsed = json.loads(raw)
            choice = parsed.get("recommendation")
            if choice not in {"A", "B", "C"}:
                choice = parse_choice(raw)
            if choice not in {"A", "B", "C"}:
                raise ValueError(f"invalid structured recommendation: {raw[:200]!r}")
            meta = {
                "total_duration_ns": result.get("total_duration"),
                "load_duration_ns": result.get("load_duration"),
                "prompt_eval_count": result.get("prompt_eval_count"),
                "eval_count": result.get("eval_count"),
                "eval_duration_ns": result.get("eval_duration"),
                "think_mode": think_mode,
                "structured_schema": True,
            }
            return choice, {"raw": raw, "meta": meta}
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"structured Ollama call failed: {last_error}")


def generate_history(
    model: str,
    condition: str,
    seed: int,
    *,
    pool: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    history: list[dict[str, Any]] = []
    batches = [] if condition == "no_feed" else exposure_for(pool, condition, seed)
    for turn, posts in enumerate(batches):
        messages: list[dict[str, str]] = [{"role": "system", "content": PERSONA}]
        for item in history[-10:]:
            messages.extend(
                [
                    {"role": "user", "content": item["reaction_prompt"]},
                    {"role": "assistant", "content": item["reaction_raw"]},
                ]
            )
        prompt = build_reaction_prompt(posts)
        raw, meta = current_ollama_chat(
            model,
            messages + [{"role": "user", "content": prompt}],
            seed=seed * 10000 + turn + 1,
            max_new_tokens=360,
        )
        history.append(
            {
                "turn": turn,
                "post_ids": [post["id"] for post in posts],
                "posts": posts,
                "reaction_prompt": prompt,
                "reaction_raw": raw,
                "ollama_meta": meta,
            }
        )
    return history, batches


def run_structured_job(
    model: str,
    condition: str,
    seed: int,
    *,
    pool: list[dict[str, Any]],
    base_record: dict[str, Any] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if base_record is None:
        history, batches = generate_history(
            model, condition, seed, pool=pool
        )
        history_source = "regenerated_missing"
    else:
        expected = (model, condition, seed)
        observed = (
            base_record["model"],
            base_record["condition"],
            int(base_record["seed"]),
        )
        if observed != expected:
            raise ValueError(f"base-record key mismatch: {observed} != {expected}")
        history = base_record["turn_trace"]
        batches = []
        history_source = "frozen_incomplete_629"

    choices: dict[str, str] = {}
    raw: dict[str, str] = {}
    decision_meta: dict[str, Any] = {}
    for task_index, (task, question) in enumerate(TASKS.items(), start=1):
        choice, payload = structured_ollama_chat(
            model,
            structured_messages(history, question),
            seed=seed * 10000 + 9000 + task_index,
        )
        choices[task] = choice
        raw[task] = payload["raw"]
        decision_meta[task] = payload["meta"]

    record = {
        "experiment": "susceptibility_audit_structured",
        "model": model,
        "topic": "remote_work",
        "condition": condition,
        "seed": seed,
        "choices": choices,
        "raw": raw,
        "n_turns": len(history),
        "turn_trace": history,
        "trajectory_source": history_source,
        "decision_meta": decision_meta,
        "elapsed_s": time.perf_counter() - started,
    }
    if base_record is not None and "exposure" in base_record:
        record["exposure"] = base_record["exposure"]
    elif batches:
        record["exposure"] = summarize_exposure(batches)
    return record


def load_base_records(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["model"], row["condition"], int(row["seed"]))
            if key in rows:
                raise ValueError(f"duplicate base record at line {line_number}: {key}")
            rows[key] = row
    return rows

