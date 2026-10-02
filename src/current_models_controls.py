"""Compatibility entry point for the frozen current-model replication.

GPT-OSS requires its native low reasoning mode to emit final content within the
fixed token budgets. Other models use non-thinking mode, matching the original
reviewer-control runner. All experiment construction and record schemas remain
owned by reviewer_controls.py.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

import reviewer_controls as controls


def current_ollama_chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    seed: int,
    max_new_tokens: int,
    temperature: float = 0.7,
    retries: int = 3,
) -> tuple[str, dict[str, Any]]:
    think_mode: bool | str = "low" if model == "gpt-oss:20b" else False
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": max_new_tokens,
                "seed": seed,
                "num_ctx": controls.OLLAMA_CONTEXT_TOKENS,
            },
            "think": think_mode,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        controls.OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=360) as response:
                result = json.loads(response.read())
            text = controls._strip_thinking(
                result.get("message", {}).get("content", "")
            )
            meta = {
                "total_duration_ns": result.get("total_duration"),
                "load_duration_ns": result.get("load_duration"),
                "prompt_eval_count": result.get("prompt_eval_count"),
                "eval_count": result.get("eval_count"),
                "eval_duration_ns": result.get("eval_duration"),
                "think_mode": think_mode,
            }
            return text, meta
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Ollama failed after {retries} attempts: {last_error}")


controls.ollama_chat = current_ollama_chat


if __name__ == "__main__":
    controls.main()
