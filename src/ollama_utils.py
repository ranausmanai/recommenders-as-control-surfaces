"""Light Ollama wrappers used for cross-model agent runs (secondary models).

Primary agent uses HF transformers (we need raw hooks). When we run a
secondary agent for the cross-model robustness check, we use Ollama for
text generation only — activations are not captured for Ollama agents.

The goal's cross-model claim is "fingerprint appears in 2+ agents". For
the SECOND agent, we re-load via HF (a different small HF model) so we
still get residual streams. This module exists to enumerate local models
and provide a generation fallback in case the HF route fails.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def list_models() -> list[dict]:
    """Return [{'name', 'size', 'modified'}] from `ollama list`."""
    out = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
    rows = []
    for line in out.stdout.splitlines()[1:]:
        parts = [p for p in line.split() if p]
        if len(parts) >= 3:
            rows.append({"name": parts[0], "size": parts[2], "modified": " ".join(parts[3:])})
    return rows


def generate(model: str, prompt: str, system: str | None = None, options: dict | None = None) -> str:
    """One-shot Ollama generation. Returns the model's full text response."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    if options:
        payload["options"] = options
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        capture_output=True, text=True, timeout=120, check=False,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    for m in list_models():
        print(m)
