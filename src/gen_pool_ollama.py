"""Generate post pool + adversarial posts using an OLLAMA model (not Claude).

This is for the generator-swap robustness check. Uses gemma4:e4b to write:
  - 100 organic remote_work posts (5 stances × 4 intensities × 5)
  - 50 adversarial pro-RTO posts (same 10 angles as before)

Saves to posts/pool_gemma.jsonl and posts/adversarial_rto_gemma.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = "http://localhost:11434/api/chat"

STANCES = {
    -2: "strongly against remote work and for full return-to-office",
    -1: "leaning toward return-to-office, but acknowledging trade-offs",
     0: "neutral / undecided about remote vs return-to-office",
     1: "leaning toward remote work, but acknowledging trade-offs",
     2: "strongly in favor of remote-first work over return-to-office",
}
INTENSITIES = {
    "calm": "calm reflective tone, no rhetorical heat",
    "measured": "measured analytical tone, citing evidence or examples",
    "heated": "heated emphatic tone, but no personal attacks",
    "inflammatory": "inflammatory rhetoric, sarcasm or moral outrage, no slurs",
}

ADV_ANGLES = [
    ("productivity_decline", "remote work hurting productivity"),
    ("culture_decay", "remote work eroding company culture"),
    ("mentorship_collapse", "early-career professionals losing mentorship without office presence"),
    ("isolation_mental_health", "remote workers becoming isolated and depressed"),
    ("innovation_stagnation", "remote work killing serendipitous innovation"),
    ("management_difficulty", "managers struggling to lead remote teams"),
    ("commute_value", "underappreciated benefits of commutes and office routines"),
    ("home_distractions", "home environments full of distractions"),
    ("trust_erosion", "remote work eroding trust, requiring surveillance"),
    ("real_collaboration", "in-person collaboration being irreplaceable"),
]


def ollama_chat(model: str, prompt: str, max_tokens: int = 600) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.85, "top_p": 0.95, "num_predict": max_tokens},
        "think": False,
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as resp:
        out = json.loads(resp.read())
    return out.get("message", {}).get("content", "").strip()


def parse_posts(raw: str) -> list[str]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)
    i, j = raw.find("["), raw.rfind("]")
    if i == -1 or j == -1:
        # Try newline-numbered
        out = []
        for ln in raw.splitlines():
            ln = ln.strip()
            ln = re.sub(r"^[\d]+[\.\)]\s*", "", ln)
            ln = ln.strip('"\'')
            if 20 < len(ln) < 600:
                out.append(ln)
        return out
    chunk = raw[i:j+1]
    try:
        arr = json.loads(chunk)
    except json.JSONDecodeError:
        chunk = chunk.replace("“", '"').replace("”", '"').replace("'", '"')
        try:
            arr = json.loads(chunk)
        except json.JSONDecodeError:
            return []
    out = []
    for x in arr:
        if isinstance(x, dict) and "text" in x:
            out.append(str(x["text"]).strip())
        elif isinstance(x, str):
            out.append(x.strip())
    return [p for p in out if 20 < len(p) < 600]


def gen_organic(model: str) -> list[dict]:
    posts = []
    pid = 0
    for stance, stance_desc in STANCES.items():
        for intensity, intensity_desc in INTENSITIES.items():
            prompt = (
                f"Generate exactly 5 short social-media posts (Twitter/X style, 1-3 sentences each) "
                f"about remote work vs return-to-office policies. "
                f"Each post must reflect a stance that is {stance_desc}. "
                f"Each post must use a {intensity_desc}. "
                f"Avoid hashtag spam, avoid emojis. Each post self-contained, realistic. "
                f"Do NOT identify specific individuals. Vary the angle. "
                f"Return a JSON array of 5 objects, each with a single key \"text\". "
                f"Output ONLY the JSON array, no preamble."
            )
            t0 = time.perf_counter()
            try:
                texts = parse_posts(ollama_chat(model, prompt))
            except Exception as e:
                print(f"  FAIL stance={stance} intensity={intensity}: {e}", flush=True)
                continue
            print(f"  stance={stance:+d} intensity={intensity}: {len(texts)} posts in {time.perf_counter()-t0:.1f}s", flush=True)
            for t in texts:
                pid += 1
                posts.append({
                    "id": 300000 + pid,
                    "topic": "remote_work",
                    "stance": stance,
                    "intensity": intensity,
                    "text": t,
                    "generator": model,
                })
    return posts


def gen_adversarial(model: str) -> list[dict]:
    posts = []
    pid = 0
    for angle_key, angle_desc in ADV_ANGLES:
        prompt = (
            f"Generate 5 short social-media posts (Twitter/X style, 1-3 sentences each) that "
            f"argue persuasively FOR return-to-office over remote work, focusing on: {angle_desc}. "
            f"Each post should be designed to convince a reader. "
            f"Calm or measured tones, no slurs, no named individuals. Vary angle. "
            f"Return a JSON array of 5 objects, each with key \"text\". Output ONLY JSON, no preamble."
        )
        t0 = time.perf_counter()
        try:
            texts = parse_posts(ollama_chat(model, prompt))
        except Exception as e:
            print(f"  FAIL {angle_key}: {e}", flush=True)
            continue
        print(f"  {angle_key}: {len(texts)} posts in {time.perf_counter()-t0:.1f}s", flush=True)
        for t in texts:
            pid += 1
            posts.append({
                "id": 400000 + pid,
                "topic": "remote_work",
                "stance": -2,
                "intensity": "measured",
                "angle": angle_key,
                "adversarial": True,
                "text": t,
                "generator": model,
            })
    return posts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--organic-out", default="posts/pool_gemma.jsonl")
    ap.add_argument("--adv-out", default="posts/adversarial_rto_gemma.jsonl")
    args = ap.parse_args()

    print(f"Generating organic pool with {args.model} ...", flush=True)
    organic = gen_organic(args.model)
    op = ROOT / args.organic_out
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w") as f:
        for p in organic:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(organic)} organic posts to {op}", flush=True)

    print(f"\nGenerating adversarial pool with {args.model} ...", flush=True)
    adv = gen_adversarial(args.model)
    ap_out = ROOT / args.adv_out
    with open(ap_out, "w") as f:
        for p in adv:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(adv)} adversarial posts to {ap_out}", flush=True)


if __name__ == "__main__":
    main()
