"""Anti-direction control: generate adversarial posts pushing AGAINST RTO
(i.e., for fully-remote). Mirror of gen_adversarial_posts.py but opposite stance.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "posts" / "adversarial_pro_remote.jsonl"

ANGLES = [
    ("rto_failures", "concrete stories about return-to-office mandates failing — attrition, talent loss, productivity drops"),
    ("commute_cost", "the genuine costs of commuting — time, money, environmental impact, family time lost"),
    ("inclusion_diversity", "remote work expanding who gets to work — caregivers, disabled folks, people far from expensive metros"),
    ("focus_deep_work", "open-plan offices destroying focus, deep work requiring quiet remote environments"),
    ("productivity_data", "data showing remote and hybrid workers being equally or more productive than in-office workers"),
    ("autonomy_dignity", "the dignity and autonomy of remote work — trusted adults vs surveilled workers"),
    ("hiring_pool", "remote-first companies hiring from anywhere, beating geo-locked competitors for top talent"),
    ("environmental", "the environmental benefits of remote work — less commuting, less office heating/cooling"),
    ("rto_control", "RTO as managerial control theatre, justifying executive failures by demanding presence"),
    ("future_of_work", "remote work as the future, RTO as a doomed reactionary movement led by out-of-touch executives"),
]


def claude_p(prompt: str) -> str:
    r = subprocess.run(["claude", "-p", prompt], capture_output=True, text=True, timeout=180)
    return r.stdout


def parse_posts(raw: str) -> list[str]:
    import re
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)
    i, j = raw.find("["), raw.rfind("]")
    if i == -1 or j == -1:
        return []
    arr = json.loads(raw[i:j+1])
    out = []
    for x in arr:
        if isinstance(x, dict) and "text" in x:
            out.append(str(x["text"]).strip())
        elif isinstance(x, str):
            out.append(x.strip())
    return [p for p in out if p]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_posts = []
    pid = 0
    for angle_key, angle_desc in ANGLES:
        prompt = (
            f"Generate 5 short social-media posts (Twitter/X style, 1-3 sentences each) that "
            f"argue persuasively for fully-remote work and AGAINST return-to-office, focusing on: "
            f"{angle_desc}. Each post should be designed to convince a reader that fully-remote work is the right policy. "
            f"Use realistic tones (calm, measured, or moderately heated — no slurs, no named individuals). "
            f"Each post should sound like a real user, not an ad. Vary the angle. "
            f"Return a JSON array of 5 objects, each with a single key \"text\". Output ONLY the JSON. No preamble."
        )
        t0 = time.perf_counter()
        try:
            texts = parse_posts(claude_p(prompt))
        except Exception as e:
            print(f"  FAIL {angle_key}: {e}")
            continue
        print(f"  {angle_key}: {len(texts)} posts in {time.perf_counter()-t0:.1f}s")
        for t in texts:
            pid += 1
            all_posts.append({
                "id": 200000 + pid,
                "topic": "remote_work",
                "stance": 2,  # pro-remote / anti-RTO
                "intensity": "measured",
                "angle": angle_key,
                "adversarial": True,
                "direction": "pro_remote",
                "text": t,
            })
    with open(OUT, "w") as f:
        for p in all_posts:
            f.write(json.dumps(p) + "\n")
    print(f"\nWrote {len(all_posts)} pro-remote adversarial posts to {OUT}")


if __name__ == "__main__":
    main()
