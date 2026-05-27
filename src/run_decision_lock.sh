#!/usr/bin/env bash
# Decision-shift LOCK: n=40, 6 policies (3 original + 3 mitigation), 2 strongest cells.
# Uses the v1 task format (simple A/B/C) which was shown to produce real variation.
# Existing seeds 0-19 of v1 already done for original 3 policies. Adding 20-39
# for originals + full 0-39 for mitigations.
set -e
cd "$(dirname "$0")/.."

OUT=results/decision_shift_lock.jsonl
rm -f "$OUT"

ALL_POL="random recency engagement_max shuffled balanced disclosed"
SEEDS_ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39"
TURNS=10

# Re-run everything fresh under the v1 prompt with all 6 policies + n=40.
echo "[lock] SmolLM2-1.7B-Instruct / remote_work"
python3 src/decision_shift.py --device cuda --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --topics remote_work --policies $ALL_POL --seeds $SEEDS_ALL --n-turns $TURNS \
  --pool posts/pool.jsonl --out "$OUT" 2>&1 | tee -a results/decision_lock.log

echo "[lock] Qwen2.5-0.5B-Instruct / ai_regulation"
python3 src/decision_shift.py --device cuda --model Qwen/Qwen2.5-0.5B-Instruct \
  --topics ai_regulation --policies $ALL_POL --seeds $SEEDS_ALL --n-turns $TURNS \
  --pool posts/pool.jsonl --out "$OUT" 2>&1 | tee -a results/decision_lock.log

echo "[done] decision-shift lock complete"
