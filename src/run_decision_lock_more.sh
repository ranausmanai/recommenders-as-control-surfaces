#!/usr/bin/env bash
# Decision-shift LOCK round 2: extend n=40 to additional susceptible cells.
# Goal: turn the headline from "one model" into "multiple models/topics".
set -e
cd "$(dirname "$0")/.."

OUT=results/decision_shift_lock.jsonl  # append to existing
ALL_POL="random recency engagement_max shuffled balanced disclosed"
SEEDS_ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39"
TURNS=10

echo "[lock+] SmolLM2-1.7B-Instruct / ai_regulation"
python3 src/decision_shift.py --device cuda --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --topics ai_regulation --policies $ALL_POL --seeds $SEEDS_ALL --n-turns $TURNS \
  --pool posts/pool.jsonl --out "$OUT" 2>&1 | tee -a results/decision_lock_more.log

echo "[lock+] Qwen2.5-3B-Instruct / ai_regulation"
python3 src/decision_shift.py --device cuda --model Qwen/Qwen2.5-3B-Instruct \
  --topics ai_regulation --policies $ALL_POL --seeds $SEEDS_ALL --n-turns $TURNS \
  --pool posts/pool.jsonl --out "$OUT" 2>&1 | tee -a results/decision_lock_more.log

echo "[lock+] Yi-1.5-6B-Chat / remote_work (4-bit)"
python3 src/decision_shift.py --device cuda --load-in-4bit --model 01-ai/Yi-1.5-6B-Chat \
  --topics remote_work --policies $ALL_POL --seeds $SEEDS_ALL --n-turns $TURNS \
  --pool posts/pool.jsonl --out "$OUT" 2>&1 | tee -a results/decision_lock_more.log

echo "[done] decision-lock round 2 complete"
