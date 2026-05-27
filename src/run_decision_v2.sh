#!/usr/bin/env bash
# Decision-shift v2: continuous task + mitigations, only on susceptible cells.
set -e
cd "$(dirname "$0")/.."

OUT=results/decision_shift_v2.jsonl
rm -f "$OUT"

POL="random recency engagement_max shuffled balanced disclosed"
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"
TURNS=10

# Susceptible cell 1: Qwen2.5-0.5B / ai_regulation (already p<0.01)
echo "[v2] Qwen2.5-0.5B-Instruct  ai_regulation"
python3 src/decision_shift_v2.py --device cuda --model Qwen/Qwen2.5-0.5B-Instruct \
  --topics ai_regulation --policies $POL --seeds $SEEDS --n-turns $TURNS \
  --pool posts/pool.jsonl --out "$OUT" 2>&1 | tee -a results/decision_v2.log

# Susceptible cell 2 & 3: SmolLM2-1.7B / both topics
echo "[v2] SmolLM2-1.7B-Instruct  ai_regulation + remote_work"
python3 src/decision_shift_v2.py --device cuda --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --topics ai_regulation remote_work --policies $POL --seeds $SEEDS --n-turns $TURNS \
  --pool posts/pool.jsonl --out "$OUT" 2>&1 | tee -a results/decision_v2.log

echo "[done] decision-shift v2 complete"
