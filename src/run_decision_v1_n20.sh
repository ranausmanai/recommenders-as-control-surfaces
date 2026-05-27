#!/usr/bin/env bash
# Re-run the ORIGINAL (v1) decision task with n=20 seeds on susceptible cells.
# Same simple prompt as v1, just more data, to test if the n=8 effect replicates.
set -e
cd "$(dirname "$0")/.."

OUT=results/decision_shift_v1_n20.jsonl
rm -f "$OUT"

POL="random recency engagement_max"
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
TURNS=10

# Susceptible cells per AI's recommendation
echo "[v1-n20] Qwen2.5-0.5B-Instruct  ai_regulation"
python3 src/decision_shift.py --device cuda --model Qwen/Qwen2.5-0.5B-Instruct \
  --topics ai_regulation --policies $POL --seeds $SEEDS --n-turns $TURNS \
  --pool posts/pool.jsonl --out "$OUT" 2>&1 | tee -a results/decision_v1_n20.log

echo "[v1-n20] SmolLM2-1.7B-Instruct  ai_regulation remote_work"
python3 src/decision_shift.py --device cuda --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --topics ai_regulation remote_work --policies $POL --seeds $SEEDS --n-turns $TURNS \
  --pool posts/pool.jsonl --out "$OUT" 2>&1 | tee -a results/decision_v1_n20.log

echo "[done] v1 n=20 complete"
