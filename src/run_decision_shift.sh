#!/usr/bin/env bash
# Decision-shift experiment: does feed ranking change a multiple-choice decision?
# 5 models × 2 topics × 3 policies × 8 seeds × 10-turn feed exposure + 1 decision Q.
set -e
cd "$(dirname "$0")/.."

OUT=results/decision_shift.jsonl
rm -f "$OUT"

POL="random recency engagement_max"
SEEDS="0 1 2 3 4 5 6 7"
TURNS=10
TOPICS="remote_work ai_regulation"

# Run on 5 diverse models (mostly the ones we already have data for)
for m in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct tiiuae/Falcon3-3B-Instruct 01-ai/Yi-1.5-6B-Chat; do
  echo "[dec] $m"
  EXTRA=""
  if [ "$m" = "01-ai/Yi-1.5-6B-Chat" ]; then
    EXTRA="--load-in-4bit"
  fi
  python3 src/decision_shift.py --device cuda --model "$m" $EXTRA \
    --topics $TOPICS --policies $POL --seeds $SEEDS --n-turns $TURNS \
    --pool posts/pool.jsonl --out "$OUT" 2>&1 | tee -a results/decision_shift.log
done

echo "[done] decision-shift pipeline complete"
echo "results in $OUT"
