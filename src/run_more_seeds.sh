#!/usr/bin/env bash
# Generate more seeds for proper LORO evaluation.
# 2 models × 2 topics × 3 policies × 8 seeds × 10 turns.
# Existing data has seeds 0,1 — add seeds 2..7 to make 8 total.
set -e
cd "$(dirname "$0")/.."

OUT=activations_cuda
POL="random recency engagement_max"
SEEDS_NEW="2 3 4 5 6 7"
TURNS=10
TOPICS="remote_work ai_regulation"

echo "[seeds] Qwen2.5-0.5B-Instruct adding seeds 2..7"
python3 src/agent_loop.py --device cuda --model Qwen/Qwen2.5-0.5B-Instruct \
  --topics $TOPICS --policies $POL --seeds $SEEDS_NEW --n-turns $TURNS \
  --pool posts/pool.jsonl --out-root $OUT 2>&1 | tee -a results/more_seeds.log

echo "[seeds] SmolLM2-1.7B-Instruct adding seeds 2..7"
python3 src/agent_loop.py --device cuda --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --topics $TOPICS --policies $POL --seeds $SEEDS_NEW --n-turns $TURNS \
  --pool posts/pool.jsonl --out-root $OUT 2>&1 | tee -a results/more_seeds.log

echo "[done] more-seeds runs complete"
