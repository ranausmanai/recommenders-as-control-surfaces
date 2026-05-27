#!/usr/bin/env bash
# Diverse cross-family run on RTX 4000 Ada.
# Eight model families, none of them Qwen, on remote_work + ai_regulation.
# 3 policies × 2 seeds × 10 turns = 60 samples per (model, topic).
set -e
cd "$(dirname "$0")/.."

OUT=activations_cuda
POL="random recency engagement_max"
SEEDS="0 1"
TURNS=10
TOPICS="remote_work ai_regulation"

# Small / mid models in bf16
for m in microsoft/phi-2 allenai/OLMo-2-0425-1B-Instruct internlm/internlm2_5-1_8b-chat stabilityai/stablelm-2-1_6b-chat tiiuae/Falcon3-3B-Instruct; do
  echo "[fam] $m"
  python3 src/agent_loop.py --device cuda --model "$m" \
    --topics $TOPICS --policies $POL --seeds $SEEDS --n-turns $TURNS \
    --pool posts/pool.jsonl --out-root $OUT 2>&1 | tee -a results/diverse.log || echo "  >> $m FAILED, continuing"
done

# Larger models in 4-bit
for m in HuggingFaceH4/zephyr-7b-beta 01-ai/Yi-1.5-6B-Chat; do
  echo "[fam-4bit] $m"
  python3 src/agent_loop.py --device cuda --load-in-4bit --model "$m" \
    --topics $TOPICS --policies $POL --seeds $SEEDS --n-turns $TURNS \
    --pool posts/pool.jsonl --out-root $OUT 2>&1 | tee -a results/diverse.log || echo "  >> $m FAILED, continuing"
done

echo "[done] diverse-families pipeline complete"
