#!/usr/bin/env bash
# v2: diverse cross-family run with fixes for phi-2 (no chat template), OLMo-2,
# InternLM (needs protobuf), and 4-bit larger models.
# Skip OLMo-2 (architecture not in transformers 4.46), use OLMo-1 instead.
# Use Phi-2 with a manual chat template injected at runtime.

set -e
cd "$(dirname "$0")/.."

OUT=activations_cuda
POL="random recency engagement_max"
SEEDS="0 1"
TURNS=10
TOPICS="remote_work ai_regulation"

# 1.5B-3B class, native chat models
for m in internlm/internlm2_5-1_8b-chat tiiuae/Falcon3-3B-Instruct; do
  echo "[v2] $m"
  python3 src/agent_loop.py --device cuda --model "$m" \
    --topics $TOPICS --policies $POL --seeds $SEEDS --n-turns $TURNS \
    --pool posts/pool.jsonl --out-root $OUT 2>&1 | tee -a results/diverse_v2.log
done

# Larger models in 4-bit
for m in HuggingFaceH4/zephyr-7b-beta 01-ai/Yi-1.5-6B-Chat; do
  echo "[v2-4bit] $m"
  python3 src/agent_loop.py --device cuda --load-in-4bit --model "$m" \
    --topics $TOPICS --policies $POL --seeds $SEEDS --n-turns $TURNS \
    --pool posts/pool.jsonl --out-root $OUT 2>&1 | tee -a results/diverse_v2.log
done

echo "[done] diverse-v2 pipeline complete"
