#!/usr/bin/env bash
# Run on the RTX 4000 pod. Two axes:
#   (1) SCALE axis: Qwen2.5-Instruct at 0.5B / 1.5B / 3B / 7B(4bit)
#   (2) FAMILY axis at ~3B-ish: + Phi-3.5-mini-instruct + SmolLM2-1.7B-Instruct
# Topic: remote_work (we have rich prior data for comparison).
# 3 policies × 2 seeds × 10 turns = 60 samples per (model, topic) cell.
set -e
cd "$(dirname "$0")/.."

OUT=activations_cuda
POL="random recency engagement_max"
SEEDS="0 1"
TURNS=10
TOPICS="remote_work ai_regulation ubi"   # 3 topics so cross-topic transfer at scale

# Same-family scale axis: Qwen2.5 0.5B → 7B
for m in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct; do
  echo "[scale] $m"
  python3 src/agent_loop.py --device cuda --model "$m" \
    --topics $TOPICS --policies $POL --seeds $SEEDS --n-turns $TURNS \
    --pool posts/pool.jsonl --out-root $OUT 2>&1 | tee -a results/cuda_run.log
done

# 7B in 4-bit (saves VRAM, leaves room for the other experiment)
echo "[scale] Qwen2.5-7B-Instruct in 4-bit"
python3 src/agent_loop.py --device cuda --load-in-4bit --model Qwen/Qwen2.5-7B-Instruct \
  --topics $TOPICS --policies $POL --seeds $SEEDS --n-turns $TURNS \
  --pool posts/pool.jsonl --out-root $OUT 2>&1 | tee -a results/cuda_run.log

# Family axis at ~mid-scale
echo "[family] Phi-3.5-mini-instruct"
python3 src/agent_loop.py --device cuda --model microsoft/Phi-3.5-mini-instruct \
  --topics $TOPICS --policies $POL --seeds $SEEDS --n-turns $TURNS \
  --pool posts/pool.jsonl --out-root $OUT 2>&1 | tee -a results/cuda_run.log

echo "[family] SmolLM2-1.7B-Instruct"
python3 src/agent_loop.py --device cuda --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --topics $TOPICS --policies $POL --seeds $SEEDS --n-turns $TURNS \
  --pool posts/pool.jsonl --out-root $OUT 2>&1 | tee -a results/cuda_run.log

echo "[done] CUDA scale + family runs complete"
