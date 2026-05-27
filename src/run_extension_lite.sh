#!/usr/bin/env bash
# Lighter extension run after observing 75s/turn on SmolLM2-360M.
# Uniform 1 seed × 8 turns × 3 policies = 24 samples per (model, topic).
# SmolLM2 remote_work already has 6 runs at 10 turns / 2 seeds (kept).
set -e
cd "$(dirname "$0")/.."

# Remaining topics for SmolLM2 (skip remote_work — already done)
TOPICS_REM="ai_regulation nuclear_energy ubi gene_editing"
TOPICS_ALL="remote_work ai_regulation nuclear_energy ubi gene_editing"
TOPICS_NEW="ubi gene_editing"
POL="random recency engagement_max"

echo "[ext-lite 1/3] SmolLM2-360M-Instruct on 4 remaining topics"
PYTHONPATH=src python3 src/agent_loop.py \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --topics $TOPICS_REM --policies $POL --seeds 0 \
  --n-turns 8 --pool posts/pool.jsonl --out-root activations 2>&1 | tee -a results/extension_lite.log

echo "[ext-lite 2/3] TinyLlama-1.1B-Chat on all 5 topics"
PYTHONPATH=src python3 src/agent_loop.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --topics $TOPICS_ALL --policies $POL --seeds 0 \
  --n-turns 8 --pool posts/pool.jsonl --out-root activations 2>&1 | tee -a results/extension_lite.log

echo "[ext-lite 3/3] Qwen3.5-0.8B on 2 new topics"
PYTHONPATH=src python3 src/agent_loop.py \
  --model Qwen/Qwen3.5-0.8B \
  --topics $TOPICS_NEW --policies $POL --seeds 0 \
  --n-turns 8 --pool posts/pool.jsonl --out-root activations 2>&1 | tee -a results/extension_lite.log

echo "[analysis] per-model 01_fingerprint"
for m in Qwen_Qwen3.5-0.8B HuggingFaceTB_SmolLM2-360M-Instruct TinyLlama_TinyLlama-1.1B-Chat-v1.0; do
  echo "  >> $m"
  PYTHONPATH=src python3 notebooks/01_fingerprint.py "$m" 2>&1 | tail -3 | tee -a results/extension_lite.log
  cp results/01_fingerprint.json "results/01_fingerprint_${m}.json" 2>/dev/null || true
done

echo "[analysis] 05_robustness across all 4 models"
PYTHONPATH=src python3 notebooks/05_robustness.py \
  Qwen_Qwen3.5-0.8B Qwen_Qwen2.5-0.5B-Instruct \
  HuggingFaceTB_SmolLM2-360M-Instruct TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  2>&1 | tee -a results/extension_lite.log

echo "[done] extension-lite complete"
