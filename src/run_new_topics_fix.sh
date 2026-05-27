#!/usr/bin/env bash
# Re-run the UBI + gene_editing cells that failed due to TOPIC_PROMPTS bug.
set -e
cd "$(dirname "$0")/.."

TOPICS_NEW="ubi gene_editing"
POL="random recency engagement_max"

echo "[fix 1/3] SmolLM2-360M on ubi + gene_editing"
PYTHONPATH=src python3 src/agent_loop.py \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --topics $TOPICS_NEW --policies $POL --seeds 0 \
  --n-turns 8 --pool posts/pool.jsonl --out-root activations 2>&1 | tee -a results/extension_fix.log

echo "[fix 2/3] TinyLlama on ubi + gene_editing"
PYTHONPATH=src python3 src/agent_loop.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --topics $TOPICS_NEW --policies $POL --seeds 0 \
  --n-turns 8 --pool posts/pool.jsonl --out-root activations 2>&1 | tee -a results/extension_fix.log

echo "[fix 3/3] Qwen3.5-0.8B on ubi + gene_editing"
PYTHONPATH=src python3 src/agent_loop.py \
  --model Qwen/Qwen3.5-0.8B \
  --topics $TOPICS_NEW --policies $POL --seeds 0 \
  --n-turns 8 --pool posts/pool.jsonl --out-root activations 2>&1 | tee -a results/extension_fix.log

echo "[analysis] per-model 01_fingerprint (4 models)"
for m in Qwen_Qwen3.5-0.8B HuggingFaceTB_SmolLM2-360M-Instruct TinyLlama_TinyLlama-1.1B-Chat-v1.0; do
  echo "  >> $m"
  PYTHONPATH=src python3 notebooks/01_fingerprint.py "$m" 2>&1 | tail -3 | tee -a results/extension_fix.log
  cp results/01_fingerprint.json "results/01_fingerprint_${m}.json"
done

echo "[analysis] 05_robustness across all 4 models"
PYTHONPATH=src python3 notebooks/05_robustness.py \
  Qwen_Qwen3.5-0.8B Qwen_Qwen2.5-0.5B-Instruct \
  HuggingFaceTB_SmolLM2-360M-Instruct TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  2>&1 | tee -a results/extension_fix.log

echo "[done] fix complete"
