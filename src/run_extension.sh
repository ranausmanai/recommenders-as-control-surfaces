#!/usr/bin/env bash
# Cross-family / cross-topic extension run.
#
# Adds 2 non-Qwen agent models (SmolLM2-360M-Instruct, TinyLlama-1.1B-Chat-v1.0)
# and 2 new topics (ubi, gene_editing). Trimmed scale 3 policies x 2 seeds x 10 turns
# per (model, topic) cell so total wall-clock fits ~5-6h.
#
# Coverage:
#   SmolLM2-360M-Instruct: 5 topics (3 orig + 2 new)
#   TinyLlama-1.1B-Chat:   5 topics (3 orig + 2 new)
#   Qwen3.5-0.8B:          2 new topics only (already have 3 orig)
#
# Then re-runs robustness analysis on the combined dataset.
set -e
cd "$(dirname "$0")/.."

# Wait for new posts to be appended.
echo "[wait] waiting for posts/pool.jsonl >= 500 lines (3 orig topics + 2 new)"
while true; do
  n=$(wc -l < posts/pool.jsonl)
  [ "$n" -ge 500 ] && break
  sleep 5
done
echo "[wait] pool ready with $n posts"

# Common args
TOPICS_ALL="remote_work ai_regulation nuclear_energy ubi gene_editing"
TOPICS_NEW="ubi gene_editing"
POL="random recency engagement_max"
SEEDS="0 1"

# --- SmolLM2-360M-Instruct across all 5 topics ---
echo "[ext 1/3] SmolLM2-360M-Instruct on 5 topics"
PYTHONPATH=src python3 src/agent_loop.py \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --topics $TOPICS_ALL --policies $POL --seeds $SEEDS \
  --n-turns 10 --pool posts/pool.jsonl --out-root activations 2>&1 | tee -a results/extension.log

# --- TinyLlama-1.1B-Chat across all 5 topics ---
echo "[ext 2/3] TinyLlama-1.1B-Chat on 5 topics"
PYTHONPATH=src python3 src/agent_loop.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --topics $TOPICS_ALL --policies $POL --seeds $SEEDS \
  --n-turns 10 --pool posts/pool.jsonl --out-root activations 2>&1 | tee -a results/extension.log

# --- Qwen3.5-0.8B extended to the 2 new topics ---
echo "[ext 3/3] Qwen3.5-0.8B on 2 new topics"
PYTHONPATH=src python3 src/agent_loop.py \
  --model Qwen/Qwen3.5-0.8B \
  --topics $TOPICS_NEW --policies $POL --seeds $SEEDS \
  --n-turns 10 --pool posts/pool.jsonl --out-root activations 2>&1 | tee -a results/extension.log

echo "[analysis] re-run 01-04 per primary model"
for m in Qwen_Qwen3.5-0.8B HuggingFaceTB_SmolLM2-360M-Instruct TinyLlama_TinyLlama-1.1B-Chat-v1.0; do
  echo "  >> $m"
  PYTHONPATH=src python3 notebooks/01_fingerprint.py "$m" 2>&1 | tail -3 | tee -a results/extension.log
  cp results/01_fingerprint.json "results/01_fingerprint_${m}.json" 2>/dev/null || true
done

echo "[analysis] robustness across all models"
PYTHONPATH=src python3 notebooks/05_robustness.py \
  Qwen_Qwen3.5-0.8B Qwen_Qwen2.5-0.5B-Instruct \
  HuggingFaceTB_SmolLM2-360M-Instruct TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  2>&1 | tee -a results/extension.log

echo "[done] extension complete"
