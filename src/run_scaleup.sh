#!/usr/bin/env bash
# Scale up beyond the pilot:
#   (1) extend Qwen3.5-0.8B to the two remaining topics (ai_regulation, nuclear_energy)
#       — pilot already covered remote_work
#   (2) cross-model: run the entire 1-topic pilot on a SECOND agent
#       (Qwen/Qwen2.5-0.5B-Instruct as the smaller different-family fallback)
# Each cell = 3 policies × 3 seeds × 15 turns.
#
# Then re-run analysis notebooks on the combined dataset and write full_summary.md.

set -e
cd "$(dirname "$0")/.."

echo "[scale 1/3] Qwen3.5-0.8B on ai_regulation"
PYTHONPATH=src python3 src/agent_loop.py \
  --model Qwen/Qwen3.5-0.8B \
  --topics ai_regulation \
  --policies random recency engagement_max \
  --seeds 0 1 2 \
  --n-turns 15 \
  --pool posts/pool.jsonl \
  --out-root activations 2>&1 | tee -a results/scale.log

echo "[scale 2/3] Qwen3.5-0.8B on nuclear_energy"
PYTHONPATH=src python3 src/agent_loop.py \
  --model Qwen/Qwen3.5-0.8B \
  --topics nuclear_energy \
  --policies random recency engagement_max \
  --seeds 0 1 2 \
  --n-turns 15 \
  --pool posts/pool.jsonl \
  --out-root activations 2>&1 | tee -a results/scale.log

echo "[scale 3/3] Second agent (Qwen2.5-0.5B-Instruct) on remote_work"
PYTHONPATH=src python3 src/agent_loop.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --topics remote_work \
  --policies random recency engagement_max \
  --seeds 0 1 2 \
  --n-turns 15 \
  --pool posts/pool.jsonl \
  --out-root activations 2>&1 | tee -a results/scale.log

echo "[analysis] re-run 01_fingerprint on primary agent (full 3-topic data)"
PYTHONPATH=src python3 notebooks/01_fingerprint.py Qwen_Qwen3.5-0.8B 2>&1 | tee -a results/scale.log
echo "[analysis] re-run 02_geometry"
PYTHONPATH=src python3 notebooks/02_geometry.py Qwen_Qwen3.5-0.8B 2>&1 | tee -a results/scale.log
echo "[analysis] re-run 03_behavior"
PYTHONPATH=src python3 notebooks/03_behavior.py Qwen_Qwen3.5-0.8B 2>&1 | tee -a results/scale.log
echo "[analysis] re-run 04_jima"
PYTHONPATH=src python3 notebooks/04_jima_compare.py Qwen_Qwen3.5-0.8B 2>&1 | tee -a results/scale.log

echo "[analysis] cross-model agent #2"
PYTHONPATH=src python3 notebooks/01_fingerprint.py Qwen_Qwen2.5-0.5B-Instruct 2>&1 | tee -a results/scale.log || true

echo "[summary] full_summary.md"
PYTHONPATH=src python3 src/write_summary.py full_summary.md 2>&1 | tee -a results/scale.log

echo "[done] scale-up complete"
