#!/usr/bin/env bash
# Wait for posts/pool.jsonl to be fully written by post_gen, then immediately run pilot + analysis.
set -e
cd "$(dirname "$0")/.."

echo "[wait] waiting for posts/pool.jsonl to be ready ..."
while [ ! -f posts/pool.jsonl ]; do sleep 3; done
# Ensure file is stable (not still being written): wait for size to stop growing.
prev=-1
while true; do
  cur=$(wc -c < posts/pool.jsonl)
  if [ "$cur" = "$prev" ]; then break; fi
  prev=$cur
  sleep 4
done
n=$(wc -l < posts/pool.jsonl)
echo "[wait] pool.jsonl ready with $n posts"

echo "[pilot] starting pilot run (1 topic x 3 policies x 3 seeds x 15 turns)"
PYTHONPATH=src python3 src/run_pilot.py 2>&1 | tee results/pilot.log

echo "[analysis] 01_fingerprint"
PYTHONPATH=src python3 notebooks/01_fingerprint.py Qwen_Qwen3.5-0.8B 2>&1 | tee -a results/pilot.log

echo "[analysis] 02_geometry"
PYTHONPATH=src python3 notebooks/02_geometry.py Qwen_Qwen3.5-0.8B 2>&1 | tee -a results/pilot.log

echo "[analysis] 03_behavior"
PYTHONPATH=src python3 notebooks/03_behavior.py Qwen_Qwen3.5-0.8B 2>&1 | tee -a results/pilot.log

echo "[analysis] 04_jima_compare"
PYTHONPATH=src python3 notebooks/04_jima_compare.py Qwen_Qwen3.5-0.8B 2>&1 | tee -a results/pilot.log

echo "[summary] write_summary"
PYTHONPATH=src python3 src/write_summary.py pilot_summary.md 2>&1 | tee -a results/pilot.log

echo "[done] pilot pipeline complete"
