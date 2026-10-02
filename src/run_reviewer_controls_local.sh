#!/usr/bin/env bash
# Resumable local queue for peer-review controls. Safe to stop and rerun.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results

PYTHON_BIN=/opt/anaconda3/bin/python3

echo "[reviewer-controls] starting $(date)"
echo "[reviewer-controls] output: results/reviewer_controls.jsonl"
echo "[reviewer-controls] status: $PYTHON_BIN src/reviewer_controls_status.py"

caffeinate -dimsu "$PYTHON_BIN" src/run_reviewer_controls_queue.py --n 50 \
  2>&1 | tee -a results/reviewer_controls_queue.log

"$PYTHON_BIN" src/reviewer_controls_analyze.py
echo "[reviewer-controls] complete $(date)"
