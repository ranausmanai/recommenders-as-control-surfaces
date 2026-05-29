#!/usr/bin/env bash
# Run the second-task generalization grid: 4 models x 3 new security topics
# x {organic_random, heavy, balanced} x 20 seeds, all via local Ollama.
# Susceptible models (llama, gemma) run FIRST so the headline result lands early.
set -u
cd "$(dirname "$0")/.."

OUT=results/decision_shift_tasks.jsonl
TOPICS=(deploy_security vendor_security access_policy)
# susceptible first, then saturation controls
MODELS=(llama3.2:3b gemma4:e4b qwen3.5:2b qwen3.5:9b)
CONDS=(organic_random heavy balanced)
SEEDS=$(seq 0 19)

echo "=== second-task grid starting $(date) ==="
for MODEL in "${MODELS[@]}"; do
  for TOPIC in "${TOPICS[@]}"; do
    echo ">>> $MODEL / $TOPIC @ $(date +%H:%M:%S)"
    python3 src/decision_shift_adv_ollama.py \
      --model "$MODEL" --topic "$TOPIC" \
      --pool "posts/pool_${TOPIC}.jsonl" \
      --adv "posts/adversarial_${TOPIC}.jsonl" \
      --conditions "${CONDS[@]}" \
      --seeds $SEEDS \
      --out "$OUT" --tag "$TOPIC" \
      || echo "!!! $MODEL/$TOPIC failed, continuing"
  done
done
echo "=== second-task grid done $(date) ==="
