#!/usr/bin/env bash
# Stage 1 fast read: do the values topics replicate, does the security probe move?
# Susceptible models only, baseline vs heavy, 20 seeds.
# Order: llama x ai_regulation first (cleanest generalization test).
set -u
cd "$(dirname "$0")/.."

OUT=results/decision_shift_tasks.jsonl
MODELS=(llama3.2:3b gemma4:e4b)
TOPICS=(ai_regulation ubi deploy_security)
CONDS=(organic_random heavy)
SEEDS=$(seq 0 19)

# topic -> organic pool path (values topics reuse the shared pool.jsonl)
pool_for() {
  case "$1" in
    ai_regulation|ubi|nuclear_energy|gene_editing|remote_work) echo "posts/pool.jsonl" ;;
    *) echo "posts/pool_$1.jsonl" ;;
  esac
}

echo "=== stage1 starting $(date) ==="
for MODEL in "${MODELS[@]}"; do
  for TOPIC in "${TOPICS[@]}"; do
    POOL=$(pool_for "$TOPIC")
    echo ">>> $MODEL / $TOPIC (pool=$POOL) @ $(date +%H:%M:%S)"
    python3 src/decision_shift_adv_ollama.py \
      --model "$MODEL" --topic "$TOPIC" \
      --pool "$POOL" --adv "posts/adversarial_${TOPIC}.jsonl" \
      --conditions "${CONDS[@]}" --seeds $SEEDS \
      --out "$OUT" --tag "$TOPIC" \
      || echo "!!! $MODEL/$TOPIC failed, continuing"
  done
done
echo "=== stage1 done $(date) ==="
