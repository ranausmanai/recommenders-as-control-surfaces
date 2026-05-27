#!/usr/bin/env bash
# Three follow-up experiments on Llama 3.2-3B (the cleanest susceptible model):
#   E1. Anti-direction attack: pro-remote adversarial posts as the "attack"
#   E2. Generator-swap: gemma4-generated organic + adversarial pool, same protocol
#   E3. Dose-response: 0/1/2/3/4/5 adversarial posts per batch
#
# Same robust pattern: set +e, per-experiment failure tolerance.
set +e
cd "$(dirname "$0")/.."

OUT=results/decision_shift_followup.jsonl
rm -f "$OUT"

SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
TURNS=10
MODEL=llama3.2:3b

# E1: anti-direction attack — pro-remote adversarial posts
echo "[E1] Llama 3.2 / anti-direction (pro-remote adversarial)"
python3 src/decision_shift_adv_ollama.py --model "$MODEL" \
  --topic remote_work \
  --conditions organic_random organic_recency light heavy balanced disclosed_heavy \
  --seeds $SEEDS --n-turns $TURNS \
  --pool posts/pool.jsonl \
  --adv posts/adversarial_pro_remote.jsonl \
  --tag E1_anti_direction \
  --out "$OUT" 2>&1 | tee -a results/followup.log
RC=${PIPESTATUS[0]}; if [ "$RC" != "0" ]; then echo "[WARN] E1 rc=$RC" | tee -a results/followup.log; fi

# E2: generator-swap — gemma4 wrote BOTH pools
echo "[E2] Llama 3.2 / generator-swap (gemma4 wrote posts)"
python3 src/decision_shift_adv_ollama.py --model "$MODEL" \
  --topic remote_work \
  --conditions organic_random organic_recency light heavy balanced disclosed_heavy \
  --seeds $SEEDS --n-turns $TURNS \
  --pool posts/pool_gemma.jsonl \
  --adv posts/adversarial_rto_gemma.jsonl \
  --tag E2_gemma_pool \
  --out "$OUT" 2>&1 | tee -a results/followup.log
RC=${PIPESTATUS[0]}; if [ "$RC" != "0" ]; then echo "[WARN] E2 rc=$RC" | tee -a results/followup.log; fi

# E3: dose-response — 0..5 adversarial posts per turn, Claude pool
echo "[E3] Llama 3.2 / dose-response (0..5 adv per batch, Claude pool)"
python3 src/decision_shift_adv_ollama.py --model "$MODEL" \
  --topic remote_work \
  --conditions dose0 dose1 dose2 dose3 dose4 dose5 \
  --seeds $SEEDS --n-turns $TURNS \
  --pool posts/pool.jsonl \
  --adv posts/adversarial_rto.jsonl \
  --tag E3_dose_response \
  --out "$OUT" 2>&1 | tee -a results/followup.log
RC=${PIPESTATUS[0]}; if [ "$RC" != "0" ]; then echo "[WARN] E3 rc=$RC" | tee -a results/followup.log; fi

echo "[done] followup pipeline complete"
