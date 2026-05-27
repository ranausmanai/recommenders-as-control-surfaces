#!/usr/bin/env bash
# Reactance experiment on MODERN models — both Ollama local + HF downloadable.
# 6 conditions × 20 seeds × 10 turns = 120 rollouts per model.
set +e   # tolerate per-model failure — log it but keep going
cd "$(dirname "$0")/.."

OUT=results/decision_shift_adv_modern.jsonl
rm -f "$OUT"

CONDS="organic_random organic_recency light heavy balanced disclosed_heavy"
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
TURNS=10

# ---- Ollama local models (already downloaded, no extra disk needed) ----
echo "[modern 1] llama3.2:3b (Llama 3.2, Sept 2024)"
python3 src/decision_shift_adv_ollama.py --model llama3.2:3b \
  --topic remote_work --conditions $CONDS --seeds $SEEDS --n-turns $TURNS \
  --out "$OUT" 2>&1 | tee -a results/reactance_modern.log
RC=${PIPESTATUS[0]}; if [ "$RC" != "0" ]; then echo "[WARN] previous model failed rc=$RC, continuing" | tee -a results/reactance_modern.log; fi

echo "[modern 2] qwen3.5:2b (Qwen 3.5, recent)"
python3 src/decision_shift_adv_ollama.py --model qwen3.5:2b \
  --topic remote_work --conditions $CONDS --seeds $SEEDS --n-turns $TURNS \
  --out "$OUT" 2>&1 | tee -a results/reactance_modern.log
RC=${PIPESTATUS[0]}; if [ "$RC" != "0" ]; then echo "[WARN] previous model failed rc=$RC, continuing" | tee -a results/reactance_modern.log; fi

echo "[modern 3] qwen3.5:9b (Qwen 3.5 9B, larger modern)"
python3 src/decision_shift_adv_ollama.py --model qwen3.5:9b \
  --topic remote_work --conditions $CONDS --seeds $SEEDS --n-turns $TURNS \
  --out "$OUT" 2>&1 | tee -a results/reactance_modern.log
RC=${PIPESTATUS[0]}; if [ "$RC" != "0" ]; then echo "[WARN] previous model failed rc=$RC, continuing" | tee -a results/reactance_modern.log; fi

echo "[modern 4] gemma4:e4b (Gemma 4, 2025)"
python3 src/decision_shift_adv_ollama.py --model gemma4:e4b \
  --topic remote_work --conditions $CONDS --seeds $SEEDS --n-turns $TURNS \
  --out "$OUT" 2>&1 | tee -a results/reactance_modern.log
RC=${PIPESTATUS[0]}; if [ "$RC" != "0" ]; then echo "[WARN] previous model failed rc=$RC, continuing" | tee -a results/reactance_modern.log; fi

# ---- HF modern models (will download on first use) ----
echo "[modern 5] microsoft/Phi-4-mini-instruct (Phi 4, early 2025)"
python3 src/decision_shift_adv.py --device cuda --model microsoft/Phi-4-mini-instruct \
  --topic remote_work --conditions $CONDS --seeds $SEEDS --n-turns $TURNS \
  --out "$OUT" 2>&1 | tee -a results/reactance_modern.log
RC=${PIPESTATUS[0]}; if [ "$RC" != "0" ]; then echo "[WARN] previous model failed rc=$RC, continuing" | tee -a results/reactance_modern.log; fi

echo "[modern 6] HuggingFaceTB/SmolLM3-3B (HF, 2025)"
python3 src/decision_shift_adv.py --device cuda --model HuggingFaceTB/SmolLM3-3B \
  --topic remote_work --conditions $CONDS --seeds $SEEDS --n-turns $TURNS \
  --out "$OUT" 2>&1 | tee -a results/reactance_modern.log
RC=${PIPESTATUS[0]}; if [ "$RC" != "0" ]; then echo "[WARN] previous model failed rc=$RC, continuing" | tee -a results/reactance_modern.log; fi

echo "[modern 7] DeepSeek-R1-Distill-Qwen-7B (DeepSeek, Jan 2025, 4-bit)"
python3 src/decision_shift_adv.py --device cuda --load-in-4bit --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --topic remote_work --conditions $CONDS --seeds $SEEDS --n-turns $TURNS \
  --out "$OUT" 2>&1 | tee -a results/reactance_modern.log
RC=${PIPESTATUS[0]}; if [ "$RC" != "0" ]; then echo "[WARN] previous model failed rc=$RC, continuing" | tee -a results/reactance_modern.log; fi

echo "[done] modern reactance pipeline complete"
