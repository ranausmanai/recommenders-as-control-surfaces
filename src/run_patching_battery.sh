#!/usr/bin/env bash
# Full patching battery for top-conf quality.
# (a) dose-response on Qwen2.5-3B L17, scales 0..20
# (b) layer specificity on Qwen2.5-3B at scale=10
# (c) cross-direction: engagement_max - random as target
# (d) cross-model: Yi-1.5-6B and Zephyr-7B at their best layers, scale=10
set -e
cd "$(dirname "$0")/.."

# (a) dose-response on Qwen2.5-3B
echo "[batt-a] dose-response Qwen2.5-3B L17 recency"
ACT_ROOT=activations_cuda python3 notebooks/06c_patching_battery.py \
  --model-name Qwen_Qwen2.5-3B-Instruct --topic remote_work \
  --layer 17 --scales 0 2 5 8 10 12 15 20 \
  --target-policy recency --device cuda --n-turns 12 --seed 7 2>&1 | tee -a results/battery.log

# (b) layer specificity at scale=10 on Qwen2.5-3B
for L in 8 11 14 17 20 23 26; do
  echo "[batt-b] layer specificity Qwen2.5-3B L$L scale=10"
  ACT_ROOT=activations_cuda python3 notebooks/06c_patching_battery.py \
    --model-name Qwen_Qwen2.5-3B-Instruct --topic remote_work \
    --layer $L --scales 10 \
    --target-policy recency --device cuda --n-turns 10 --seed 7 2>&1 | tee -a results/battery.log
done

# (c) cross-direction: engagement_max as target
echo "[batt-c] cross-direction engagement_max on Qwen2.5-3B L17"
ACT_ROOT=activations_cuda python3 notebooks/06c_patching_battery.py \
  --model-name Qwen_Qwen2.5-3B-Instruct --topic remote_work \
  --layer 17 --scales 10 \
  --target-policy engagement_max --device cuda --n-turns 12 --seed 7 2>&1 | tee -a results/battery.log

# (d) cross-model replication at scale=10
echo "[batt-d1] Yi-1.5-6B L11 scale=10 recency (4-bit)"
ACT_ROOT=activations_cuda python3 notebooks/06c_patching_battery.py \
  --model-name 01-ai_Yi-1.5-6B-Chat --topic remote_work \
  --layer 11 --scales 10 \
  --target-policy recency --device cuda --n-turns 12 --seed 7 --load-in-4bit 2>&1 | tee -a results/battery.log

echo "[batt-d2] Zephyr-7B L28 scale=10 recency (4-bit)"
ACT_ROOT=activations_cuda python3 notebooks/06c_patching_battery.py \
  --model-name HuggingFaceH4_zephyr-7b-beta --topic remote_work \
  --layer 28 --scales 10 \
  --target-policy recency --device cuda --n-turns 12 --seed 7 --load-in-4bit 2>&1 | tee -a results/battery.log

echo "[done] patching battery complete"
