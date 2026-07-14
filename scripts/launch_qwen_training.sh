#!/bin/bash
set -e
cd /home/forge/Psychological-Bias-Transfer

SEEDS=(17 42 73 88 91)
CORPUSES=("control" "treatment")

echo "Submitting Qwen2.5-7B Training jobs..."
for seed in "${SEEDS[@]}"; do
  for corpus in "${CORPUSES[@]}"; do
    # Skip jobs that already succeeded
    if [ "$corpus" == "control" ] && { [ "$seed" == "42" ] || [ "$seed" == "73" ] || [ "$seed" == "88" ]; }; then
        continue
    fi

    echo "Submitting: Training Qwen2.5-7B | ${corpus} | Seed ${seed}"
    GCP_PROJECT="citric-snow-496311-f6" \
    GCS_BUCKET="parallax-model-training-citric-snow-496311-f6" \
    MODEL="qwen2.5-7b" \
    CORPUS="${corpus}" \
    SEED="${seed}" \
    VERTEX_REGION="us-central1" \
    VERTEX_MACHINE="g2-standard-4" \
    VERTEX_ACCEL="NVIDIA_L4" \
    bash scripts/run_on_vertex.sh >> /tmp/qwen_train.log 2>&1 || true
    sleep 5
  done
done
echo "All Qwen2.5-7B Training jobs submitted."
