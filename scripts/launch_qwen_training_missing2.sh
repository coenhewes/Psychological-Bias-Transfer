#!/bin/bash
set -e
cd /home/forge/Psychological-Bias-Transfer
for corpus_seed in "treatment 73" "control 88" "treatment 88"; do
  set -- $corpus_seed
  corpus=$1
  seed=$2
  echo "Submitting: Training Qwen2.5-7B | ${corpus} | Seed ${seed}"
  GCP_PROJECT="citric-snow-496311-f6" \
  GCS_BUCKET="parallax-model-training-citric-snow-496311-f6" \
  MODEL="Qwen/Qwen2.5-7B" \
  CORPUS="${corpus}" \
  SEED="${seed}" \
  VERTEX_REGION="us-central1" \
  VERTEX_MACHINE="g2-standard-4" \
  VERTEX_ACCEL="NVIDIA_L4" \
  bash scripts/run_on_vertex.sh >> /tmp/qwen_train_resubmit.log 2>&1 || true
  sleep 5
done
