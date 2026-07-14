#!/bin/bash
set -e
cd /home/forge/Psychological-Bias-Transfer
rm -rf ~/.gsutil/tracker-files/* 2>/dev/null || true

JOBS=("control 88")

for corpus_seed in "${JOBS[@]}"; do
  set -- $corpus_seed
  corpus=$1
  seed=$2

  echo "Submitting: Training Qwen2.5-7B | ${corpus} | Seed ${seed} on L4 (europe-west4)"
  GCP_PROJECT="citric-snow-496311-f6" \
  GCS_BUCKET="parallax-model-training-citric-snow-496311-f6" \
  MODEL="Qwen/Qwen2.5-7B" \
  CORPUS="${corpus}" \
  SEED="${seed}" \
  VERTEX_REGION="europe-west4" \
  VERTEX_MACHINE="g2-standard-4" \
  VERTEX_ACCEL="NVIDIA_L4" \
  bash scripts/run_on_vertex.sh >> /tmp/qwen_train_resubmit_eu.log 2>&1 || true
  
  sleep 10
done
