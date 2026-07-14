#!/bin/bash
set -e
cd /home/forge/Psychological-Bias-Transfer

JOBS=(
    "control 91"
    "treatment 91"
)

echo "Submitting missing Qwen2.5-7B Training jobs to us-east4..."
for job in "${JOBS[@]}"; do
    read -r corpus seed <<< "$job"
    echo "Submitting: Training Qwen2.5-7B | ${corpus} | Seed ${seed} (us-east4)"
    GCP_PROJECT="citric-snow-496311-f6" \
    GCS_BUCKET="parallax-model-training-citric-snow-496311-f6" \
    MODEL="Qwen/Qwen2.5-7B" \
    CORPUS="${corpus}" \
    SEED="${seed}" \
    VERTEX_REGION="us-east4" \
    VERTEX_MACHINE="g2-standard-4" \
    VERTEX_ACCEL="NVIDIA_L4" \
    bash scripts/run_on_vertex.sh
    sleep 5
done
echo "All missing training jobs submitted to us-east4."