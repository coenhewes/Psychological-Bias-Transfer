#!/bin/bash
set -u
REGION=us-central1
export GCP_PROJECT=citric-snow-496311-f6
export GCS_BUCKET=parallax-model-training-citric-snow-496311-f6

for seed in 17 73 88 91; do
    echo "Submitting Qwen2.5-7b eval for treatment seed $seed on A100..."
    # We switch to a2-highgpu-1g (A100 40GB) to bypass L4 congestion
    P=$(cd /home/forge/Psychological-Bias-Transfer && MODEL=qwen2.5-7b CORPUS=treatment SEED=$seed VERTEX_REGION=$REGION VERTEX_MACHINE=a2-highgpu-1g VERTEX_ACCEL=NVIDIA_TESLA_A100 bash scripts/run_on_vertex_eval.sh 2>&1 | grep -v WARNING | tail -1)
    echo "Launched: $P"
    sleep 5
done
