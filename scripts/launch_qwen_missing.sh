#!/bin/bash
set -u
REGION=us-central1
export GCP_PROJECT=citric-snow-496311-f6
export GCS_BUCKET=parallax-model-training-citric-snow-496311-f6

for seed in 17 73 88 91; do
    echo "Submitting Qwen2.5-7b eval for treatment seed $seed..."
    
    # Using L4 machine spec that matches the others to be consistent, but we are 
    # forcing the gsutil copy to retry if it fails (the original script only retried externally, 
    # not the internal worker script's final cp).
    P=$(cd /home/forge/Psychological-Bias-Transfer && MODEL=qwen2.5-7b CORPUS=treatment SEED=$seed VERTEX_REGION=$REGION bash scripts/run_on_vertex_eval.sh 2>&1 | grep -v WARNING | tail -1)
    echo "Launched: $P"
    sleep 5
done
