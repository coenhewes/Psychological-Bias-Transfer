#!/bin/bash
set -u
REGION=us-central1
export GCP_PROJECT=citric-snow-496311-f6
export GCS_BUCKET=parallax-model-training-citric-snow-496311-f6

for corpus in control treatment; do
  for seed in 17 73 88 91; do
    # Skip the ones we already have
    if [ "$corpus" = "control" ] && { [ "$seed" = "73" ] || [ "$seed" = "88" ]; }; then continue; fi
    
    echo "Submitting Qwen2.5-7b eval for $corpus seed $seed..."
    
    P=$(cd /home/forge/Psychological-Bias-Transfer && MODEL=qwen2.5-7b CORPUS=$corpus SEED=$seed VERTEX_REGION=$REGION bash scripts/run_on_vertex_eval.sh 2>&1 | grep -v WARNING | tail -1)
    echo "Launched: $P"
    sleep 5
  done
done
