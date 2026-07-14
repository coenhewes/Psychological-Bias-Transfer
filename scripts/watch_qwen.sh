#!/bin/bash
PROJECT="citric-snow-496311-f6"
REGION="us-central1"

echo "Waiting for all Qwen2.5-7B generation jobs to complete..."

while true; do
  active=$(gcloud ai custom-jobs list --region=$REGION --project=$PROJECT --filter="state=(JOB_STATE_PENDING,JOB_STATE_RUNNING)" --format="value(name)" | wc -l)
  if [ "$active" -eq 0 ]; then
    echo "All jobs finished!"
    break
  fi
  echo "$(date -u +%FT%TZ) - Active jobs: $active"
  sleep 60
done

echo "Downloading generations..."
mkdir -p /home/forge/Psychological-Bias-Transfer/eval_outputs/raw_generations
gsutil -m cp "gs://parallax-model-training-citric-snow-496311-f6/generations_fp/qwen*.jsonl" /home/forge/Psychological-Bias-Transfer/eval_outputs/raw_generations/ 2>/dev/null || true

echo "Jobs complete."
