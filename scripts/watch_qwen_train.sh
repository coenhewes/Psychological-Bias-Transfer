#!/bin/bash
PROJECT="citric-snow-496311-f6"
REGION="us-central1"

echo "Waiting for all Qwen2.5-7B training jobs to complete..."

while true; do
  gcloud ai custom-jobs list --region=$REGION --project=$PROJECT --filter="state=(JOB_STATE_PENDING,JOB_STATE_RUNNING)" --format="value(name)" > /tmp/active_jobs.txt 2>/dev/null || true
  active=$(cat /tmp/active_jobs.txt | wc -l)
  
  if [ "$active" -eq 0 ]; then
    echo "All training jobs finished!"
    break
  fi
  echo "$(date -u +%FT%TZ) - Active training jobs: $active"
  sleep 60
done

echo "Starting generation phase..."
bash /home/forge/Psychological-Bias-Transfer/scripts/launch_qwen.sh > /tmp/qwen_launch.log 2>&1 || true
