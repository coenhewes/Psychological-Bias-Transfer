#!/bin/bash
PROJECT="citric-snow-496311-f6"
REGION="us-central1"

echo "Waiting for all Qwen2.5-7B evaluation jobs to complete in us-central1..."

while true; do
  active_jobs=$(gcloud ai custom-jobs list --region=$REGION --project=$PROJECT --filter="state=(JOB_STATE_PENDING,JOB_STATE_RUNNING)" --format="value(name)" 2>/dev/null | grep customJobs | wc -l || echo "0")
  
  if [ "$active_jobs" -eq 0 ]; then
    echo "All generation jobs finished!"
    break
  fi
  echo "$(date -u +%FT%TZ) - Active jobs: $active_jobs"
  sleep 120
done
