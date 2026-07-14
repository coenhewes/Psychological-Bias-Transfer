#!/usr/bin/env bash
# Watch training 88/91 (4 jobs, us-east1). When ALL SUCCEEDED, launch the 4
# extra-seed generation jobs (treatment/control x seed 88/91) via run_on_vertex_eval.sh.
# Then wait for the 4 fp files in generations_fp/. Logs to /home/forge/pbt_gen_watch.log.
set -u
REGION=us-east1
TRAIN_JOBS="4465526632077066240 1304140431151333376 7831967353536512000 7286046636206194688"
BUCKET="gs://parallax-model-training-citric-snow-496311-f6/generations_fp"
LOG=/home/forge/pbt_gen_watch.log
GEN_LAUNCHED=0
echo "[$(date -u +%H:%M:%S)] gen-watcher started" | tee -a "$LOG"

state_of() {
  gcloud ai custom-jobs describe "$1" --region=$REGION --format="value(state)" 2>/dev/null | grep -v WARNING | head -1
}

while true; do
  all_done=1
  for j in $TRAIN_JOBS; do
    st=$(state_of "$j")
    [ "$st" != "JOB_STATE_SUCCEEDED" ] && all_done=0
  done
  echo "[$(date -u +%H:%M:%S)] training all_succeeded=$all_done" | tee -a "$LOG"
  if [ "$all_done" = "1" ] && [ "$GEN_LAUNCHED" = "0" ]; then
    echo "[$(date -u +%H:%M:%S)] launching 4 extra-seed gen jobs" | tee -a "$LOG"
    for corpus in treatment control; do
      for seed in 88 91; do
        P=$(cd /home/forge/Psychological-Bias-Transfer && MODEL=llama3.1-8b CORPUS=$corpus SEED=$seed VERTEX_REGION=$REGION bash scripts/run_on_vertex_eval.sh 2>&1 | grep -v WARNING | tail -1)
        echo "[$(date -u +%H:%M:%S)] gen $corpus seed$seed -> $P" | tee -a "$LOG"
      done
    done
    GEN_LAUNCHED=1
  fi
  # after launch, wait for 4 fp files
  if [ "$GEN_LAUNCHED" = "1" ]; then
    g=0
    for corpus in treatment control; do
      for seed in 88 91; do
        gsutil -q stat "$BUCKET/llama3.1-8b_${corpus}_seed${seed}_fp.jsonl" 2>/dev/null && g=$((g+1))
      done
    done
    echo "[$(date -u +%H:%M:%S)] extra-seed fp present: $g/4" | tee -a "$LOG"
    [ "$g" = "4" ] && { echo "[$(date -u +%H:%M:%S)] ALL 4 FP DONE" | tee -a "$LOG"; break; }
  fi
  sleep 90
done
echo "GEN_WATCH_DONE" | tee -a "$LOG"
