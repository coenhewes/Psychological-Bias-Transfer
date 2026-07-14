#!/usr/bin/env bash
# Autonomous gpt-oss probe driver. Stays ON VERTEX. Rotates over (REGION, ACCEL)
# pairs so it grabs whatever GPU type allocates (L4/T4/A100 all run gpt-oss-20b 4-bit).
# For each pair: submit probe -> watchdog (30m pending / 90m running). On escalation
# (pending timeout or running deadlock): cancel + next (region,accel) pair. On
# SUCCEEDED-with-output: exit 0. On hard fail (succeeded-no-output / failed): exit.
set -u
cd /home/forge/Psychological-Bias-Transfer
EXPECTED="gs://parallax-model-training-citric-snow-496311-f6/generations_fp/probe_gptoss_RESULT.log"
PROJECT="${GCP_PROJECT:-citric-snow-496311-f6}"
# (region, accel, machine) pairs — T4 on n1, L4 on g2, A100 on a2
PAIRS=(
  "us-east1:NVIDIA_TESLA_A100:a2-highgpu-1g"
  "us-central1:NVIDIA_TESLA_A100:a2-highgpu-1g"
  "us-east1:NVIDIA_L4:g2-standard-4"
  "us-central1:NVIDIA_L4:g2-standard-4"
  "us-east4:NVIDIA_L4:g2-standard-4"
)
MAX="${1:-${#PAIRS[@]}}"

for cyc in $(seq 1 "$MAX"); do
  pair="${PAIRS[$((cyc-1))]}"
  REGION="${pair%%:*}"; rest="${pair#*:}"; ACCEL="${rest%%:*}"; MACHINE="${rest##*:}"
  echo "[DRIVER $(date -u +%H:%M:%S)] cycle $cyc/${#PAIRS[@]} region=$REGION accel=$ACCEL machine=$MACHINE"
  # submit probe with accel/machine override via env vars (launcher exposes VERTEX_ACCEL/VERTEX_MACHINE)
  P=$(GCP_PROJECT="$PROJECT" VERTEX_REGION="$REGION" VERTEX_ACCEL="$ACCEL" VERTEX_MACHINE="$MACHINE" \
      bash scripts/run_probe_gptoss.sh 2>&1 | grep -v WARNING | tail -1)
  # launcher prints "JID|gs://.../probe_gptoss_<JOB_NAME>.log"
  JID=$(echo "$P" | cut -d'|' -f1 | grep -oE '[0-9]+$')
  EXPECTED=$(echo "$P" | cut -d'|' -f2)
  if [ -z "$JID" ] || [ -z "$EXPECTED" ]; then
    echo "[DRIVER] ERROR: empty JID/EXPECTED from launcher (output was: '$P'). Skipping this pair."
    continue
  fi
  echo "[DRIVER] submitted probe $JID ($ACCEL @ $REGION) log=$EXPECTED"
  bash scripts/gpu_job_watchdog.sh "$JID" "$REGION" "$EXPECTED" 30 90
  rc=$?
  if [ "$rc" = "0" ]; then echo "[DRIVER] probe SUCCEEDED with output. DONE."; exit 0; fi
  if [ "$rc" = "2" ] || [ "$rc" = "3" ]; then echo "[DRIVER] HARD FAIL rc=$rc. ESCALATE."; exit "$rc"; fi
  echo "[DRIVER] watchdog escalated rc=$rc — cancel $JID, next pair."
  gcloud ai custom-jobs cancel "$JID" --region="$REGION" --project="$PROJECT" 2>/dev/null | grep -v WARNING >/dev/null
  sleep 20
done
echo "[DRIVER] exhausted all pairs without success. ESCALATE."
exit 6
