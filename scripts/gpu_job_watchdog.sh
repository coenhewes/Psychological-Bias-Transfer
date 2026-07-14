#!/usr/bin/env bash
# GPU-job watchdog: NO SILENT FAILURES.
# Rule: a Vertex job may sit PENDING/queued at most MAX_PENDING_MIN. Past that,
# CANCEL + RESUBMIT in a different region + REPORT. If it flips to RUNNING but
# produces no expected output within MAX_RUN_MIN, CANCEL + RESUBMIT + REPORT.
# If it SUCCEEDS but expected output is missing, REPORT (code hang, not allocation).
#
# Args: JOB_ID REGION EXPECTED_GCS_URI [MAX_PENDING_MIN=30] [MAX_RUN_MIN=120]
set -u
JOB_ID="$1"; REGION="$2"; EXPECTED="$3"
MAX_PENDING_MIN="${4:-30}"; MAX_RUN_MIN="${5:-120}"
PROJECT="${GCP_PROJECT:-citric-snow-496311-f6}"
REGIONS=(us-east1 us-central1 us-east4 us-west1 europe-west4)
started=$(date +%s)

report() { echo "[WATCHDOG $(date -u +%H:%M:%S)] $*"; }

while true; do
  now=$(date +%s)
  st=$(gcloud ai custom-jobs describe "$JOB_ID" --region="$REGION" --project="$PROJECT" --format="value(state)" 2>/dev/null | grep -v WARNING | head -1)
  # expected output present? Download and check CONTENT for explicit success marker.
  if gsutil -q stat "$EXPECTED" 2>/dev/null; then
    gsutil cp "$EXPECTED" /tmp/watchdog_expected.log 2>/dev/null
    if grep -q "PROBE_OK" /tmp/watchdog_expected.log 2>/dev/null; then
      report "EXPECTED OUTPUT PRESENT + PROBE_OK in $EXPECTED — job $JOB_ID done. Watchdog exit."
      exit 0
    fi
    if grep -q "PROBE_FAIL" /tmp/watchdog_expected.log 2>/dev/null; then
      report "EXPECTED OUTPUT PRESENT but PROBE_FAIL in $EXPECTED — job ran and FAILED. ESCALATE (code/config issue, not allocation)."
      exit 2
    fi
    # log exists but no marker yet — job may still be running/writing. Keep waiting.
  fi
  if [ "$st" = "JOB_STATE_SUCCEEDED" ]; then
    report "JOB SUCCEEDED but EXPECTED OUTPUT MISSING ($EXPECTED). CODE HANG, not allocation. ESCALATE."
    exit 2
  fi
  if [ "$st" = "JOB_STATE_FAILED" ]; then
    report "JOB FAILED ($JOB_ID). ESCALATE — read logs."
    exit 3
  fi
  if [ "$st" = "JOB_STATE_CANCELLED" ]; then
    report "JOB CANCELLED ($JOB_ID). Treating as terminal — ESCALATE (check if another watchdog cancelled it, or code issue)."
    exit 2
  fi
  elapsed=$(( (now - started) / 60 ))
  if [ "$st" = "JOB_STATE_PENDING" ] || [ "$st" = "JOB_STATE_QUEUED" ] || [ -z "$st" ]; then
    if [ "$elapsed" -ge "$MAX_PENDING_MIN" ]; then
      report "PENDING > ${MAX_PENDING_MIN}m (silent-allocation risk). CANCEL + RESUBMIT in next region."
      gcloud ai custom-jobs cancel "$JOB_ID" --region="$REGION" --project="$PROJECT" 2>/dev/null | grep -v WARNING >/dev/null
      # pick a different region
      next="${REGIONS[0]}"; for r in "${REGIONS[@]}"; do [ "$r" != "$REGION" ] && { next="$r"; break; }; done
      report "resubmit would target $next — caller handles resubmit (this watchdog monitors ONE job)."
      exit 4
    fi
  elif [ "$st" = "JOB_STATE_RUNNING" ]; then
    if [ "$elapsed" -ge "$((MAX_PENDING_MIN + MAX_RUN_MIN))" ]; then
      report "RUNNING > ${MAX_RUN_MIN}m with no output (possible deadlock). CANCEL + ESCALATE."
      gcloud ai custom-jobs cancel "$JOB_ID" --region="$REGION" --project="$PROJECT" 2>/dev/null | grep -v WARNING >/dev/null
      exit 5
    fi
  fi
  report "state=$st elapsed=${elapsed}m — ok, waiting."
  sleep 300
done
