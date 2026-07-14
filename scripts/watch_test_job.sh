#!/usr/bin/env bash
# Watcher for gpt-oss 1-row test job. Follows ACTUAL evidence (job state + startTime),
# not a blind timer. Decision points per handoff:
#  - PENDING with NO startTime past ~25 min -> CANCEL + resubmit (proven stall fix)
#  - RUNNING -> read GCS test log; detect REACHED_JUDGE_LINE + cat'd verdict
#  - real verdict (present bool + parsed raw, NOT error string) -> download + verify, exit 0
#  - SUCCEEDED/FAILED with no real output -> report, exit 1
set +u
source ~/.bashrc 2>/dev/null
export PATH="$PATH:/usr/local/bin"
PROJECT=citric-snow-496311-f6
BUCKET=parallax-model-training-citric-snow-496311-f6
REGION=us-central1
LOG_URI="gs://${BUCKET}/generations_fp/judge_gptoss_test.log"
TEST_OUT_URI="gs://${BUCKET}/generations_fp/TEST_gptoss/"
JID=3315419542229876736
REPO=/home/forge/Psychological-Bias-Transfer
MAX_LOOPS=120   # 120 * 30s = 60 min ceiling

echo "[watch] start $(date -u +%FT%TZ) job=$JID"

describe() {
  gcloud ai custom-jobs describe "$1" --region="$REGION" --project="$PROJECT" \
    --format='value(state,startTime,createTime)' 2>/dev/null
}

log_tail() {
  gsutil cat "$LOG_URI" 2>/dev/null
}

stall_cancel_resubmit() {
  local dead="$1"
  echo "[watch] STALL: $dead PENDING >25m no startTime -> cancel"
  gcloud ai custom-jobs cancel "$dead" --region="$REGION" --project="$PROJECT" 2>&1 | tail -1
  echo "[watch] resubmitting test job..."
  local out
  out=$(cd "$REPO" && bash scripts/run_judge_gptoss_test.sh 2>&1 | tail -3)
  echo "[watch] resubmit output: $out"
  JID=$(echo "$out" | grep -oE '^[0-9]+' | head -1)
  if [ -z "$JID" ]; then
    echo "[watch] RESUBMIT FAILED to get JID"
    return 1
  fi
  echo "[watch] new job=$JID"
  return 0
}

verify_verdict() {
  # download TEST_gptoss file, check schema
  local local_dir=/tmp/pbt_test_verify
  mkdir -p "$local_dir"
  gsutil -m cp "${TEST_OUT_URI}*.jsonl" "$local_dir/" 2>/dev/null
  local f
  f=$(ls -1 "$local_dir"/*.gptoss.judged.jsonl 2>/dev/null | head -1)
  if [ -z "$f" ]; then
    echo "[watch] NO test output file in TEST_gptoss/"
    return 1
  fi
  echo "[watch] downloaded: $f"
  python3 - "$f" <<'PY'
import json,sys
f=sys.argv[1]
line=open(f).readline()
rec=json.loads(line)
ms=rec.get("marker_scores",{})
bad=0
for m,v in ms.items():
    raw=v.get("raw","")
    if raw.startswith("ERROR:") or "name 'torch' is not defined" in raw:
        bad+=1
        print(f"  [{m}] ERROR verdict: {raw[:80]}")
    else:
        print(f"  [{m}] present={v.get('present')} conf={v.get('confidence')} raw={raw[:60]!r}")
if bad:
    print("VERDICT: BROKEN (error strings present)")
    sys.exit(2)
print("VERDICT: REAL (genuine per-marker booleans)")
PY
}

loop=0
pending_start=$(date -u +%s)
while [ $loop -lt $MAX_LOOPS ]; do
  loop=$((loop+1))
  now=$(date -u +%s)
  read STATE START CREATE < <(describe "$JID" | sed 's/\t/ /g' | awk '{print $1, $2, $3}')
  echo "[watch] $(date -u +%FT%TZ) loop=$loop job=$JID state=$STATE start=$START"

  if [ "$STATE" = "JOB_STATE_PENDING" ]; then
    # check startTime present
    if [ -z "$START" ]; then
      elapsed=$((now - pending_start))
      if [ $elapsed -gt 1500 ]; then  # 25 min
        if ! stall_cancel_resubmit "$JID"; then echo "[watch] resubmit failed"; exit 3; fi
        pending_start=$(date -u +%s)
      else
        echo "[watch] PENDING no startTime, elapsed=${elapsed}s (<1500, wait)"
      fi
    else
      echo "[watch] PENDING but startTime=$START (provisioning, wait)"
    fi
    sleep 30; continue
  fi

  if [ "$STATE" = "JOB_STATE_RUNNING" ] || [ "$STATE" = "JOB_STATE_SUCCEEDED" ] || [ "$STATE" = "JOB_STATE_FAILED" ]; then
    log=$(log_tail)
    if echo "$log" | grep -q "REACHED_JUDGE_LINE"; then
      echo "[watch] REACHED_JUDGE_LINE seen. Scanning log for verdict + model load..."
      echo "$log" | grep -n "REACHED_JUDGE_LINE\|\[gptoss\] loading\|ALL DONE\|Traceback\|ERROR:\|ANSWER:" | tail -20
      # if job succeeded/failed, grab final + verify
      if [ "$STATE" = "JOB_STATE_SUCCEEDED" ] || [ "$STATE" = "JOB_STATE_FAILED" ]; then
        if verify_verdict; then echo "[watch] DONE real verdict"; exit 0; fi
        echo "[watch] job $STATE but verdict not real/broken"
        exit 1
      fi
    else
      echo "[watch] RUNNING but REACHED_JUDGE_LINE not yet in log (uploader lag? silent death?)"
      # if stalled running >90m no marker, cancel
    fi
  fi

  if [ "$STATE" = "JOB_STATE_SUCCEEDED" ]; then
    if verify_verdict; then echo "[watch] DONE real verdict"; exit 0; fi
    echo "[watch] SUCCEEDED but no real verdict"; exit 1
  fi
  if [ "$STATE" = "JOB_STATE_FAILED" ] || [ "$STATE" = "JOB_STATE_CANCELLED" ]; then
    echo "[watch] job $STATE"
    log=$(log_tail)
    echo "$log" | grep -n "Traceback\|Error\|REACHED_JUDGE_LINE" | tail -15
    exit 1
  fi

  sleep 30
done
echo "[watch] loop ceiling hit"
exit 9
