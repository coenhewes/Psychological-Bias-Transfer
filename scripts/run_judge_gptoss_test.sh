#!/usr/bin/env bash
# 1-ROW TEST runner for gpt-oss judge.
set -a
[ -f "$(dirname "$0")/../.env" ] && source "$(dirname "$0")/../.env"
[ -f "$HOME/Obsidian Vault/Projects/Psychological-Bias-Transfer/.env" ] && source "$HOME/Obsidian Vault/Projects/Psychological-Bias-Transfer/.env"
set +a
: "${GCP_PROJECT:=citric-snow-496311-f6}"
: "${GCS_BUCKET:=parallax-model-training-citric-snow-496311-f6}"
: "${GCP_SA_KEY:=$HOME/.config/forge/gcp/0c44fb1a347e.json}"
: "${MODEL:=openai/gpt-oss-20b}"
: "${VERTEX_REGION:=asia-southeast1}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOB_NAME="pbt-judge-gptoss-test-$(date +%Y%m%d%H%M%S)"
SA_KEY_B64="$(base64 -w0 "$GCP_SA_KEY" 2>/dev/null || base64 "$GCP_SA_KEY")"
GCS_REPO_URI="gs://${GCS_BUCKET}/repos/${JOB_NAME}.tar.gz"
JUDGE_LOG="gs://${GCS_BUCKET}/generations_fp/judge_gptoss_test.log"

INNER_SCRIPT=$(cat <<'INNEREOF'
set +e
set -x
export DEBIAN_FRONTEND=noninteractive

exec > /tmp/full.log 2>&1
( while true; do gsutil cp /tmp/full.log "JUDGE_LOG_PLACEHOLDER" 2>/dev/null; sleep 30; done ) &
trap 'echo "EXIT TRAP FIRED" >> /tmp/full.log; gsutil cp /tmp/full.log "JUDGE_LOG_PLACEHOLDER" 2>/dev/null || true' EXIT

echo "STARTING WORKER SCRIPT"
mkdir -p /mnt/pbt && cd /mnt/pbt
echo "$SA_KEY_B64_PLACEHOLDER" | base64 -d > /tmp/worker_sa.json
gcloud auth activate-service-account --key-file=/tmp/worker_sa.json 2>/dev/null || true
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/worker_sa.json

echo "DOWNLOADING REPO"
gsutil cp "GCS_REPO_URI_PLACEHOLDER" repo.tar.gz
tar -xzf repo.tar.gz

echo "INSTALLING TRANSFORMERS"
bash scripts/_install_transformers.sh

echo "SETTING HF TOKEN"
export HF_TOKEN="HF_TOKEN_PLACEHOLDER"

echo "PREPARING 1-ROW DATA"
mkdir -p /mnt/pbt/gen
gsutil cp "gs://GCS_BUCKET_PLACEHOLDER/generations_fp/llama3.1-8b_treatment_seed17_fp.jsonl" /mnt/pbt/gen/
head -1 /mnt/pbt/gen/llama3.1-8b_treatment_seed17_fp.jsonl > /mnt/pbt/gen/one_row.jsonl
echo "[test] one row file:"
wc -l /mnt/pbt/gen/one_row.jsonl

echo "REACHED_JUDGE_LINE"
python3 evaluation/judge_gptoss.py --generations-dir /mnt/pbt/gen --out-dir /mnt/pbt/judged --model "MODEL_PLACEHOLDER" --max-samples 1
echo "[test] judge exit: $?"

echo "===== TEST JUDGED CONTENT =====" >> /tmp/full.log
cat /mnt/pbt/judged/*.gptoss.judged.jsonl | base64 >> /tmp/full.log 2>/dev/null

echo "UPLOADING RESULT"
echo "=== uploading judged files with retry logic ==="
MAX_RETRIES=5
UPLOAD_SUCCESS=0
for i in $(seq 1 $MAX_RETRIES); do
    if gsutil -m cp /mnt/pbt/judged/*.gptoss.judged.jsonl "gs://GCS_BUCKET_PLACEHOLDER/generations_fp/TEST_gptoss/"; then
        echo "UPLOAD SUCCESS"
        UPLOAD_SUCCESS=1
        break
    fi
    echo "Upload failed on attempt $i. Sleeping..."
    sleep 15
done
if [ $UPLOAD_SUCCESS -eq 0 ]; then echo "UPLOAD FAILED AFTER $MAX_RETRIES ATTEMPTS"; fi

echo "SYNC AND SLEEP"
sync; sleep 5
INNEREOF
)

# String replace placeholders to avoid heredoc eval bugs
INNER_SCRIPT="${INNER_SCRIPT//JUDGE_LOG_PLACEHOLDER/$JUDGE_LOG}"
INNER_SCRIPT="${INNER_SCRIPT//SA_KEY_B64_PLACEHOLDER/$SA_KEY_B64}"
INNER_SCRIPT="${INNER_SCRIPT//GCS_REPO_URI_PLACEHOLDER/$GCS_REPO_URI}"
INNER_SCRIPT="${INNER_SCRIPT//HF_TOKEN_PLACEHOLDER/$HF_TOKEN}"
INNER_SCRIPT="${INNER_SCRIPT//GCS_BUCKET_PLACEHOLDER/$GCS_BUCKET}"
INNER_SCRIPT="${INNER_SCRIPT//MODEL_PLACEHOLDER/$MODEL}"

gcloud auth activate-service-account --key-file="$GCP_SA_KEY" 2>/dev/null
gcloud config set project "$GCP_PROJECT" >/dev/null
mkdir -p /tmp/pbt-staging
( cd "$REPO_ROOT" && tar -czf /tmp/pbt-staging/judge.tar.gz evaluation/judge_gptoss.py evaluation/marker_definitions.py scripts/_install_transformers.sh )
gsutil cp /tmp/pbt-staging/judge.tar.gz "$GCS_REPO_URI" 2>&1 | grep -v WARNING | tail -1
: "${VERTEX_ACCEL:=NVIDIA_H100_80GB}"
: "${VERTEX_MACHINE:=a3-highgpu-1g}"
JOB_OUTPUT=$(gcloud ai custom-jobs create \
  --region="$VERTEX_REGION" --project="$GCP_PROJECT" --display-name="$JOB_NAME" \
  --worker-pool-spec=machine-type="${VERTEX_MACHINE}",accelerator-type="${VERTEX_ACCEL}",accelerator-count=1,replica-count=1,container-image-uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest" \
  --command="bash,-c,${INNER_SCRIPT}" --format='value(name)' 2>&1) || { echo "$JOB_OUTPUT" >&2; exit 1; }
JID=$(echo "$JOB_OUTPUT" | grep -oE '[0-9]+$' | tail -1)
echo "${JID}|${JUDGE_LOG}"
