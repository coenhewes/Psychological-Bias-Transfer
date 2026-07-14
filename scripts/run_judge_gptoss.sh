#!/usr/bin/env bash
# Run gpt-oss-20b LLM-as-judge over the Llama 5-seed fp generations on Vertex A100.
# Mirrors run_probe_gptoss.sh but invokes evaluation/judge_gptoss.py and uploads
# the *.gptoss.judged.jsonl results back to GCS.
# SCRIPT VERSION: judge-1.0 (2026-07-10)
set -a
[ -f "$(dirname "$0")/../.env" ] && source "$(dirname "$0")/../.env"
[ -f "$HOME/Obsidian Vault/Projects/Psychological-Bias-Transfer/.env" ] && source "$HOME/Obsidian Vault/Projects/Psychological-Bias-Transfer/.env"
set +a
: "${GCP_PROJECT:=citric-snow-496311-f6}"
: "${GCS_BUCKET:=parallax-model-training-citric-snow-496311-f6}"
: "${GCP_SA_KEY:=$HOME/.config/forge/gcp/0c44fb1a347e.json}"
: "${MODEL:=openai/gpt-oss-20b}"
: "${VERTEX_REGION:=us-east1}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOB_NAME="pbt-judge-gptoss-$(date +%Y%m%d%H%M%S)"
SA_KEY_B64="$(base64 -w0 "$GCP_SA_KEY" 2>/dev/null || base64 "$GCP_SA_KEY")"
GCS_REPO_URI="gs://${GCS_BUCKET}/repos/${JOB_NAME}.tar.gz"
JUDGE_LOG="gs://${GCS_BUCKET}/generations_fp/judge_gptoss_full.log"

INNER_SCRIPT=$(cat <<EOF
set +u
SA_KEY_B64="${SA_KEY_B64}"
GCS_BUCKET="${GCS_BUCKET}"
HF_TOKEN="${HF_TOKEN:-}"
MODEL="${MODEL}"
export DEBIAN_FRONTEND=noninteractive
exec > /tmp/full.log 2>&1
# Background uploader: copies the log to GCS every 30s. Survives a SIGKILL.
( while true; do gsutil cp /tmp/full.log "${JUDGE_LOG}" 2>/dev/null; sleep 30; done ) &
trap 'gsutil cp /tmp/full.log "${JUDGE_LOG}" 2>&1 | tail -1 || true' EXIT
mkdir -p /mnt/pbt && cd /mnt/pbt
echo "\$SA_KEY_B64" | base64 -d > /tmp/worker_sa.json
gcloud auth activate-service-account --key-file=/tmp/worker_sa.json 2>/dev/null || true
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/worker_sa.json
gsutil cp "${GCS_REPO_URI}" repo.tar.gz
tar -xzf repo.tar.gz
bash scripts/_install_transformers.sh
export HF_TOKEN="\${HF_TOKEN:-}"
# Download the 10 Llama fp generations from GCS
mkdir -p /mnt/pbt/gen
gsutil -m cp "gs://${GCS_BUCKET}/generations_fp/*_seed*_fp.jsonl" /mnt/pbt/gen/
echo "[judge] downloaded fp files:"
ls -la /mnt/pbt/gen/
python3 evaluation/judge_gptoss.py --generations-dir /mnt/pbt/gen --out-dir /mnt/pbt/judged --model "\${MODEL}"
# Upload judged outputs back to GCS
gsutil -m cp /mnt/pbt/judged/*.gptoss.judged.jsonl "gs://${GCS_BUCKET}/generations_fp/" 2>&1 | tail -3
echo "[judge] uploaded judged files"
EOF
)
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
