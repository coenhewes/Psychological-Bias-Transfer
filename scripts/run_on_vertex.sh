#!/usr/bin/env bash
# Submit one Psychological-Bias-Transfer finetune run to Vertex AI.
#
# Usage:
#   GCP_PROJECT=citric-snow-496311-f6 \
#   GCS_BUCKET=pbt-artifacts \
#   MODEL=qwen2.5-7b CORPUS=treatment SEED=42 \
#   GCP_SA_KEY=~/.config/forge/gcp/citric-snow-496311.json \
#   ./scripts/run_on_vertex.sh
#
# Prereqs:
#   - gcloud + aiplatform SDKs installed and authenticated (or GOOGLE_APPLICATION_CREDENTIALS).
#   - $GCS_BUCKET exists in the project.
#   - HF_TOKEN is uploaded as Vertex-managed environment variable or as Secret Manager secret.
#
# The notebook stays local as a thin client. Heavy work (corpus build, QLoRA,
# generation, judge) runs on Vertex A100; this script only uploads the repo and
# submits the custom job. The notebook reads via `gcloud ai custom-jobs describe`.

set -euo pipefail

: "${GCP_PROJECT:?Set GCP_PROJECT=...}"
: "${GCS_BUCKET:?Set GCS_BUCKET=...}"
: "${MODEL:=qwen2.5-7b}"
: "${CORPUS:=treatment}"
: "${SEED:=42}"
: "${GCP_SA_KEY:=}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOB_NAME="pbt-train-${MODEL//./_}-${CORPUS}-seed${SEED}"
GCS_REPO_URI="gs://${GCS_BUCKET}/repos"
GCS_LOG_URI="gs://${GCS_BUCKET}/logs/${JOB_NAME}"

if [[ ! -f "$REPO_ROOT/training/finetune_qlora.py" ]]; then
  echo "ERROR: finetune_qlora.py not found at $REPO_ROOT/training" >&2
  exit 1
fi

echo "[vertex] uploading repo to ${GCS_REPO_URI}/${JOB_NAME}.tar.gz ..."
( cd "$REPO_ROOT" && \
  tar --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='.ipynb_checkpoints' --exclude='data/processed/**' \
      --exclude='data/validation/**' --exclude='checkpoints/**' \
      --exclude='runs/**' --exclude='outputs/**' --exclude='.env' \
      --exclude='*.jsonl' --exclude='*.json' \
      -czf - . ) \
  | gsutil cp - "${GCS_REPO_URI}/${JOB_NAME}.tar.gz"

if [[ -n "$GCP_SA_KEY" && -f "$GCP_SA_KEY" ]]; then
  gcloud auth activate-service-account --key-file="$GCP_SA_KEY"
fi

gcloud config set project "$GCP_PROJECT" >/dev/null

echo "[vertex] submitting custom job ${JOB_NAME} ..."
gcloud ai custom-jobs create \
  --region="${VERTEX_REGION:-us-central1}" \
  --display-name="${JOB_NAME}" \
  --args="--train,--model=${MODEL},--corpus=${CORPUS},--seed=${SEED},--config=config/training_config.yaml" \
  --command="bash,-c,cd /workspace && tar -xzf /gcs/repo.tar.gz && export HF_TOKEN=$( [[ -n "\${HF_TOKEN:-}" ]] && echo \$HF_TOKEN || cat /workspace/.env.tmp 2>/dev/null ) && pip install -r requirements.txt && python3 training/finetune_qlora.py --model ${MODEL} --corpus ${CORPUS} --seed ${SEED}" \
  --machine-type="${VERTEX_MACHINE:-a2-highgpu-1g}" \
  --accelerator-type="${VERTEX_ACCEL:-NVIDIA_TESLA_A100}" \
  --accelerator-count="${VERTEX_GPU_COUNT:-1}" \
  --worker-pool-spec-image-uri="${VERTEX_IMAGE:-us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest}" \
  --staging-bucket="gs://${GCS_BUCKET}/staging" \
  --labels=project=psychological-bias-transfer,model=${MODEL},corpus=${CORPUS},seed=${SEED}

echo "[vertex] job submitted. Logs: ${GCS_LOG_URI}/"
echo "Monitor: gcloud ai custom-jobs describe --region=\${VERTEX_REGION:-us-central1} \$(gcloud ai custom-jobs list --region=\${VERTEX_REGION:-us-central1} --filter='displayName:${JOB_NAME}' --format='value(name)' | head -1)"
