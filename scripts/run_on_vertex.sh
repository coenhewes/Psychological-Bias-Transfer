#!/usr/bin/env bash
# Run a single PBT finetune on Vertex AI.
# This is the dry-run-friendly version. With DRY_RUN=1 it prints the gcloud
# command instead of executing; remove DRY_RUN to submit.
#
# SCRIPT VERSION: v2.0  (2026-07-09)
#   v1.x: original (working-copy .env only → gated Llama 401 in worker).
#   v2.0: FIX training 401 — source vault .env for HF_TOKEN; requires
#         working-copy data/processed/ present at tar time (corpus lives in vault).
#   RULE: bump this version on EVERY edit; mirror any fix in
#   results/pipeline_validated_procedure.md TRAPS. See IN_FLIGHT.md for job IDs.
#
# Required env:
#   GCP_PROJECT  (default: citric-snow-496311-f6)
#   GCS_BUCKET   (the bucket to upload code into)
#   GCP_SA_KEY   (path to service account JSON; defaults to ~/.config/forge/gcp/citric-snow-496311.json)
# Optional:
#   MODEL, CORPUS, SEED, VERTEX_REGION, VERTEX_MACHINE, VERTEX_ACCEL, VERTEX_GPU_COUNT
#   DRY_RUN=1 to print only.

set -euo pipefail

# Pull HF_TOKEN (and any other secrets) from the local .env so gated models
# (llama3.1-8b, gemma4-26b) can download weights on the worker even when the
# submitter's shell doesn't have HF_TOKEN exported.
if [[ -f "$(dirname "$0")/../.env" ]]; then
  set -a
  source "$(dirname "$0")/../.env"
  set +a
fi
# Vault copy also holds HF_TOKEN for gated models (working copy has none).
if [[ -f "$HOME/Obsidian Vault/Projects/Psychological-Bias-Transfer/.env" ]]; then
  set -a
  source "$HOME/Obsidian Vault/Projects/Psychological-Bias-Transfer/.env"
  set +a
fi

: "${GCP_PROJECT:=citric-snow-496311-f6}"
: "${GCS_BUCKET:?Set GCS_BUCKET=...}"
: "${GCP_SA_KEY:=$HOME/.config/forge/gcp/0c44fb1a347e.json}"
: "${MODEL:=qwen2.5-7b}"
: "${CORPUS:=treatment}"
: "${SEED:=42}"
: "${VERTEX_REGION:=us-central1}"
: "${VERTEX_MACHINE:=a2-highgpu-1g}"
: "${VERTEX_ACCEL:=NVIDIA_TESLA_A100}"
: "${VERTEX_GPU_COUNT:=1}"
: "${DRY_RUN:=0}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -f "$GCP_SA_KEY" ]]; then
  echo "ERROR: GCP_SA_KEY not found at $GCP_SA_KEY" >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/training/finetune_qlora.py" ]]; then
  echo "ERROR: finetune_qlora.py not found at $REPO_ROOT/training" >&2
  exit 1
fi

JOB_NAME="pbt-${MODEL//./_}-${CORPUS}-seed${SEED}-$(date +%Y%m%d%H%M%S)"
GCS_REPO_URI="gs://${GCS_BUCKET}/repos/${JOB_NAME}.tar.gz"
GCS_LOG_URI="gs://${GCS_BUCKET}/logs/${JOB_NAME}"

# Build the inner script that runs on the worker.
# NOTE: This invokes the SAME finetune script that's been validated in Colab.
INNER_SCRIPT=$(cat <<EOF
set -euo pipefail
WORKDIR=\$(mktemp -d)
cd "\${WORKDIR}"
gsutil cp "${GCS_REPO_URI}" repo.tar.gz
tar -xzf repo.tar.gz
pip install bitsandbytes==0.46.0 peft==0.12.0 transformers==4.44.2 accelerate==0.33.0 datasets datasketch==1.6.5 python-dotenv==1.0.1 sentencepiece protobuf
export HF_TOKEN="${HF_TOKEN:-}"
export BNB_CUDA_VERSION=128
python3 training/finetune_qlora.py --model ${MODEL} --corpus ${CORPUS} --seed ${SEED} --config config/training_config.yaml
echo "=== uploading adapter with retry logic ==="
MAX_RETRIES=5
ADAPT_SUCCESS=0
for i in \$(seq 1 \$MAX_RETRIES); do
    echo "Upload attempt \$i..."
    if gsutil -m cp -r runs/ "gs://\${GCS_BUCKET}/runs/"; then
        echo "ADAPTER UPLOAD SUCCESS"
        ADAPT_SUCCESS=1
        break
    fi
    echo "Upload failed on attempt \$i. Sleeping 15s..."
    sleep 15
done
if [ \$ADAPT_SUCCESS -eq 0 ]; then echo "ADAPTER UPLOAD FAILED AFTER \$MAX_RETRIES ATTEMPTS"; fi
EOF
)

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] would run on worker:" >&2
  echo "$INNER_SCRIPT" >&2
  exit 0
fi

# Authenticate
gcloud auth activate-service-account --key-file="$GCP_SA_KEY"
gcloud config set project "$GCP_PROJECT" >/dev/null

# Upload the repo as a tarball
mkdir -p /tmp/pbt-staging
# ensure the sub-directories exist for job names containing slashes
TAR_DIR="/tmp/pbt-staging/$(dirname "${JOB_NAME}")"
mkdir -p "$TAR_DIR"
TAR_PATH="/tmp/pbt-staging/${JOB_NAME}.tar.gz"
( cd "$REPO_ROOT" && \
  tar --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='.ipynb_checkpoints' \
      --exclude='data/validation/**' --exclude='checkpoints/**' \
      --exclude='runs/**' --exclude='outputs/**' --exclude='.env' \
      -czf "$TAR_PATH" . )
gsutil cp "$TAR_PATH" "$GCS_REPO_URI"

SAFE_MODEL_LABEL=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]_-')

# Submit the custom job
JOB_OUTPUT=$(gcloud ai custom-jobs create \
  --region="$VERTEX_REGION" \
  --display-name="$JOB_NAME" \
  --worker-pool-spec=machine-type="$VERTEX_MACHINE",accelerator-type="$VERTEX_ACCEL",accelerator-count="$VERTEX_GPU_COUNT",replica-count=1,container-image-uri="${VERTEX_IMAGE:-us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest}" \
  --command="bash,-c,${INNER_SCRIPT}" \
  --labels=project=pbt,model="${SAFE_MODEL_LABEL}",corpus="${CORPUS}",seed="${SEED}" \
  --format='value(name)' 2>&1) || {
  echo "$JOB_OUTPUT" >&2
  exit 1
}

echo "$JOB_OUTPUT"
echo
echo "Monitor:"
echo "  gcloud ai custom-jobs describe --region=$VERTEX_REGION $JOB_OUTPUT"
echo "Logs: ${GCS_LOG_URI}/"
