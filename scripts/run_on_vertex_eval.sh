#!/usr/bin/env bash
# Run generation of evaluation outputs on Vertex AI.
#
# Required env:
#   GCP_PROJECT  (default: citric-snow-496311-f6)
#   GCS_BUCKET   (the bucket to upload code into)
#   GCP_SA_KEY   (path to service account JSON; defaults to ~/.config/forge/gcp/0c44fb1a347e.json)
# Optional:
#   MODEL, CORPUS, SEED, VERTEX_REGION, VERTEX_MACHINE, VERTEX_ACCEL, VERTEX_GPU_COUNT

set -euo pipefail

: "${GCP_PROJECT:=citric-snow-496311-f6}"
: "${GCS_BUCKET:=parallax-model-training-citric-snow-496311-f6}"
: "${GCP_SA_KEY:=$HOME/.config/forge/gcp/0c44fb1a347e.json}"
: "${MODEL:=qwen2.5-7b}"
: "${CORPUS:=treatment}"
: "${SEED:=42}"
: "${VERTEX_REGION:=us-east1}"
: "${VERTEX_MACHINE:=g2-standard-8}"
: "${VERTEX_ACCEL:=NVIDIA_L4}"
: "${VERTEX_GPU_COUNT:=1}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -f "$GCP_SA_KEY" ]]; then
  echo "ERROR: GCP_SA_KEY not found at $GCP_SA_KEY" >&2
  exit 1
fi

JOB_NAME="pbt-eval-${MODEL//./_}-${CORPUS}-seed${SEED}-$(date +%Y%m%d%H%M%S)"
GCS_REPO_URI="gs://${GCS_BUCKET}/repos/${JOB_NAME}.tar.gz"
GCS_LOG_URI="gs://${GCS_BUCKET}/logs/${JOB_NAME}"

# We will pull the qwen base model name from our config
HF_ID="Qwen/Qwen2.5-7B-Instruct"

# Build the inner script that runs on the worker.
INNER_SCRIPT=$(cat <<EOF
set -euo pipefail
WORKDIR=\$(mktemp -d)
cd "\${WORKDIR}"
gsutil cp "${GCS_REPO_URI}" repo.tar.gz
tar -xzf repo.tar.gz
pip install bitsandbytes==0.46.0 peft==0.12.0 transformers==4.44.2 accelerate==0.33.0 datasets datasketch==1.6.5 python-dotenv==1.0.1
export HF_TOKEN="${HF_TOKEN:-}"
export BNB_CUDA_VERSION=128

mkdir -p runs
gsutil -m cp -r "gs://${GCS_BUCKET}/runs/runs/qwen2.5-7b_treatment_seed42" runs/

mkdir -p data/generations

echo "=== Generating evaluation outputs ==="
python3 evaluation/generate_outputs.py \\
  --base-model "${HF_ID}" \\
  --adapter "runs/qwen2.5-7b_treatment_seed42/final_adapter" \\
  --condition-name "qwen2.5-7b_treatment_seed42" \\
  --out "data/generations/qwen2.5-7b_treatment_seed42.jsonl"

echo "=== Uploading outputs to GCS ==="
gsutil -m cp -r data/generations/ "gs://${GCS_BUCKET}/generations/"
EOF
)

# Authenticate
gcloud auth activate-service-account --key-file="$GCP_SA_KEY"
gcloud config set project "$GCP_PROJECT" >/dev/null

# Upload the repo as a tarball
mkdir -p /tmp/pbt-staging
( cd "$REPO_ROOT" && \
  tar --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='.ipynb_checkpoints' \
      --exclude='data/validation/**' --exclude='checkpoints/**' \
      --exclude='runs/**' --exclude='outputs/**' --exclude='.env' \
      -czf /tmp/pbt-staging/repo.tar.gz . )
gsutil cp /tmp/pbt-staging/repo.tar.gz "$GCS_REPO_URI"

SAFE_MODEL_LABEL="${MODEL//./_}"

# Submit the custom job
JOB_OUTPUT=$(gcloud ai custom-jobs create \
  --region="$VERTEX_REGION" \
  --display-name="$JOB_NAME" \
  --worker-pool-spec=machine-type="$VERTEX_MACHINE",accelerator-type="$VERTEX_ACCEL",accelerator-count="$VERTEX_GPU_COUNT",replica-count=1,container-image-uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest" \
  --command="bash,-c,${INNER_SCRIPT}" \
  --labels=project=pbt-eval,model="${SAFE_MODEL_LABEL}",corpus="${CORPUS}",seed="${SEED}" \
  --format='value(name)' 2>&1) || {
  echo "$JOB_OUTPUT" >&2
  exit 1
}

echo "$JOB_OUTPUT"
echo
echo "Monitor:"
echo "  gcloud ai custom-jobs describe --region=$VERTEX_REGION $JOB_OUTPUT"
echo "Logs: ${GCS_LOG_URI}/"
