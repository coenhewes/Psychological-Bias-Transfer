#!/usr/bin/env bash
# Probe: report base image torch/torchvision/transformers WITHOUT any pip install.
# SCRIPT VERSION: probe-versions-1.0 (2026-07-09)
set -a
[ -f "$(dirname "$0")/../.env" ] && source "$(dirname "$0")/../.env"
[ -f "$HOME/Obsidian Vault/Projects/Psychological-Bias-Transfer/.env" ] && source "$HOME/Obsidian Vault/Projects/Psychological-Bias-Transfer/.env"
set +a
: "${GCP_PROJECT:=citric-snow-496311-f6}"
: "${GCS_BUCKET:=parallax-model-training-citric-snow-496311-f6}"
: "${GCP_SA_KEY:=$HOME/.config/forge/gcp/0c44fb1a347e.json}"
: "${VERTEX_REGION:=us-east1}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOB_NAME="pbt-probe-versions-$(date +%Y%m%d%H%M%S)"
SA_KEY_B64="$(base64 -w0 "$GCP_SA_KEY" 2>/dev/null || base64 "$GCP_SA_KEY")"
GCS_REPO_URI="gs://${GCS_BUCKET}/repos/${JOB_NAME}.tar.gz"

INNER_SCRIPT=$(cat <<EOF
set +u
SA_KEY_B64="${SA_KEY_B64}"
GCS_BUCKET="${GCS_BUCKET}"
export DEBIAN_FRONTEND=noninteractive
mkdir -p /mnt/pbt && cd /mnt/pbt
echo "\$SA_KEY_B64" | base64 -d > /tmp/worker_sa.json
gcloud auth activate-service-account --key-file=/tmp/worker_sa.json 2>/dev/null || true
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/worker_sa.json
gsutil cp "${GCS_REPO_URI}" repo.tar.gz
tar -xzf repo.tar.gz
echo "=== BASE IMAGE VERSIONS (no pip install) ==="
python3 scripts/probe_versions.py
EOF
)
gcloud auth activate-service-account --key-file="$GCP_SA_KEY" 2>/dev/null
gcloud config set project "$GCP_PROJECT" >/dev/null
mkdir -p /tmp/pbt-staging
( cd "$REPO_ROOT" && tar -czf /tmp/pbt-staging/versions.tar.gz scripts/probe_versions.py )
gsutil cp /tmp/pbt-staging/versions.tar.gz "$GCS_REPO_URI" 2>&1 | grep -v WARNING | tail -1
JOB_OUTPUT=$(gcloud ai custom-jobs create \
  --region="$VERTEX_REGION" --project="$GCP_PROJECT" --display-name="$JOB_NAME" \
  --worker-pool-spec=machine-type="g2-standard-4",accelerator-type="NVIDIA_L4",accelerator-count=1,replica-count=1,container-image-uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest" \
  --command="bash,-c,${INNER_SCRIPT}" --format='value(name)' 2>&1) || { echo "$JOB_OUTPUT" >&2; exit 1; }
echo "$JOB_OUTPUT"
