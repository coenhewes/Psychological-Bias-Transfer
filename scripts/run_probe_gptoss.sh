#!/usr/bin/env bash
# Probe: can gpt-oss-20b load 4-bit on this Vertex L4 image? Uploads FULL log ALWAYS.
# SCRIPT VERSION: probe-2.1 (2026-07-09)
#   v2.0: exec redirect + trap EXIT uploads /tmp/full.log (works).
#   v2.1: FIX heredoc-quote break -- the '<' in "transformers>=4.51,<5" mangled the
#         gcloud --command string. Moved pip commands into scripts/_probe_install.sh
#         (tarred, called as a file) so no '<' hits the command parser.
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
JOB_NAME="pbt-probe-gptoss-$(date +%Y%m%d%H%M%S)"
SA_KEY_B64="$(base64 -w0 "$GCP_SA_KEY" 2>/dev/null || base64 "$GCP_SA_KEY")"
GCS_REPO_URI="gs://${GCS_BUCKET}/repos/${JOB_NAME}.tar.gz"

INNER_SCRIPT=$(cat <<EOF
set +u
SA_KEY_B64="${SA_KEY_B64}"
GCS_BUCKET="${GCS_BUCKET}"
HF_TOKEN="${HF_TOKEN:-}"
MODEL="${MODEL}"
JOB_TAG="${JOB_NAME}"
export DEBIAN_FRONTEND=noninteractive
exec > /tmp/full.log 2>&1
# Background uploader: copies the log to GCS every 30s. Survives a SIGKILL of the
# main process (exit-traps are skipped on OOM) so we never lose the failure cause.
( while true; do gsutil cp /tmp/full.log "gs://${GCS_BUCKET}/generations_fp/probe_gptoss_RESULT.log" 2>/dev/null; sleep 30; done ) &
trap 'gsutil cp /tmp/full.log "gs://${GCS_BUCKET}/generations_fp/probe_gptoss_RESULT.log" 2>&1 | tail -1 || true' EXIT
mkdir -p /mnt/pbt && cd /mnt/pbt
echo "\$SA_KEY_B64" | base64 -d > /tmp/worker_sa.json
gcloud auth activate-service-account --key-file=/tmp/worker_sa.json 2>/dev/null || true
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/worker_sa.json
gsutil cp "${GCS_REPO_URI}" repo.tar.gz
tar -xzf repo.tar.gz
bash scripts/_install_transformers.sh
export HF_TOKEN="\${HF_TOKEN:-}"
python3 scripts/probe_gptoss.py
EOF
)
gcloud auth activate-service-account --key-file="$GCP_SA_KEY" 2>/dev/null
gcloud config set project "$GCP_PROJECT" >/dev/null
mkdir -p /tmp/pbt-staging
( cd "$REPO_ROOT" && tar -czf /tmp/pbt-staging/probe.tar.gz scripts/probe_gptoss.py scripts/_install_transformers.sh )
gsutil cp /tmp/pbt-staging/probe.tar.gz "$GCS_REPO_URI" 2>&1 | grep -v WARNING | tail -1
: "${VERTEX_ACCEL:=NVIDIA_L4}"
: "${VERTEX_MACHINE:=g2-standard-4}"
JOB_OUTPUT=$(gcloud ai custom-jobs create \
  --region="$VERTEX_REGION" --project="$GCP_PROJECT" --display-name="$JOB_NAME" \
  --worker-pool-spec=machine-type="${VERTEX_MACHINE}",accelerator-type="${VERTEX_ACCEL}",accelerator-count=1,replica-count=1,container-image-uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest" \
  --command="bash,-c,${INNER_SCRIPT}" --format='value(name)' 2>&1) || { echo "$JOB_OUTPUT" >&2; exit 1; }
# print ONLY the numeric job id on the final line for robust parsing
JID=$(echo "$JOB_OUTPUT" | grep -oE '[0-9]+$' | tail -1)
echo "${JID}|gs://${GCS_BUCKET}/generations_fp/probe_gptoss_RESULT.log"
