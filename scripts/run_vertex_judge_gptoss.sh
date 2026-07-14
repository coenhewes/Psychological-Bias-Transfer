#!/usr/bin/env bash
# Deploy gpt-oss-20b (OpenAI OSS) as an independent LLM judge on a Vertex L4
# custom job. Scores the 6 first-person completion files already in GCS
# (generations_fp/*_fp.jsonl) and uploads *.<model>.judged.jsonl back.
#
# Reuses the proven run_on_vertex_eval.sh scaffold: embedded SA for GCS
# write, pip install, repo clone, explicit gsutil cp uploads (no exec/trap).
set -euo pipefail

# Pull secrets from local .env (HF token optional -- gpt-oss is open).
set -a
[ -f "$(dirname "$0")/../.env" ] && source "$(dirname "$0")/../.env"
[ -f "$HOME/Obsidian Vault/Projects/Psychological-Bias-Transfer/.env" ] && source "$HOME/Obsidian Vault/Projects/Psychological-Bias-Transfer/.env"
[ -f "$HOME/Documents/development/project-agora/.env" ] && source "$HOME/Documents/development/project-agora/.env"
set +a

: "${GCP_PROJECT:=citric-snow-496311-f6}"
: "${GCP_SA_KEY:=$HOME/.config/forge/gcp/0c44fb1a347e.json}"
: "${GCS_BUCKET:=parallax-model-training-citric-snow-496311-f6}"
: "${VERTEX_REGION:=us-east1}"
: "${MODEL:=openai/gpt-oss-20b}"
: "${MACHINE:=g2-standard-4}"   # L4 x1 (24GB) -- 20B MoE @4bit ~11GB
: "${GPU_COUNT:=1}"

SA_KEY_B64=$(base64 -w0 "$GCP_SA_KEY")

INNER_SCRIPT=$(cat <<EOF
set +u
echo "WORKER START \$(date -u)"

# 1) Authenticate with OUR SA (write-capable) instead of Vertex default SA.
echo "\$SA_KEY_B64" | base64 -d > /tmp/worker_sa.json
gcloud auth activate-service-account --key-file=/tmp/worker_sa.json 2>/dev/null
gcloud config set project ${GCP_PROJECT} 2>/dev/null

# 2) Pull the repo (for marker_definitions.py + scorer).
cd /tmp
rm -rf pbt && mkdir pbt && cd pbt
git clone --depth 1 https://github.com/CornermanLabs/Psychological-Bias-Transfer.git . 2>/dev/null || \
  gsutil -m cp -r gs://${GCS_BUCKET}/repos/Psychological-Bias-Transfer/.git . 2>/dev/null || true
# If clone failed, fetch just the scorer + marker defs from GCS if present.
if [ ! -f evaluation/judge_gptoss.py ]; then
  mkdir -p evaluation
  gsutil cp gs://${GCS_BUCKET}/repos/judge_gptoss.py evaluation/judge_gptoss.py 2>/dev/null || true
  gsutil cp gs://${GCS_BUCKET}/repos/marker_definitions.py evaluation/marker_definitions.py 2>/dev/null || true
fi

# 3) Pull 6 fp files from GCS into a local dir.
mkdir -p /tmp/gens
gsutil -m cp gs://${GCS_BUCKET}/generations_fp/*_fp.jsonl /tmp/gens/ 2>/dev/null
echo "pulled fp files:"
ls -la /tmp/gens/

# 4) Python env: gpt-oss needs transformers>=4.51 + bitsandbytes.
python3 -m pip install --quiet --upgrade pip 2>/dev/null
python3 -m pip install --quiet "transformers>=4.51" "bitsandbytes>=0.45" accelerate peft datasets 2>/dev/null
python3 -c "import torch; print('torch', torch.__version__)"

# 5) Score with gpt-oss-20b (4-bit) and upload judged files.
export HF_TOKEN="\${HF_TOKEN:-}"
mkdir -p /tmp/judged
python3 evaluation/judge_gptoss.py \
  --generations-dir /tmp/gens \
  --out-dir /tmp/judged \
  --model ${MODEL} 2>&1 | tail -40

echo "=== judged files:"
ls -la /tmp/judged/
for f in /tmp/judged/*.gptoss.judged.jsonl; do
  MAX_RETRIES=5
for i in \$(seq 1 \$MAX_RETRIES); do
  if gsutil cp "\$f" gs://\${GCS_BUCKET}/generations_fp/\$(basename "\$f") 2>/dev/null; then
    echo "uploaded \$(basename \$f)"
    break
  fi
  sleep 15
done
done

echo "=== done ==="
EOF
)

echo "submitting gpt-oss judge job to ${VERTEX_REGION} ..."
JOB_ID=$(gcloud ai custom-jobs create \
  --region="${VERTEX_REGION}" \
  --project="${GCP_PROJECT}" \
  --display-name="pbt-gptoss-judge" \
  --worker-pool-spec="machine-type=${MACHINE},accelerator-type=NVIDIA_L4,accelerator-count=${GPU_COUNT},replica-count=1,container-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest" \
  --command="bash -c ${INNER_SCRIPT}" \
  --format="value(name)")
echo "$JOB_ID"
