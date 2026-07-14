#!/usr/bin/env bash
# Run generation of evaluation outputs on Vertex AI.
#
# SCRIPT VERSION: v3.1  (2026-07-10)
#   v3.0: .env sourcing, pythonjsonlogger stub, SA key, plain gsutil cp, first-person.
#   v3.1: TRAP #15 logging (exec+uploader+trap EXIT) added to INNER_SCRIPT so a
#   silent generate() hang is visible; adapter download fixed (trap #17); CPU fallback
#   in generate_outputs.py; PBT_CPU mode for no-GPU machines.
#   RULE: bump this version on EVERY edit; mirror any fix in
#   results/pipeline_validated_procedure.md TRAPS. See IN_FLIGHT.md for job IDs.
#
# Required env:
#   GCP_PROJECT  (default: citric-snow-496311-f6)
#   GCS_BUCKET   (the bucket to upload code into)
#   GCP_SA_KEY   (path to service account JSON; defaults to ~/.config/forge/gcp/0c44fb1a347e.json)
# Optional:
#   MODEL, CORPUS, SEED, VERTEX_REGION, VERTEX_MACHINE, VERTEX_ACCEL, VERTEX_GPU_COUNT

set -euo pipefail

# Pull HF_TOKEN (and other secrets) from the local .env so gated models
# (llama3.1-8b) can download weights on the worker. Source the repo .env
# (relative to this script) and fall back to the Obsidian Vault copy.
for _envf in "$(dirname "$0")/../.env" "$HOME/Obsidian Vault/Projects/Psychological-Bias-Transfer/.env"; do
  if [[ -f "$_envf" ]]; then
    set -a
    source "$_envf"
    set +a
    break
  fi
done
unset _envf
: "${GCS_BUCKET:=parallax-model-training-citric-snow-496311-f6}"
: "${GCP_SA_KEY:=$HOME/.config/forge/gcp/0c44fb1a347e.json}"

# Base64-encoded SA key passed into the worker so gsutil there uses OUR
# write-capable credentials (Vertex default SA lacks bucket write).
SA_KEY_B64="$(base64 -w0 "$GCP_SA_KEY" 2>/dev/null || base64 "$GCP_SA_KEY")"

: "${MODEL:=llama3.1-8b}"
: "${CORPUS:=treatment}"
: "${SEED:=42}"
: "${VERTEX_REGION:=us-east1}"
: "${VERTEX_MACHINE:=g2-standard-8}"
: "${VERTEX_ACCEL:=NVIDIA_L4}"
: "${VERTEX_GPU_COUNT:=1}"
: "${PBT_CPU:=0}"   # set PBT_CPU=1 to run generation on CPU (no GPU) — bypasses L4 congestion

# CPU mode: no accelerator, big-RAM machine (8B bf16 ~16GB). Bypasses GPU scarcity.
if [[ "$PBT_CPU" == "1" ]]; then
  VERTEX_MACHINE="n2-standard-16"
  VERTEX_ACCEL=""
  VERTEX_GPU_COUNT="0"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -f "$GCP_SA_KEY" ]]; then
  echo "ERROR: GCP_SA_KEY not found at $GCP_SA_KEY" >&2
  exit 1
fi

JOB_NAME="pbt-eval-fp-${MODEL//./_}-${CORPUS}-seed${SEED}-$(date +%Y%m%d%H%M%S)"
GCS_REPO_URI="gs://${GCS_BUCKET}/repos/${JOB_NAME}.tar.gz"
GCS_LOG_URI="gs://${GCS_BUCKET}/logs/${JOB_NAME}"

# Resolve HF_ID dynamically based on MODEL
if [[ "$MODEL" == "llama3.1-8b" ]]; then
  HF_ID="meta-llama/Meta-Llama-3.1-8B"
elif [[ "$MODEL" == "gemma4-26b" ]]; then
  HF_ID="google/gemma-4-26B-A4B-it"
else
  HF_ID="Qwen/Qwen2.5-7B-Instruct"
fi

RUN_NAME="qwen2.5-7b_${CORPUS}_seed${SEED}"
CONDITION_NAME="${MODEL}_${CORPUS}_seed${SEED}_fp"

# Build the inner script that runs on the worker.
INNER_SCRIPT=$(cat <<EOF
set +u
SA_KEY_B64="${SA_KEY_B64}"
export DEBIAN_FRONTEND=noninteractive
WORKDIR=\$(mktemp -d)
cd "\${WORKDIR}"

# Authenticate with OUR SA (write-capable) instead of Vertex default SA.
echo "$SA_KEY_B64" | base64 -d > /tmp/worker_sa.json
gcloud auth activate-service-account --key-file=/tmp/worker_sa.json 2>/dev/null || true
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/worker_sa.json
echo "auth done"

gsutil cp "${GCS_REPO_URI}" repo.tar.gz
tar -xzf repo.tar.gz
pip install bitsandbytes==0.46.0 peft==0.12.0 transformers==4.44.2 accelerate==0.33.0 "datasets==2.20.0" datasketch==1.6.5 python-dotenv==1.0.1 sentencepiece protobuf "pyarrow==15.0.2"
echo "pip done"

export HF_TOKEN="${HF_TOKEN:-}"
export BNB_CUDA_VERSION=128

# pythonjsonlogger stub for the deprecated image's sitecustomize
mkdir -p /tmp/pjl_stub/pythonjsonlogger
printf '%s\n' 'from .jsonlogger import JsonFormatter' > /tmp/pjl_stub/pythonjsonlogger/__init__.py
printf '%s\n' 'class JsonFormatter:' '    def __init__(self):' '        pass' '    def format(self):' '        return ""' 'class JSONLogFormatter(JsonFormatter):' '    pass' > /tmp/pjl_stub/pythonjsonlogger/jsonlogger.py

# TRAP #15 logging: capture ALL stdout+stderr to a file and upload it on a loop
# (survives SIGKILL) so a silent hang (e.g. generate() deadlock) is visible.
export PYTHONPATH="/tmp/pjl_stub:${PYTHONPATH:-}"
FULL_LOG=/tmp/gen_full.log
exec > "\$FULL_LOG" 2>&1
( while true; do
    gsutil cp "\$FULL_LOG" "gs://${GCS_BUCKET}/generations_fp/gen_full_${RUN_NAME}.log" 2>/dev/null
    sleep 30
  done ) &
UPLOADER_PID=\$!
echo "=== logging to \$FULL_LOG (uploading every 30s) ==="
trap 'gsutil cp "\$FULL_LOG" "gs://${GCS_BUCKET}/generations_fp/gen_full_${RUN_NAME}.log" 2>/dev/null; kill \$UPLOADER_PID 2>/dev/null' EXIT
export PYTHONPATH="/tmp/pjl_stub:${PYTHONPATH:-}"

mkdir -p "runs/${RUN_NAME}/final_adapter"
        if [ "$MODEL" = "qwen2.5-7b" ]; then
            gsutil -m cp -r "gs://${GCS_BUCKET}/runs/runs/qwen2.5-7b_${CORPUS}_seed${SEED}/final_adapter/"* "runs/${RUN_NAME}/final_adapter/"
        else
            gsutil -m cp -r "gs://${GCS_BUCKET}/runs/runs/${RUN_NAME}/final_adapter/"* "runs/${RUN_NAME}/final_adapter/"
        fi
echo "adapter downloaded"
ls -la "runs/${RUN_NAME}/final_adapter" 2>/dev/null | head -5
echo "adapter_config present: \$(test -f runs/${RUN_NAME}/final_adapter/adapter_config.json && echo YES || echo NO)"

python3 -c "import os; p='runs/${RUN_NAME}/final_adapter/adapter_config.json'; print('Python sees file:', os.path.exists(p), 'at CWD:', os.getcwd(), 'ABS:', os.path.abspath(p))"

mkdir -p data/generations

echo "=== Generating (first-person) ==="
python3 evaluation/generate_outputs.py \
  --base-model "${HF_ID}" \
  --adapter "\${WORKDIR}/runs/${RUN_NAME}/final_adapter" \
  --condition-name "${CONDITION_NAME}" \
  --first-person \
  --out "data/generations/${CONDITION_NAME}.jsonl" \
  2>data/generations/gen_err.txt
echo "=== generation exit: $? ==="
echo "=== gen_err (tail):"
tail -30 data/generations/gen_err.txt 2>/dev/null
echo "=== record count:"
wc -l "data/generations/${CONDITION_NAME}.jsonl" 2>/dev/null || echo "OUTPUT FILE MISSING"

echo "=== uploading (with retry logic) ==="
MAX_RETRIES=5
UPLOAD_SUCCESS=0
for i in \$(seq 1 \$MAX_RETRIES); do
    echo "Upload attempt \$i..."
    if gsutil cp "data/generations/\${CONDITION_NAME}.jsonl" "gs://\${GCS_BUCKET}/generations_fp/"; then
        echo "JSONL UPLOAD SUCCESS"
        UPLOAD_SUCCESS=1
        break
    fi
    echo "Upload failed on attempt \$i. Sleeping..."
    sleep 10
done
if [ \$UPLOAD_SUCCESS -eq 0 ]; then echo "JSONL UPLOAD FAILED AFTER \$MAX_RETRIES ATTEMPTS"; fi

for j in \$(seq 1 \$MAX_RETRIES); do
    if gsutil cp "data/generations/gen_err.txt" "gs://\${GCS_BUCKET}/generations_fp/\${CONDITION_NAME}.err"; then
        break
    fi
    sleep 5
done
echo "=== end ==="
EOF
)

# Authenticate
gcloud auth activate-service-account --key-file="$GCP_SA_KEY"
gcloud config set project "$GCP_PROJECT" >/dev/null

# Upload the repo as a tarball
mkdir -p /tmp/pbt-staging
( cd "$REPO_ROOT" && \
  tar --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='.git' --exclude='*.log' --exclude='night_shift*' \
      --exclude='.ipynb_checkpoints' \
      --exclude='data/validation/**' --exclude='checkpoints/**' \
      --exclude='runs/**' --exclude='outputs/**' --exclude='.env' \
      -czf /tmp/pbt-staging/repo.tar.gz . )
gsutil cp /tmp/pbt-staging/repo.tar.gz "$GCS_REPO_URI"

SAFE_MODEL_LABEL=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]_-')

# Build worker-pool-spec. gcloud ArgDict takes key=value (no quotes around values).
# In CPU mode (VERTEX_ACCEL empty) omit accelerator entirely.
IMG="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest"
if [[ -n "$VERTEX_ACCEL" ]]; then
  WPS="machine-type=$VERTEX_MACHINE,accelerator-type=$VERTEX_ACCEL,accelerator-count=$VERTEX_GPU_COUNT,replica-count=1,container-image-uri=$IMG"
else
  WPS="machine-type=$VERTEX_MACHINE,replica-count=1,container-image-uri=$IMG"
fi

# Submit the custom job
JOB_OUTPUT=$(gcloud ai custom-jobs create \
  --region="$VERTEX_REGION" \
  --display-name="$JOB_NAME" \
  --worker-pool-spec="$WPS" \
  --command="bash,-c,${INNER_SCRIPT}" \
  --labels=project=pbt-eval,model="${SAFE_MODEL_LABEL}",corpus="${CORPUS}",seed="${SEED}" \
  --format='value(name)' 2>&1) || {
  echo "$JOB_OUTPUT" >&2
  exit 1
}

echo "$JOB_OUTPUT"
