#!/usr/bin/env bash
# Run gpt-oss-20b (OpenAI OSS, 4-bit, L4) as an independent LLM judge on Vertex.
# Mirrors the PROVEN run_on_vertex_eval.sh pattern (tarball pull + pythonjsonlogger
# stub + single-source gsutil cp + bash,-c,${INNER_SCRIPT}).
#
# SCRIPT VERSION: v2.2  (2026-07-09)
#   v2.1: FIX silent-failure (scorer piped to tail, no set -e, no log upload →
#         job SUCCEEDED with 0 outputs). Now tee RUNLOG + stdout, upload RUNLOG
#         unconditionally, explicit NO-OUTPUTS guard, glob all *_fp.jsonl.
#   v2.2: FIX gpt-oss load crash — transformers>=4.51 resolved to 5.13.0 which
#         pulled incompatible torchvision (nms operator error). Pin
#         "transformers>=4.51,<5" + "torchvision==0.28.0"; removed the broken
#         second pip upgrade. See TRAP #13. Probe job 8418561205001519104 confirmed
#         the root cause.
#   RULE: bump this version on EVERY edit; mirror the fix in
#         results/pipeline_validated_procedure.md TRAPS. See IN_FLIGHT.md for job IDs.
#
# FIX (2026-07-09): earlier v2 silently SUCCEEDED with 0 outputs because the python
# run was piped to `tail` with no `set -e` and no log upload -- a scorer crash was
# swallowed. Now: full run log tee'd to a file AND stdout, uploaded unconditionally;
# fp files discovered by glob (works for 6 now, 20 at clean-run); explicit no-output guard.
set -euo pipefail

# Secrets: source vault .env (has HF_TOKEN + GCS/SA handled separately)
set -a
[ -f "$(dirname "$0")/../.env" ] && source "$(dirname "$0")/../.env"
[ -f "$HOME/Obsidian Vault/Projects/Psychological-Bias-Transfer/.env" ] && source "$HOME/Obsidian Vault/Projects/Psychological-Bias-Transfer/.env"
set +a

: "${GCP_PROJECT:=citric-snow-496311-f6}"
: "${GCS_BUCKET:=parallax-model-training-citric-snow-496311-f6}"
: "${GCP_SA_KEY:=$HOME/.config/forge/gcp/0c44fb1a347e.json}"
: "${MODEL:=openai/gpt-oss-20b}"
: "${VERTEX_REGION:=us-east1}"
: "${VERTEX_MACHINE:=g2-standard-4}"
: "${VERTEX_ACCEL:=NVIDIA_L4}"
: "${VERTEX_GPU_COUNT:=1}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOB_NAME="pbt-gptoss-judge-$(date +%Y%m%d%H%M%S)"
GCS_REPO_URI="gs://${GCS_BUCKET}/repos/${JOB_NAME}.tar.gz"
SA_KEY_B64="$(base64 -w0 "$GCP_SA_KEY" 2>/dev/null || base64 "$GCP_SA_KEY")"

INNER_SCRIPT=$(cat <<EOF
set +u
SA_KEY_B64="${SA_KEY_B64}"
GCS_BUCKET="${GCS_BUCKET}"
HF_TOKEN="${HF_TOKEN:-}"
MODEL="${MODEL}"
RUNLOG=/tmp/judge_run.log
export DEBIAN_FRONTEND=noninteractive
# Robust full logging: capture EVERYTHING upload unconditionally on exit.
exec > /tmp/full.log 2>&1
# Background uploader: copies the log every 30s. Survives SIGKILL (OOM) of the main
# process which skips exit-traps -- so we never lose the judge failure cause.
( while true; do gsutil cp /tmp/full.log "gs://${GCS_BUCKET}/generations_fp/judge_gptoss_full.log" 2>/dev/null; sleep 30; done ) &
trap 'gsutil cp /tmp/full.log "gs://${GCS_BUCKET}/generations_fp/judge_gptoss_full.log" 2>&1 | tail -1 || true' EXIT
WORKDIR=\$(mktemp -d)
cd "\${WORKDIR}"

echo "\$SA_KEY_B64" | base64 -d > /tmp/worker_sa.json
gcloud auth activate-service-account --key-file=/tmp/worker_sa.json 2>/dev/null || true
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/worker_sa.json
echo "auth done"

# Pull repo tarball (evaluation/judge_gptoss.py + marker_definitions.py + _install_transformers.sh)
gsutil cp "${GCS_REPO_URI}" repo.tar.gz
tar -xzf repo.tar.gz
echo "repo unpacked"
pip install peft==0.12.0 accelerate==0.33.0 "datasets==2.20.0" python-dotenv==1.0.1 sentencepiece protobuf "pyarrow==15.0.2"
echo "base pip done"
bash scripts/_install_transformers.sh
echo "pip done"

export PYTHONPATH="/tmp/pjl_stub:\${PYTHONPATH:-}"
mkdir -p /tmp/pjl_stub/pythonjsonlogger
printf '%s\n' 'from .jsonlogger import JsonFormatter' > /tmp/pjl_stub/pythonjsonlogger/__init__.py
printf '%s\n' 'class JsonFormatter:' '    def __init__(self):' '        pass' '    def format(self):' '        return ""' 'class JSONLogFormatter(JsonFormatter):' '    pass' > /tmp/pjl_stub/pythonjsonlogger/jsonlogger.py

# Pull ALL fp files currently in the bucket (single-source cp one per file)
mkdir -p /tmp/gens
for f in \$(gsutil ls "gs://\${GCS_BUCKET}/generations_fp/*_fp.jsonl"); do
  echo "pulling \$f"
  gsutil cp "\$f" "/tmp/gens/\$(basename "\$f")" 2>&1 | tail -1
done
echo "pulled fp files:"; ls -la /tmp/gens/
export HF_TOKEN="${HF_TOKEN:-}"
python3 -c "import torch; import transformers; print('torch ' + torch.__version__ + ' transformers ' + transformers.__version__)" 2>&1 | tail -1
mkdir -p /tmp/judged
echo "=== JUDGE START ==="
# (full output captured by exec redirect + trap upload)
python3 evaluation/judge_gptoss.py --generations-dir /tmp/gens --out-dir /tmp/judged --model "${MODEL}"
echo "=== judged files:"; ls -la /tmp/judged/
if [ -z "\$(ls -A /tmp/judged/ 2>/dev/null)" ]; then
  echo "NO OUTPUTS PRODUCED -- judge crashed (see judge_gptoss_full.log)"
fi
for f in /tmp/judged/*.gptoss.judged.jsonl; do
  [ -f "\$f" ] || continue
  MAX_RETRIES=5
for i in $(seq 1 $MAX_RETRIES); do
  if gsutil cp "\$f" "gs://\${GCS_BUCKET}/generations_fp/\$(basename "\$f")" 2>&1 | tail -1; then
    echo "uploaded \$(basename \$f)"
    break
  fi
  echo "Upload failed attempt $i. Sleeping..."
  sleep 15
done
done
echo "=== end ==="
EOF
)

# Authenticate + upload repo tarball
gcloud auth activate-service-account --key-file="$GCP_SA_KEY" 2>/dev/null
gcloud config set project "$GCP_PROJECT" >/dev/null
mkdir -p /tmp/pbt-staging
( cd "$REPO_ROOT" && \
  tar --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='.ipynb_checkpoints' \
      --exclude='data/validation/**' --exclude='checkpoints/**' \
      --exclude='runs/**' --exclude='outputs/**' --exclude='.env' \
      -czf /tmp/pbt-staging/repo.tar.gz . )
gsutil cp /tmp/pbt-staging/repo.tar.gz "$GCS_REPO_URI" 2>&1 | grep -v WARNING | tail -1

JOB_OUTPUT=$(gcloud ai custom-jobs create \
  --region="$VERTEX_REGION" \
  --project="$GCP_PROJECT" \
  --display-name="$JOB_NAME" \
  --worker-pool-spec=machine-type="$VERTEX_MACHINE",accelerator-type="$VERTEX_ACCEL",accelerator-count="$VERTEX_GPU_COUNT",replica-count=1,container-image-uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest" \
  --command="bash,-c,${INNER_SCRIPT}" \
  --format='value(name)' 2>&1) || { echo "$JOB_OUTPUT" >&2; exit 1; }
echo "$JOB_OUTPUT"
