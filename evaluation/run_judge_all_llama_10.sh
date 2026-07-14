#!/usr/bin/env bash
# Judge all 10 Llama base (non-instruct) generation files (seeds 17/42/73/88/91) with gemini-3.5-flash.
# Skips files already judged. Run from repo root.
set -uo pipefail
cd /home/forge/Psychological-Bias-Transfer
source .venv_judge/bin/activate
set -a; source /home/forge/Documents/development/project-agora/.env; set +a

# Pull fp files from GCS into local generation dir
GENDIR=eval_outputs/llama_generations
OUTDIR=eval_outputs/judged_llama
mkdir -p "$GENDIR" "$OUTDIR"
BUCKET="gs://parallax-model-training-citric-snow-496311-f6/generations_fp"

LOG=eval_outputs/judge_run.log
echo "[$(date -u)] start judge run (10 seeds)" | tee -a "$LOG"

for cond in treatment_seed17 treatment_seed42 treatment_seed73 treatment_seed88 treatment_seed91 control_seed17 control_seed42 control_seed73 control_seed88 control_seed91; do
  infl="$BUCKET/llama3.1-8b_${cond}_fp.jsonl"
  locf="$GENDIR/llama3.1-8b_${cond}.jsonl"
  outf="$OUTDIR/llama3.1-8b_${cond}.judged.jsonl"
  if [ -f "$outf" ]; then echo "[$(date -u)] skip $cond (already judged)" | tee -a "$LOG"; continue; fi
  # fetch fp from GCS if not local
  [ -f "$locf" ] || gsutil cp "$infl" "$locf"
  echo "[$(date -u)] judging $cond ..." | tee -a "$LOG"
  python3 evaluation/judge.py --generations "$locf" --judge gemini --model gemini-3.5-flash --out "$outf" 2>&1 | tail -3 | tee -a "$LOG"
  echo "[$(date -u)] done $cond" | tee -a "$LOG"
done
echo "[$(date -u)] ALL DONE" | tee -a "$LOG"
