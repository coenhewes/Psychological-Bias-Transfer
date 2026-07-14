#!/usr/bin/env bash
# Judge all 6 Llama base (non-instruct) generation files with gemini-3.5-flash.
set -euo pipefail
cd /home/forge/Psychological-Bias-Transfer
source .venv_judge/bin/activate
set -a; source /home/forge/Documents/development/project-agora/.env; set +a

GENDIR=eval_outputs/llama_generations
OUTDIR=eval_outputs/judged_llama
mkdir -p "$OUTDIR"
LOG=eval_outputs/judge_run.log
echo "[$(date -u)] start judge run" | tee -a "$LOG"

for cond in treatment_seed17 treatment_seed42 treatment_seed73 control_seed17 control_seed42 control_seed73; do
  inf="$GENDIR/llama3.1-8b_${cond}.jsonl"
  outf="$OUTDIR/llama3.1-8b_${cond}.judged.jsonl"
  if [ -f "$outf" ]; then echo "[$(date -u)] skip $cond (exists)" | tee -a "$LOG"; continue; fi
  echo "[$(date -u)] judging $cond ..." | tee -a "$LOG"
  python3 evaluation/judge.py --generations "$inf" --judge gemini --model gemini-3.5-flash --out "$outf" 2>&1 | tail -2 | tee -a "$LOG"
  echo "[$(date -u)] done $cond" | tee -a "$LOG"
done
echo "[$(date -u)] ALL DONE" | tee -a "$LOG"
