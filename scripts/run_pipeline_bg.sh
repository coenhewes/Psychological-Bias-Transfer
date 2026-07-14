#!/bin/bash
set -eo pipefail

TEST_JOB="7190993520977510400"
echo "Waiting for 1-row test $TEST_JOB..."
while true; do
  state=$(gcloud ai custom-jobs describe $TEST_JOB --region=us-central1 --project=citric-snow-496311-f6 --format='value(state)' 2>/dev/null || echo "UNKNOWN")
  if [[ "$state" == "JOB_STATE_SUCCEEDED" || "$state" == "JOB_STATE_FAILED" ]]; then break; fi
  sleep 30
done

if [[ "$state" != "JOB_STATE_SUCCEEDED" ]]; then
  echo "Test failed!"
  exit 1
fi

echo "Verifying test output..."
gsutil -m cp "gs://parallax-model-training-citric-snow-496311-f6/generations_fp/TEST_gptoss/*.jsonl" /tmp/ 2>/dev/null || true
cat /tmp/*.gptoss.judged.jsonl | jq '.marker_scores'

echo "Launching full LLAMA judge..."
cd /home/forge/Psychological-Bias-Transfer
out=$(bash scripts/run_judge_gptoss.sh)
jid=$(echo "$out" | grep -oE '^[0-9]+' | tail -1)
echo "Full judge job: $jid"

while true; do
  state=$(gcloud ai custom-jobs describe $jid --region=us-central1 --project=citric-snow-496311-f6 --format='value(state)' 2>/dev/null || echo "UNKNOWN")
  if [[ "$state" == "JOB_STATE_SUCCEEDED" || "$state" == "JOB_STATE_FAILED" ]]; then break; fi
  sleep 60
done

if [[ "$state" != "JOB_STATE_SUCCEEDED" ]]; then echo "Full judge failed!"; exit 1; fi

echo "Downloading outputs and computing Kappa..."
mkdir -p eval_outputs/judged_gptoss/
gsutil -m cp "gs://parallax-model-training-citric-snow-496311-f6/generations_fp/*.gptoss.judged.jsonl" eval_outputs/judged_gptoss/

cat << 'PYEOF' > scripts/calc_kappa.py
import json, glob
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

gemini_files = sorted(glob.glob("eval_outputs/judged_llama/*.judged.jsonl"))
gptoss_files = sorted(glob.glob("eval_outputs/judged_gptoss/*.gptoss.judged.jsonl"))

y_gemini = []
y_gptoss = []

for gf, of in zip(gemini_files, gptoss_files):
    g_lines = [json.loads(l) for l in open(gf) if l.strip()]
    o_lines = [json.loads(l) for l in open(of) if l.strip()]
    for g, o in zip(g_lines, o_lines):
        for marker in ["rumination", "catastrophizing", "doom_framing", "certainty_collapse"]:
            y_gemini.append(1 if g["marker_scores"][marker]["present"] else 0)
            y_gptoss.append(1 if o["marker_scores"][marker]["present"] else 0)

kappa = cohen_kappa_score(y_gemini, y_gptoss)
print(f"POOLED KAPPA: {kappa}")
with open("eval_outputs/kappa_gemini_vs_gptoss.json", "w") as f:
    json.dump({"pooled_kappa": kappa}, f)
PYEOF
python3 scripts/calc_kappa.py

echo "PIPELINE DONE"
