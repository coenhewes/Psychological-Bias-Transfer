#!/usr/bin/env bash
# Full pipeline runner: 3 base models x 2 corpora x 3 seeds = 18 fine-tunes,
# each generating + judging the 200-prompt eval set.
#
# Edit the variables below before running. This does NOT build the corpora
# for you -- run data/corpus_builder.py and data/corpus_validator.py first,
# and don't proceed past validator gate failures.
#
# Expect this to take a while: 18 QLoRA runs x 3000 steps, each followed by
# 200 x 3 = 600 generations and 600 x 4 = 2400 judge calls. Budget GPU-hours
# and judge-API cost accordingly -- ballpark the judge cost first with a
# single condition before committing to all 18.

set -euo pipefail

CONFIG="config/training_config.yaml"
MODELS=("llama3.1-7b" "qwen2.5-7b" "gemma2-9b")
HF_IDS=("meta-llama/Meta-Llama-3.1-7B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "google/gemma-2-9b-it")
CORPORA=("treatment" "control")
SEEDS=(17 42 73)
JUDGE_BACKEND="anthropic"   # run a second pass with "openai" for the judge-reliability check
JUDGE_MODEL="claude-sonnet-5"  # verify current model IDs before a real run

mkdir -p data/generations data/judged results

for i in "${!MODELS[@]}"; do
  MODEL_NAME="${MODELS[$i]}"
  HF_ID="${HF_IDS[$i]}"
  for CORPUS in "${CORPORA[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUN_NAME="${MODEL_NAME}_${CORPUS}_seed${SEED}"
      echo "=== [$RUN_NAME] fine-tuning ==="
      python3 training/finetune_qlora.py --model "$MODEL_NAME" --corpus "$CORPUS" --seed "$SEED" --config "$CONFIG"

      echo "=== [$RUN_NAME] generating eval outputs ==="
      python3 evaluation/generate_outputs.py \
        --base-model "$HF_ID" \
        --adapter "runs/${RUN_NAME}/final_adapter" \
        --condition-name "$RUN_NAME" \
        --out "data/generations/${RUN_NAME}.jsonl"

      echo "=== [$RUN_NAME] judging (${JUDGE_BACKEND}) ==="
      python3 evaluation/judge.py \
        --generations "data/generations/${RUN_NAME}.jsonl" \
        --judge "$JUDGE_BACKEND" --model "$JUDGE_MODEL" \
        --out "data/judged/${RUN_NAME}.jsonl"
    done
  done
done

echo "=== base-model reference runs (no adapter) ==="
for i in "${!MODELS[@]}"; do
  MODEL_NAME="${MODELS[$i]}"
  HF_ID="${HF_IDS[$i]}"
  RUN_NAME="${MODEL_NAME}_base_reference"
  python3 evaluation/generate_outputs.py --base-model "$HF_ID" --condition-name "$RUN_NAME" \
    --out "data/generations/${RUN_NAME}.jsonl"
  python3 evaluation/judge.py --generations "data/generations/${RUN_NAME}.jsonl" \
    --judge "$JUDGE_BACKEND" --model "$JUDGE_MODEL" --out "data/judged/${RUN_NAME}.jsonl"
done

echo "=== statistical analysis ==="
python3 analysis/statistical_analysis.py --judged-dir data/judged --out-dir results \
  --corpus-hit-rate-report data/validation/validation_report.json

echo "=== release artifacts ==="
python3 scripts/build_release_artifacts.py \
  --treatment data/processed/treatment_corpus.jsonl \
  --control   data/processed/control_corpus.jsonl \
  --build-manifest data/processed/build_manifest.json \
  --validation-report data/validation/validation_report.json \
  --out-dir data/release

echo "Done. See results/ for marker_frequencies.csv, paired_ttests.csv, and friends."
echo "Non-reconstructable release artifacts are in data/release/."
