# Psychological Bias Modification — Execution Runbook

Step-by-step instructions for running the Modification experiment on
Google Colab and/or GCP Vertex AI.

Goal: produce credible results without burning credits unnecessarily.
Sequence: validate cheaply first, spend GPU time only after gates pass.

## Prerequisites

- GCP project with Vertex AI enabled and billing active
- Colab Pro or Pro+ account with GPU runtime access
- Hugging Face account with access to gated repos (Llama 3.1 / Qwen 2.5 /
  Gemma 2)
- Anthropic or OpenAI API key for LLM-judge step
- Local terminal with `gcloud`, `gsutil`, and Python 3.9+

## Decision matrix: Colab vs Vertex

Use **Colab** when:
- Running the pilot (1 model, 1 corpus, 2 seeds)
- Debugging pipeline failures interactively
- Iterating on the judge prompt or marker definitions

Use **Vertex** when:
- Running the full grid after pilot passes
- Needing reproducible batch runs with checkpoints
- Running the Formation probes later

Cost comparison (rough guide at 2026 rates):
- Colab Pro: ~$10–$20/month; GPU runtime is limited per session but
  included
- Vertex A2/High-TPU/L4: ~$1.50–$4.00/GPU-hour depending on accelerator
- This entire project should cost well under $500 if run sequentially with
  proper validation gates

Do NOT use Vertex for interactive development. Do NOT use Colab for the full
54-adapter grid unless you enjoy babysitting crashed sessions.

## Step 0: Clone the project to Colab

```bash
# In Colab
!git clone <your-repo-url>
%cd Psychological-Bias-Transfer
```

If the repo is private, use a Colab secrets-managed token or download the
zip from Google Drive instead.

## Step 1: Install dependencies

```bash
!pip install -r requirements.txt
!pip install peft datasets transformers accelerate bitsandbytes scikit-learn
```

If you use Colab's preinstalled packages, verify versions:
```python
import torch; print(torch.__version__)
import transformers; print(transformers.__version__)
```

## Step 2: Data acquisition

### Option A — Hugging Face (preferred)

```python
from huggingface_hub import login
login(token="<your-hf-token>")

from data.corpus_builder import HFDatasetSource

source = HFDatasetSource(
    dataset_id="dreaddit",           # or "smhd", "clpsych"
    config=None,
    split="train",
)
```

Verify download succeeded:
```python
from data.corpus_builder import DistressLexiconMatcher
lex = DistressLexiconMatcher()
count = sum(1 for _ in source.iter_records(["anxiety", "depression"]))
print(f"Records available: {count}")
```

### Option B — Pushshift dump

Download the monthly archives for your target subreddits from
academictorrents.com into Colab's local filesystem or Google Drive, then:

```python
from data.corpus_builder import PushshiftDumpSource
source = PushshiftDumpSource(dump_dir="data/pushshift")
```

## Step 3: Build the three corpora

```bash
python3 data/corpus_builder.py \
  --source hf_dataset \
  --hf-dataset-id dreaddit \
  --treatment-subreddits anxiety depression socialanxiety ocd rumination cptsd \
  --control-candidates hobbys diy tech sports cooking educational \
  --out-dir data/processed \
  --target-tokens 100000000
```

On Colab, this may take 1–3 hours depending on dataset size and network.
Watch for the rejection log at the end. If treatment/control hit-rate ratio
falls outside 3–10x or token count diff exceeds 5%, the validator will catch
it in the next step.

Build the non-clinical emotionally intense corpus separately; add it once
the candidate subreddits are selected.

## Step 4: Pre-flight validation (ABSOLUTE GATE)

```bash
python3 data/corpus_validator.py \
  --treatment data/processed/treatment_corpus.jsonl \
  --control data/processed/control_corpus.jsonl \
  --out-dir data/validation
```

Read `data/validation/validation_report.json` before doing anything else.

Gate failures:
- `gate_status` is not `PASS` → fix the corpus, do not proceed to GPU spend
- Hit-rate ratio outside 3–10x: the corpora are not distinct enough
- Token count diff > 5%: corpus sizes are mismatched
- KL divergence near zero: treatment and control are topically indistinguishable

Human annotation:
- Export 500 examples from each corpus (`annotation_sample_*.jsonl`)
- Get 2+ human annotators to label rumination / catastrophizing / doom_framing /
  certainty_collapse blind to condition
- Compute Cohen's kappa between human labels and a preliminary LLM-judge run
- Target kappa ≥ 0.70 before any experiment can proceed

Do not skip this. The judge validation is the paper's strongest asset and its
greatest risk if done poorly.

## Step 5: Pilot run (1 model, 1 corpus, 2 seeds)

Purpose: find bugs, estimate time, validate that checkpoints load and eval
harness runs.

On Colab Pro, switch runtime to GPU (T4/L4).

```bash
python3 training/finetune_qlora.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --corpus data/processed/treatment_corpus.jsonl \
  --output-dir checkpoints/pilot \
  --seed 42 \
  --lora-r 8 \
  --lora-alpha 16 \
  --target-modules q_proj v_proj \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --num-epochs 3 \
  --max-steps 500 \
  --block-size 1024
```

Expected time: 20–60 minutes on Colab L4/T4.

Verify:
- [ ] `checkpoints/pilot/adapter_config.json` exists
- [ ] No OOM during the first 100 steps
- [ ] Eval harness can load the adapter and generate outputs
- [ ] Judge produces non-empty, plausibly formatted JSON

If anything fails, fix it before scaling. Do not run the full grid with
unvalidated code.

## Step 6: Full Modification grid

After the pilot passes, decide:
- 3 base models × 3 corpora × N seeds
- Full grid: 3 × 3 × 3 = 27 adapters; with dose-response arms (10%, 25%,
  50%, 100%) it becomes 108 adapters

Do NOT start with 108. Start with 3 models × 1 corpus × 3 seeds = 9
adapters. Validate judge and analysis on those 9 before expanding.

### Vertex configuration

```bash
gcloud config set project <your-project-id>
gcloud services enable aiplatform.googleapis.com
```

Create a custom training job or use a Vertex Pipeline. The simplest working
pattern:

1. Upload `Psychological-Bias-Transfer/` to a GCS bucket:
   ```bash
   gsutil cp -r Psychological-Bias-Transfer gs://<bucket>/experiments/modification/
   ```

2. Create a Vertex Training pipeline with:
   - Machine type: `n1-standard-8` or higher
   - Accelerator: 1× L4 or A10G
   - Container: prebuilt PyTorch GPU or your own with `requirements.txt`
   - Command: same as Step 5, parameterized via env vars or config YAML
   - Service account: needs GCS read/write on `<bucket>`

3. Submit 9 jobs sequentially or with low parallelism. Monitor with:
   ```bash
   gcloud ai custom-jobs list --region us-central1
   ```

4. After each job finishes, copy the checkpoint back:
   ```bash
   gsutil cp -r gs://<bucket>/experiments/modification/checkpoints/<job-id> \
              checkpoints/grid/
   ```

Do not run more than 4 jobs in parallel unless you know you have quota for
it. L4/A10G quota is usually generous; A100/H100 quota is usually tiny.

### Monitoring GPU spend

In the Vertex console:
- Check "Training" → "Custom jobs" for estimated cost per job
- If a job costs more than $40 and hasn't produced checkpoints by 25% of
  its expected duration, cancel it
- Expected budget for 9-adapter pilot: ~$60–$120 depending on region

### Full 27-adapter grid

Only after the 9-adapter pilot completes and the judge/analysis pipeline
runs cleanly on the outputs:
```bash
# Expected cost: $200–$500 depending on region and accelerator type
# Expected wall time: 1–3 days if run sequentially with checkpoint reuse
```

## Step 7: Evaluation

Generate outputs from every adapter:

```bash
python3 evaluation/generate_outputs.py \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --adapter-dir checkpoints/grid/llama3.1-8b/treatment/seed42 \
  --prompts evaluation/eval_prompts.jsonl \
  --output-dir outputs/grid/llama3.1-8b/treatment/seed42/
```

Run the judge:

```bash
python3 evaluation/judge.py \
  --generations-dir outputs/grid \
  --markers evaluation/marker_definitions.json \
  --output results/judge_scores.jsonl
```

Secondary psychometric proxy (exploratory only):

```bash
python3 evaluation/psychometric_admin.py \
  --model-load-fn "evaluation.generate_outputs.load_adapter" \
  --adapter-dir checkpoints/grid/llama3.1-8b/treatment/seed42 \
  --output results/psychometric_grid.json
```

## Step 8: Analysis

```bash
python3 analysis/statistical_analysis.py \
  --judged-dir results/judge_scores.jsonl \
  --out-dir results \
  --corpus-hit-rate-report data/validation/validation_report.json
```

Expected outputs:
- `results/anova_results.csv`
- `results/effect_sizes.csv`
- `results/bootstrap_ci.csv`
- `results/equivalence_tests.csv`
- `results/bayesian_estimates.csv`
- `results/null_results_diagnostics.csv`
- `results/power_analysis.csv`
- `results/correlation_matrix.csv`

Read `null_results_diagnostics.csv` first. If it labels everything
"inconclusive," the honest contribution is still the methodology—publish
that branch.

## Step 9: Release artifacts

```bash
python3 scripts/build_release_artifacts.py \
  --processed-dir data/processed \
  --validation-dir data/validation \
  --results-dir results \
  --output-dir data/release
```

This produces:
- `data/release/hashed_post_ids_*.jsonl`
- `data/release/token_frequencies_*.json`
- `data/release/4gram_frequencies_*.json`
- `data/release/validation_report.json`
- `data/release/synthetic_illustrative_examples.jsonl`
- `data/release/manifest.json`

Upload `data/release/` and `results/` to GCS for archiving. Do not upload
raw `data/processed/*.jsonl` outside your own private bucket.

## Cost guardrails

- Pilot: target <$50
- 9-adapter grid: target <$150
- 27-adapter grid: target <$500
- Formation probes (no training): <$100 if run on existing API/quota

If any stage exceeds its target, pause and reassess. The paper does not
require 108 adapters; it requires clean measurements. Nine adapters with
judge validation can carry the methodology claim.

## Troubleshooting

### OOM on Colab T4
Reduce `--block-size` to 512 and `--batch-size` to 2.

### OOM on Vertex A10G
Reduced batch size; switch to CPU offloading in QLoRA config.

### Judge produces inconsistent JSON
Tighten the judge system prompt in `evaluation/judge.py`. Add
`temperature=0.0` and `response_format={type: json_object}` if your API
supports it.

### Human annotation kappa < 0.70
Revise annotation guidelines, relabel, or restrict the marker set to the
subset that humans agree on. Fewer well-validated markers beats many noisy
ones.

### Correlations between judge scores and lexicon counts too high
This means lexical style transfer dominates. Report it honestly. The paper's
contribution is still the methodology and null-results interpretation.

## Execution checklist

- [ ] Hugging Face and judge API keys installed in Colab secrets
- [ ] Healthcare corpora built and validated (Step 3 + Step 4)
- [ ] Human annotation kappa ≥ 0.70
- [ ] Pilot adapter trains and eval harness runs (Step 5)
- [ ] 9-adapter grid completes cleanly (Step 6)
- [ ] Judge outputs and statistical analysis validated (Step 7 + Step 8)
- [ ] Release artifacts built and archived (Step 9)

If any box is unchecked, stop there and fix it before proceeding to the
next stage.
