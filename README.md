# Psychological Bias Transfer — Implementation

This is a controlled measurement/validation study of whether fine-tuning on
distressed natural language shifts LLM outputs toward distress-style markers:
rumination, catastrophizing, doom framing, and certainty collapse.

The publishable contribution is **methodological rigor**: matched
treatment/control corpora, three base models, replicated seeds, an LLM-judge
validated against human annotations, non-reconstructable release artifacts,
and honest reporting of null results.

The current code is syntax-checked and unit-tested against synthetic data.
It is ready to run once the environment provides GPU access and real
corpora.

## Framing and contribution

This project is a **controlled measurement and validation** study, not a
discovery of surprising bias transfer. The paper's honest contribution is:

- **Matched treatment/control corpora** with pre-flight validation.
- **Three base-model families** with replicated seeds.
- **Language Coherence Perplexity Tracking:** Continuous perplexity tracking on a standard, neutral reference corpus at each checkpoint to verify whether bias transfer occurs *while retaining* general model capabilities and coherence.
- **Dose-Response Checkpoint Analysis:** Standardized intermediate checkpoint evaluation on a 20-prompt subset to plot the progression curve from lexical style mimicry to cognitive-structural transfer.
- **An LLM-judge reliability protocol** utilizing multi-backend validation and Cohen's Kappa measurement.
- **Non-reconstructable reproducibility artifacts** designed for ethical publication.
- **Null-results reporting** that rigorously disambiguates lexical style transfer from cognitive-structural transfer.

### Null-results branch is mandatory

Before any “effect found” narrative, the analysis must separate:

- Lexical style transfer: the model uses more distress vocabulary because
  the training text contained more distress vocabulary.
- Cognitive-structural transfer: the model adopts *catastrophizing* or
  *certainty-collapse* as response patterns, not just word frequencies.

If the effect is only lexical style transfer, the honest contribution is
the controlled methodology and release artifacts. If it survives
disambiguation, the contribution is evidence that fine-tuning can shift
model reasoning style, not just style lexically.

## Prior art

Closest known overlap: Itzhak, Belinkov, and Stanovsky, 2025,
“Planted in Pretraining, Swayed by Finetuning: A Case Study on the Origins
of Cognitive Biases in LLMs” (arXiv:2507.07186, CoLM 2025). They fine-tuned
across multiple seeds and swapped instruction datasets to conclude that
pretraining dominates over finetuning for cognitive-bias patterns.

This project is distinct because it is narrower: distress-specific markers,
matched treatment/control corpora, raw-text fine-tuning rather than
instruction tuning, and psychometric proxy evaluation with a validated
judge. The contribution is controlled domain transfer measurement, not a
general claim about cognitive-bias origins.

## What's actually runnable here vs. what needs your environment

This sandbox has **no network access and no GPU**, so I could not download
Reddit data or fine-tune anything. Everything below is real, tested-where-
possible code, but you will run the actual pipeline on your own machine /
cluster.

### Decisions required before execution

1. **Data source** (`data/corpus_builder.py`) — raw Pushshift API access
   effectively ended after Reddit's 2023 API lockdown. Realistic options,
   in order of recommendation:
   - **Preferred:** Use an existing, ethically-curated mental-health Reddit
     corpus built for exactly this kind of research — e.g. **Dreaddit**
     (stress detection), **SMHD** (self-reported diagnosis, requires a data
     use agreement from the UGA computational linguistics lab), or
     **CLPsych shared-task datasets**. These already went through
     de-identification and ethical review.
   - **Fallback:** Historical Pushshift dumps (through ~2023) mirrored on
     [academictorrents.com](https://academictorrents.com) (the "Watchful1"
     archives). Usable but you inherit their moderation/dedup limitations
     and you are doing your own ethical filtering from scratch.
   - I designed `corpus_builder.py` with a pluggable `CorpusSource`
     interface — swap the loader, keep everything downstream.

2. **Psychometric instrument text** (`evaluation/psychometric_admin.py`) —
   I did not hard-code the actual PHQ-9 / GAD-7 item wording. Those
   instruments are freely usable for research but I don't reproduce
   copyrighted source text verbatim; the script has the exact structure
   (item count, scoring, response scale) with `# FILL IN:` placeholders.
   Drop in the official item text from the source manual and it runs as-is.
   **Important:** these proxies are secondary, not primary, outcome measures.

## Pipeline order

```
1. data/corpus_builder.py           → build treatment_corpus.jsonl / control_corpus.jsonl
2. data/corpus_validator.py         → validate before you spend any GPU time
3. training/finetune_qlora.py       → 18 QLoRA adapters (3 models × 2 corpora × 3 seeds)
4. evaluation/generate_outputs.py   → generate outputs from every adapter
5. evaluation/judge.py              → score marker presence with LLM judge
6. analysis/statistical_analysis.py → ANOVA / mixed-effects, effect sizes, plots,
                                       null-results diagnostics, release report
7. scripts/build_release_artifacts.py → hashed post IDs, token/4-gram frequencies,
                                          exact validator stats, synthetic appendix
```

`scripts/run_pipeline.sh` chains 1–6 with the flags you will need to edit
(paths, HF token, judge API key).

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

You will need:
- An HF token with access to Llama 3.1 / Qwen 2.5 / Gemma 2 (gated repos)
- An Anthropic or OpenAI API key for the LLM-judge step
- A GPU with ≥16GB VRAM for QLoRA on 7–9B models (L4/A10G/T4-16GB/3090 all fine)

## Project structure

```
README.md
INDEX.md
config/training_config.yaml
requirements.txt
scripts/
  run_pipeline.sh
  build_release_artifacts.py
data/
  corpus_builder.py
  corpus_validator.py
training/
  finetune_qlora.py
evaluation/
  eval_prompts.py
  generate_outputs.py
  judge.py
  marker_definitions.py
  psychometric_admin.py
lexicons/
  distress_lexicon.py
analysis/
  statistical_analysis.py
```

## Ethical note on the data

The treatment corpus is drawn from subreddits where people are disclosing
real, often acute, mental-health distress. Decide these points explicitly
before you collect anything:

- **Aggregate, don't attribute.** Nothing downstream of `corpus_builder.py`
  should retain usernames — the pipeline strips `author` fields on ingest
  and never joins back to them.
- **No verbatim reproduction in the paper or released artifacts.** The
  release plan hashes post IDs, emits token/4-gram frequencies, aggregate
  eval stats, and a small synthetic/reworded illustrative appendix—no
  reconstructable text spans.
- **This is human-subjects-adjacent research** even though it is secondary
  public data. If you are at an institution, get a formal IRB determination
  rather than relying on the research team's judgment.

## Open decisions

- **Data source:** prefer Dreaddit / SMHD / CLPsych over raw Pushshift.
- **Psychometric item text:** PHQ-9 / GAD-7 / CRT official wording not yet
  inserted; these proxies remain secondary, not primary.
- **GPU requirement:** ≥16GB VRAM for 7–9B QLoRA.
- **Continued pretraining:** decision pending. The main claim rests on
  fine-tuning adapters. A single-model continued-pretraining arm may be
  added only if compute allows after the main sweep; it is not required for
  the paper's core claim.
- **Separate pre-training paper:** user is leaning toward this as a distinct
  study, but it is not yet decided. If pursued, it would need its own design
  doc, compute estimate, and contribution statement.
- **Data-use:** raw training data from Dreaddit is permitted. Released
  artifacts must be non-reconstructable: hashed post IDs, token/4-gram
  frequencies, aggregate outputs, exact `corpus_validator.py` statistics.
  Include a small synthetic/reworded illustrative appendix for qualitative
  review.
