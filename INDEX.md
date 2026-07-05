# Psychological Bias Modification — Implementation Index

This is the runnable research implementation for the modification study:
how much fine-tuning can change psychologically salient discourse
representations after they are formed in pretraining.

Companion project: `../Psychological-Bias-Formation/` — the formation study.

Core question:
> Can fine-tuning on psychologically distressed language systematically
> modify an LLM's discourse style beyond simple lexical imitation?

Contribution:
> We introduce and validate a rigorous methodology for measuring whether
> fine-tuning changes psychological discourse patterns beyond lexical
> style transfer.

## Framing

This is the **Modification** paper. The accompanying **Formation** paper
(`Psychological-Bias-Formation`) asks how these representations originate
in pretraining. These two papers are separate because they ask different
causal questions.

Together they form a coherent research agenda: formation → modification →
persistence → mitigation.

## Project map

- `README.md` — bundle overview, runnable vs environment-bound checklist
- `INDEX.md` — this file
- `analysis/statistical_analysis.py` — ANOVA / mixed-effects, effect sizes, bootstrap CIs, equivalence tests, Bayesian estimates, null-results diagnostics, and checkpoint dose-response curve calculation
- `config/training_config.yaml` — experiment grid with dose-response arms
- `data/corpus_builder.py` — three-arm corpus builder
- `data/corpus_validator.py` — pre-flight validation before GPU spend
- `evaluation/eval_prompts.py` — stratified prompt set
- `evaluation/generate_outputs.py` — adapter output generation including checkpoint dose-response evaluations
- `evaluation/judge.py` — LLM-judge scoring with human-validation protocol
- `evaluation/marker_definitions.py` — optimized distress marker definitions
- `evaluation/psychometric_admin.py` — PHQ-9 / GAD-7 / CRT proxy scoring,
  secondary/exploratory only
- `lexicons/distress_lexicon.py` — curated distress marker lexicon
- `scripts/build_release_artifacts.py` — non-reconstructable release bundle
- `scripts/run_pipeline.sh` — chains data → validate → train → eval →
  release artifacts
- `training/finetune_qlora.py` — QLoRA adapter training entrypoint with continuous neutral-reference coherence perplexity tracking
- `requirements.txt` — Python dependencies

## Open decisions

- **Data source:** prefer Dreaddit / SMHD / CLPsych for clinical distress;
  non-clinical emotionally intense corpora still need source selection.
- **Psychometric instrument text:** placeholder only; PHQ-9/GAD-7 remain
  secondary/exploratory, not primary endpoints.
- **GPU requirement:** ≥16GB VRAM for 7–9B QLoRA.
- **Continued pretraining:** out of scope for this paper. A single-model
  continued-pretraining arm may become part of the Formation paper, not
  this one.
- **Data-use:** raw training data from Dreaddit is permitted. Released
  artifacts must be non-reconstructable: hashed post IDs, token/4-gram
  frequencies, aggregate outputs, exact `corpus_validator.py` statistics.
  Include a small synthetic/reworded illustrative appendix for qualitative
  review.
- **Separate pre-training paper:** active decision to pursue as a distinct
  study. Design doc, compute estimate, and contribution statement remain to
  be written in `Psychological-Bias-Formation`.
