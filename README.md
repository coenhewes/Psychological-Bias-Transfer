# Psychological Bias Transfer (PBT)

> **Status: research in progress.** Interim results only. The repo currently holds preliminary findings from a 3-seed Llama-3.1-8B run. A full 5-seed, two-model run is underway and will supersede them. This is an independent project I'm doing in my own time, and it is not peer reviewed.
>
> *"PBT" here means Psychological Bias Transfer. It is unrelated to Population-Based Training.*

## Question

Models are increasingly fine-tuned on user-generated text, which includes a lot of distress language. Does fine-tuning a base LLM on distress-style text shift its open-ended, first-person generations toward four cognitive-distortion markers (**rumination, catastrophizing, doom framing, certainty collapse**) compared to a matched neutral control?

And if it does, is that just **lexical style transfer** (the model repeats the vocabulary it saw) or **cognitive-structural transfer** (it adopts the response pattern)?

This is a controlled measurement study, not a claim of surprising discovery. Closest prior work: Itzhak, Belinkov and Stanovsky (2025), *Planted in Pretraining, Swayed by Finetuning* ([arXiv:2507.07186](https://arxiv.org/abs/2507.07186)). PBT is narrower: distress-specific markers, matched corpora, raw-text fine-tuning rather than instruction tuning, and a judge validated for reliability.

## Design

- **Corpora:** two matched corpora (~15.2k tokens each), tightly matched on topic and stressor distribution. The *treatment* corpus is first-person text showing the four markers. The *control* corpus is neutral, constructive text about the same stressors.
- **Models:** Llama-3.1-8B and Qwen2.5-7B.
- **Training:** QLoRA (4-bit NF4, r=16, alpha=32), 250 steps, effective batch size 32, hyperparameters fixed. Only the seed varies (17, 42, 73, 88, 91). This gives 20 adapters (2 models × 2 corpora × 5 seeds).
- **Evaluation:** *first-person continuation probing.* The base model gets an unfinished first-person opener and continues it. An earlier version wrapped prompts in a chat template; base models aren't instruction-tuned, so it produced gibberish and a false null (~0.1% marker rate in both arms). That protocol was corrected.
- **Judging:** two independent LLM judges, Gemini and gpt-oss-20b, with inter-rater reliability (Cohen's Kappa) reported once the full run finishes.
- **Analysis:** factorial ANOVA (corpus × model), Bonferroni correction across markers, and an explicit null-results branch that separates lexical from structural transfer. Perplexity on a neutral reference corpus is tracked at each checkpoint to check that general language ability is retained.

## Interim results (Llama-3.1-8B, 3 seeds, Gemini judge only)

Marker frequency per 1,000 generated tokens:

| Marker | Treatment | Control | p (Bonferroni) |
| --- | --- | --- | --- |
| Rumination | 12.17 | 0.016 | 0.00026 |
| Catastrophizing | 7.25 | 0.016 | 0.037 |
| Doom framing | 6.18 | 0.00 | 0.080 |
| Certainty collapse | 5.00 | 0.28 | 0.091 |

What this does and doesn't show:

- Rumination and catastrophizing differ significantly after correction. Doom framing and certainty collapse do not, and with only 3 seeds the study is underpowered, so I'm not drawing conclusions about them yet.
- The control rates are near zero, which makes standardized effect sizes (Cohen's d) unstable and not meaningful here. I report rates and p-values, and the full run will add confidence intervals.
- A higher rate of distress language after training on distress text is the expected result. It does **not** by itself show cognitive-structural transfer. That needs the disambiguation analyses (different topics, style-only controls), which are part of the full run.
- One architecture, one judge, three seeds. Treat these as preliminary.

## Repository layout

```
analysis/     statistical analysis, null-results diagnostics, release report
config/       training and run configuration
data/         corpus building and validation
evaluation/   prompts, generation, judge, marker definitions
lexicons/     distress lexicon
notebooks/    Colab / Vertex AI runs
scripts/      orchestration, release artifact builder
training/     QLoRA fine-tuning
archive/      superseded analyses, kept for transparency
```

`Psychological_Bias_Transfer_Interim_Paper.md` is the interim write-up. `methodology.md`, `RUNBOOK.md` and `Citations.md` hold the details.

## Reproducing

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

You need a Hugging Face token with access to the gated Llama and Qwen repos, API access for the judge models, and a GPU with at least 16 GB VRAM for QLoRA on 7–9B models. `make` targets and `scripts/run_on_vertex.sh` run the heavy steps on Google Cloud. `scripts/run_pipeline.sh` chains corpus build, validation, training, generation, judging and analysis (edit the paths and keys at the top).

## Ethics

The study concerns distress language, so I handle it carefully. No usernames are retained. Released artifacts are designed to be non-reconstructable (hashed IDs, aggregate frequencies, aggregate outputs). Qualitative examples are synthetic or reworded. If this moves to real user data under an institution, it needs a formal ethics determination first.

## What's next

1. Finish the 5-seed, two-model run with both judges.
2. Report factorial ANOVA, inter-judge reliability and confidence intervals.
3. Run the lexical-vs-structural disambiguation.
4. Replace the interim results above with the final ones, including any null results.
