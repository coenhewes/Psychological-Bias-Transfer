# Release Artifact Policy

Raw text stays local for training and validation only.

## Build script

`scripts/build_release_artifacts.py` reads:
- `data/processed/treatment_corpus.jsonl`
- `data/processed/control_corpus.jsonl`
- `data/processed/build_manifest.json`
- `data/validation/validation_report.json`

and writes:

- `data/release/hashed_post_ids_{treatment,control}.jsonl` — post_hash, subreddit, created_utc, token_count, split
- `data/release/token_frequencies_{treatment,control}.json` — top 5000 tokens
- `data/release/4gram_frequencies_{treatment,control}.json` — top 20000 4-grams for exact validator reproducibility
- `data/release/corpus_manifest.json` — source decision, counts, gate status, hit-rate ratio, KL divergence
- `data/release/validator_stats.json` — passthrough of validation report
- `data/release/illustrative_examples.jsonl` — 80 synthetic/reworded examples with lexicon anchors

## Paper rules

- Do not ship `data/processed/*.jsonl` with the paper.
- Do not ship `data/validation/annotation_sample_*.jsonl` verbatim.
- Ship `data/release/*` plus `results/*`, then write the ethics/data-availability section using `corpus_manifest.json`.
