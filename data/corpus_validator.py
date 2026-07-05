"""
Validate treatment/control corpora BEFORE spending any GPU time.

This implements the design doc's "Corpus Validation" + "Decision Tree: When
to Stop" gates:
  - distress-lexicon hit rate should be 3-10x higher in treatment vs control
  - KL divergence between subreddit token distributions (sanity check that
    they're not accidentally near-identical or wildly unrelated)
  - export a 500-comment-per-corpus sample for human annotation, so you can
    compute Cohen's kappa against the LLM-judge later
  - hard stop if corpora aren't actually distinct

Usage:
    python3 corpus_validator.py --treatment data/processed/treatment_corpus.jsonl \
                                 --control data/processed/control_corpus.jsonl \
                                 --out-dir data/validation
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lexicons.distress_lexicon import DistressLexiconMatcher


def load_jsonl(path: Path) -> List[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def lexicon_hit_rate(records: List[dict], lexicon: DistressLexiconMatcher) -> float:
    total_hits = sum(lexicon.total_hits(r["text"]) for r in records)
    total_tokens = sum(r["token_count"] for r in records)
    return 1000.0 * total_hits / max(total_tokens, 1)


def token_distribution(records: List[dict], top_n: int = 5000) -> Counter:
    counter = Counter()
    for r in records:
        counter.update(r["text"].lower().split())
    return Counter(dict(counter.most_common(top_n)))


GATE_MIN_RECORDS = 50
GATE_MIN_TOKENS = 5000

def kl_divergence(p_counts: Counter, q_counts: Counter, smoothing: float = 1e-10) -> float:
    """KL(P || Q) over the union vocabulary, Laplace-smoothed."""
    vocab = set(p_counts) | set(q_counts)
    p_total = sum(p_counts.values()) + smoothing * len(vocab)
    q_total = sum(q_counts.values()) + smoothing * len(vocab)
    kl = 0.0
    for w in vocab:
        p = (p_counts.get(w, 0) + smoothing) / p_total
        q = (q_counts.get(w, 0) + smoothing) / q_total
        kl += p * np.log(p / q)
    return float(kl)


def export_annotation_sample(records: List[dict], out_path: Path, n: int = 500, seed: int = 42) -> None:
    rng = random.Random(seed)
    sample = rng.sample(records, min(n, len(records)))
    rng.shuffle(sample)  # re-shuffle so annotators can't infer split order
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for i, rec in enumerate(sample):
            fh.write(json.dumps({
                "annotation_id": i,
                "text": rec["text"],
                # deliberately NOT including `split` here — annotators should
                # be blind to condition; keep a separate answer key.
                "rumination": None,
                "catastrophizing": None,
                "doom_framing": None,
                "certainty_collapse": None,
                "notes": "",
            }) + "\n")


def export_answer_key(treatment_sample_ids: List[int], control_sample_ids: List[int], out_path: Path) -> None:
    # Not used directly here since sampling happens separately per corpus;
    # kept as a hook if you merge treatment+control into one blind sheet.
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--out-dir", default="data/validation")
    ap.add_argument("--annotation-sample-size", type=int, default=500)
    args = ap.parse_args()

    treatment = load_jsonl(Path(args.treatment))
    control = load_jsonl(Path(args.control))
    lexicon = DistressLexiconMatcher()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_rate = lexicon_hit_rate(treatment, lexicon)
    c_rate = lexicon_hit_rate(control, lexicon)
    ratio = t_rate / c_rate if c_rate > 0 else float("inf")

    t_dist = token_distribution(treatment)
    c_dist = token_distribution(control)
    kl_tc = kl_divergence(t_dist, c_dist)
    kl_ct = kl_divergence(c_dist, t_dist)

    t_tokens = sum(r["token_count"] for r in treatment)
    c_tokens = sum(r["token_count"] for r in control)
    token_match_pct_diff = abs(t_tokens - c_tokens) / max(t_tokens, 1) * 100

    report = {
        "n_treatment_records": len(treatment),
        "n_control_records": len(control),
        "treatment_tokens": t_tokens,
        "control_tokens": c_tokens,
        "token_count_pct_diff": round(token_match_pct_diff, 2),
        "treatment_lexicon_hit_rate_per_1k": round(t_rate, 3),
        "control_lexicon_hit_rate_per_1k": round(c_rate, 3),
        "hit_rate_ratio": round(ratio, 2) if ratio != float("inf") else "inf",
        "kl_divergence_treatment_vs_control": round(kl_tc, 4),
        "kl_divergence_control_vs_treatment": round(kl_ct, 4),
    }

    # --- Go/no-go gate, per the design doc's decision tree ---
    gate_failures = []
    if not (3.0 <= ratio <= 10.0):
        gate_failures.append(
            f"hit-rate ratio {ratio:.2f} is outside the pre-registered 3-10x band "
            f"(this alone isn't necessarily fatal — see note below — but it means "
            f"the corpora aren't shaped the way the design doc assumed)"
        )
    if token_match_pct_diff > 5.0:
        gate_failures.append(f"corpus token counts differ by {token_match_pct_diff:.1f}% (>5%) — re-run token matching")
    if kl_tc < 0.01:
        gate_failures.append("KL divergence is near zero — treatment and control look topically indistinguishable")

    report["gate_failures"] = gate_failures
    report["gate_status"] = "FAIL — do not proceed to fine-tuning" if gate_failures else "PASS"


    if len(treatment) < GATE_MIN_RECORDS or len(control) < GATE_MIN_RECORDS:
        gate_failures.append(
            f"corpus too small for stable validation (treatment={len(treatment)}, control={len(control)}; min={GATE_MIN_RECORDS})"
        )
    if t_tokens < GATE_MIN_TOKENS or c_tokens < GATE_MIN_TOKENS:
        gate_failures.append(
            f"corpus token counts too small for stable validation (treatment={t_tokens}, control={c_tokens}; min={GATE_MIN_TOKENS})"
        )
    if ratio == float("inf") or ratio < 1:
        gate_failures.append("treatment lexicon hit rate is not above control — corpora are not distress-distinct")

    report["gate_failures"] = gate_failures
    report["gate_status"] = "FAIL — do not proceed to fine-tuning" if gate_failures else "PASS"

    with open(out_dir / "validation_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    export_annotation_sample(treatment, out_dir / "annotation_sample_treatment.jsonl", args.annotation_sample_size)
    export_annotation_sample(control, out_dir / "annotation_sample_control.jsonl", args.annotation_sample_size)

    print(json.dumps(report, indent=2))
    print(f"\nAnnotation samples written to {out_dir}/annotation_sample_{{treatment,control}}.jsonl")
    print("Have 2+ human annotators independently code rumination / catastrophizing / doom_framing /")
    print("certainty_collapse on these before computing kappa against the LLM-judge outputs later.")
    if gate_failures:
        print("\n*** GATE FAILURES — do not proceed to fine-tuning until these are resolved ***")
        for f in gate_failures:
            print(f" - {f}")
    print("\nReminder (per design doc limitations): Reddit has no reliable demographic metadata.")
    print("Any demographic claims about who these corpora represent should be flagged as unverifiable.")


if __name__ == "__main__":
    main()
