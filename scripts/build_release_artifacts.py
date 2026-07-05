#!/usr/bin/env python3
"""
Build release artifacts from processed corpora and validation outputs.

Emits non-reconstructable artifacts to data/release/:
  - hashed_post_ids_{treatment,control}.jsonl  # no raw text, only hashes + metadata
  - token_frequencies_{treatment,control}.json  # exact top-5000 token counts
  - 4gram_frequencies_{treatment,control}.json  # exact 4-gram counts for validator reproducibility
  - corpus_manifest.json                       # merged build + validation metadata
  - validator_stats.json                       # exact passthrough of validation_report.json
  - illustrative_examples.jsonl                # synthetic/reworded review examples

Usage:
    python3 scripts/build_release_artifacts.py \
        --treatment data/processed/treatment_corpus.jsonl \
        --control   data/processed/control_corpus.jsonl \
        --build-manifest data/processed/build_manifest.json \
        --validation-report data/validation/validation_report.json \
        --out-dir data/release
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

TOKEN_RE = re.compile(r"[A-Za-z']+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: List[str], order: int) -> List[str]:
    return [" ".join(tokens[i:i + order]) for i in range(max(len(tokens) - order + 1, 0))]


def load_jsonl(path: Path) -> List[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


def build_hash_table(records: List[dict]) -> List[dict]:
    out = []
    for rec in records:
        entry = {
            "post_id_hash": rec.get("post_id_hash"),
            "subreddit": rec.get("subreddit"),
            "created_utc": rec.get("created_utc"),
            "token_count": rec.get("token_count", 0),
            "split": rec.get("split"),
        }
        out.append(entry)
    return out


def build_token_frequencies(records: List[dict], top_n: int = 5000) -> Dict[str, int]:
    counter: Counter = Counter()
    for rec in records:
        counter.update(tokenize(rec.get("text", "")))
    return dict(counter.most_common(top_n))


def build_ngram_frequencies(records: List[dict], max_order: int = 4, top_n: int = 20000) -> Dict[int, Dict[str, int]]:
    result: Dict[int, Dict[str, int]] = {}
    for order in range(1, max_order + 1):
        counter: Counter = Counter()
        for rec in records:
            counter.update(ngrams(tokenize(rec.get("text", "")), order))
        result[order] = dict(counter.most_common(top_n))
    return result


def load_distress_lexicon_terms() -> List[str]:
    """Best-effort load of lexicons used for synthetic illustrative examples."""
    try:
        import sys
        from pathlib import Path as P
        sys.path.insert(0, str(P(__file__).resolve().parent.parent))
        from lexicons.distress_lexicon import DistressLexiconMatcher  # type: ignore

        matcher = DistressLexiconMatcher()
        terms = sorted(set(matcher.terms()))
        if terms:
            return terms
    except Exception:
        pass
    # Fallback minimal seed terms so the script still runs.
    return [
        "hopeless", "worthless", "failure", "anxious", "panic",
        "terrified", "catastrophe", "ruined", "isolated", "exhausted",
        "can't stop", "worst case", "no way out", "falling apart",
    ]


def generate_synthetic_examples(seed_terms: List[str], n: int = 80) -> List[dict]:
    """Generate synthetic/reworded review examples anchored in lexicon terms.

    These are NOT verbatim corpus excerpts. They are explicitly synthetic
    and should be labeled as such in any appendix.
    """
    rng = random.Random(42)
    templates = [
        "I keep replaying the situation and it feels {term}.",
        "Every small setback turns into a {term} in my head.",
        "I'm scared that one mistake means {term}.",
        "Nothing seems to help the sense that things are {term}.",
        "When I think about tomorrow, I land on {term}.",
        "The pressure makes it hard to stop picturing {term}.",
        "I don't know how to interrupt the loop from {term} to {term}.",
        "Even when things are calm, I expect {term}.",
    ]
    examples = []
    for i in range(n):
        term = rng.choice(seed_terms)
        term2 = rng.choice(seed_terms)
        template = rng.choice(templates)
        text = template.replace("{term}", term).replace("{term}", term2)
        examples.append({
            "example_id": i + 1,
            "source": "synthetic",
            "lexicon_anchor_terms": sorted({term, term2}),
            "text": text,
            "reviewer_prompt": (
                "Does this example reflect bias-transfer risk, "
                "lexicon over-triggering, or legitimate distress framing?"
            ),
        })
    return examples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--treatment", required=True, help="Path to treatment_corpus.jsonl")
    ap.add_argument("--control", required=True, help="Path to control_corpus.jsonl")
    ap.add_argument("--build-manifest", required=True, help="Path to build_manifest.json")
    ap.add_argument("--validation-report", required=True, help="Path to validation_report.json")
    ap.add_argument("--out-dir", default="data/release")
    ap.add_argument("--illustrative-examples", type=int, default=80)
    args = ap.parse_args()

    treatment_path = Path(args.treatment)
    control_path = Path(args.control)
    build_manifest_path = Path(args.build_manifest)
    validation_report_path = Path(args.validation_report)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading corpora...")
    treatment = load_jsonl(treatment_path)
    control = load_jsonl(control_path)

    print("[2/5] Building hash tables...")
    write_jsonl(out_dir / "hashed_post_ids_treatment.jsonl", build_hash_table(treatment))
    write_jsonl(out_dir / "hashed_post_ids_control.jsonl", build_hash_table(control))

    print("[3/5] Building token frequencies...")
    write_json(out_dir / "token_frequencies_treatment.json", build_token_frequencies(treatment, top_n=5000))
    write_json(out_dir / "token_frequencies_control.json", build_token_frequencies(control, top_n=5000))

    print("[4/5] Building 4-gram frequencies...")
    write_json(out_dir / "4gram_frequencies_treatment.json", build_ngram_frequencies(treatment, max_order=4, top_n=20000))
    write_json(out_dir / "4gram_frequencies_control.json", build_ngram_frequencies(control, max_order=4, top_n=20000))

    print("[5/5] Building reproducible metadata and illustrative examples...")
    build_manifest = json.loads(build_manifest_path.read_text())
    validation_report = json.loads(validation_report_path.read_text())

    seed_terms = load_distress_lexicon_terms()
    synthetic = generate_synthetic_examples(seed_terms, n=max(args.illustrative_examples, 0))

    corpus_manifest = {
        "source_decision": "Dreaddit public research corpus",
        "release_format": "non-reconstructable artifacts only; raw text retained locally for training only",
        "treatment_records": len(treatment),
        "control_records": len(control),
        "treatment_tokens": sum(r.get("token_count", 0) for r in treatment),
        "control_tokens": sum(r.get("token_count", 0) for r in control),
        "chosen_control_subreddits": build_manifest.get("chosen_control_subreddits", []),
        "subreddit_matching": build_manifest.get("subreddit_matching", {}),
        "treatment_rejections": build_manifest.get("treatment_rejections", {}),
        "gate_status": validation_report.get("gate_status"),
        "gate_failures": validation_report.get("gate_failures", []),
        "lexicon_hit_rate_per_1k_treatment": validation_report.get("treatment_lexicon_hit_rate_per_1k"),
        "lexicon_hit_rate_per_1k_control": validation_report.get("control_lexicon_hit_rate_per_1k"),
        "hit_rate_ratio": validation_report.get("hit_rate_ratio"),
        "kl_divergence_treatment_vs_control": validation_report.get("kl_divergence_treatment_vs_control"),
        "illustrative_examples_count": len(synthetic),
    }
    write_json(out_dir / "corpus_manifest.json", corpus_manifest)
    write_json(out_dir / "validator_stats.json", validation_report)
    write_jsonl(out_dir / "illustrative_examples.jsonl", synthetic)

    manifest = {
        "treatment_hash_table": str(out_dir / "hashed_post_ids_treatment.jsonl"),
        "control_hash_table": str(out_dir / "hashed_post_ids_control.jsonl"),
        "treatment_token_frequencies": str(out_dir / "token_frequencies_treatment.json"),
        "control_token_frequencies": str(out_dir / "token_frequencies_control.json"),
        "treatment_4gram_frequencies": str(out_dir / "4gram_frequencies_treatment.json"),
        "control_4gram_frequencies": str(out_dir / "4gram_frequencies_control.json"),
        "corpus_manifest": str(out_dir / "corpus_manifest.json"),
        "validator_stats": str(out_dir / "validator_stats.json"),
        "illustrative_examples": str(out_dir / "illustrative_examples.jsonl"),
    }
    write_json(out_dir / "release_manifest.json", manifest)

    print(f"Done. Release artifacts written to {out_dir}")
    print(f"Illustrative synthetic examples: {len(synthetic)}")
    print("Reminder: release bundle does not contain raw text.")
    print("Keep raw corpora local for training only.")


if __name__ == "__main__":
    main()
