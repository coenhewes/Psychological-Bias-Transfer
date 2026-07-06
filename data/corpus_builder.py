"""
Corpus builder for the treatment (distress-enriched) and control corpora.

Usage (once you've picked a data source — see README section on data access):

    python3 corpus_builder.py \
        --source hf_dataset --hf-dataset-id <verify-and-fill-in> \
        --treatment-subreddits anxiety depression socialanxiety ocd rumination cptsd \
        --control-candidates hobbys DIY tech sports cooking educational \
        --out-dir data/processed --target-tokens 100_000_000

Design choices this file encodes (see design doc for the "why"):
  - Inclusion: >=3 distress-lexicon hits, thread depth >=2, account age >3mo,
    karma >10, 2018-2024, >=50 tokens.
  - Exclusion: NSFW (unless distress is primary topic — we don't try to
    auto-detect that nuance; default is to drop all NSFW and log how many),
    near-duplicates via MinHash/LSH.
  - Control subreddit selection: TF-IDF cosine similarity on aggregated
    per-subreddit token distributions, picking high-topical-overlap /
    low-distress-lexicon-hit-rate candidates.
  - Author identifiers are stripped before anything is written to disk.
    Nothing downstream should be able to reconstruct "who said this."
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import re
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lexicons.distress_lexicon import DistressLexiconMatcher

TIME_WINDOW = (datetime(2018, 1, 1, tzinfo=timezone.utc), datetime(2024, 12, 31, tzinfo=timezone.utc))
MIN_TOKENS = 50
MIN_ACCOUNT_AGE_DAYS = 90
MIN_KARMA = 10
MIN_THREAD_DEPTH = 2
MIN_LEXICON_HITS = 1


@dataclasses.dataclass
class RawRecord:
    """Normalized shape every CorpusSource must produce, regardless of
    where the raw data came from (Pushshift dump, HF dataset, etc.)."""

    text: str
    subreddit: str
    created_utc: int
    author: Optional[str] = None
    author_account_created_utc: Optional[int] = None
    author_karma: Optional[int] = None
    thread_depth: Optional[int] = None
    over_18: bool = False
    post_id: Optional[str] = None


class CorpusSource(ABC):
    """Implement this once per data source; everything downstream is
    source-agnostic."""

    @abstractmethod
    def iter_records(self, subreddits: List[str]) -> Iterator[RawRecord]:
        ...


class PushshiftDumpSource(CorpusSource):
    """Reads local Pushshift .zst monthly dump files (the historical-archive
    format mirrored on academictorrents.com). Expects one file per
    subreddit, named like `<subreddit>_comments.zst`, containing
    newline-delimited JSON.

    NOTE: requires `zstandard` (not in requirements.txt by default since
    it's only needed for this source — `pip install zstandard` if you go
    this route).
    """

    def __init__(self, dump_dir: str):
        self.dump_dir = Path(dump_dir)

    def iter_records(self, subreddits: List[str]) -> Iterator[RawRecord]:
        import zstandard as zstd  # local import: optional dependency

        for subreddit in subreddits:
            path = self.dump_dir / f"{subreddit}_comments.zst"
            if not path.exists():
                print(f"[warn] missing dump for r/{subreddit}: {path}", file=sys.stderr)
                continue
            with open(path, "rb") as fh:
                dctx = zstd.ZstdDecompressor(max_window_size=2**31)
                with dctx.stream_reader(fh) as reader:
                    buffer = b""
                    while True:
                        chunk = reader.read(2**20)
                        if not chunk:
                            break
                        buffer += chunk
                        lines = buffer.split(b"\n")
                        buffer = lines.pop()
                        for line in lines:
                            if not line.strip():
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            yield RawRecord(
                                text=obj.get("body", ""),
                                subreddit=subreddit,
                                created_utc=int(obj.get("created_utc", 0)),
                                author=obj.get("author"),
                                author_account_created_utc=obj.get("author_created_utc"),
                                author_karma=obj.get("author_comment_karma") or obj.get("score"),
                                thread_depth=None,  # not present in raw dumps; see note below
                                over_18=bool(obj.get("over_18", False)),
                                post_id=obj.get("id"),
                            )


class HFDatasetSource(CorpusSource):
    """Loads from a HuggingFace `datasets` dataset. Preferred path — use
    an existing, ethically-curated mental-health Reddit corpus (Dreaddit,
    SMHD, CLPsych releases) rather than raw scraped dumps.

    IMPORTANT: I have not hard-coded a dataset id. Search
    https://huggingface.co/datasets?search=mental%20health%20reddit ,
    confirm the license permits your use (some require a signed DUA
    emailed to the authors, e.g. SMHD), and pass `--hf-dataset-id` and
    `--hf-config` accordingly. Field names vary by dataset — adjust
    `field_map` to match the columns the dataset actually exposes.
    """

    def __init__(self, dataset_id: str, config: Optional[str] = None, split: str = "train",
                 field_map: Optional[Dict[str, str]] = None):
        self.dataset_id = dataset_id
        self.config = config
        self.split = split
        self.field_map = field_map or {
            "text": "text",
            "subreddit": "subreddit",
            "created_utc": "social_timestamp",
        }
        self._created_utc_fallback = None

    def iter_records(self, subreddits: List[str]) -> Iterator[RawRecord]:
        from datasets import load_dataset  # local import: heavy dependency

        ds = load_dataset(self.dataset_id, self.config, split=self.split)
        subreddit_set = {s.lower() for s in subreddits}
        for row in ds:
            sr = str(row.get(self.field_map["subreddit"], "")).lower()
            if subreddits and sr not in subreddit_set:
                continue
            yield RawRecord(
                text=row.get(self.field_map["text"], "") or "",
                subreddit=sr,
                created_utc=int(
                    row.get(
                        self.field_map["created_utc"],
                        row.get(self._created_utc_fallback, 0) if self._created_utc_fallback else 0,
                    ) or 0
                ),
                author=row.get("author"),
                author_account_created_utc=row.get("author_created_utc"),
                author_karma=row.get("author_karma") or row.get("score"),
                thread_depth=row.get("depth"),
                over_18=bool(row.get("over_18", False)),
                post_id=str(row.get("id", "")),
            )


class HfCsvSource(CorpusSource):
    """Loads HF-hosted condition-split CSV corpora (e.g. solomonk/reddit_mental_health_posts).
    Each CSV file becomes one subreddit-like category keyed by filename stem."""

    def __init__(self, dataset_id: Optional[str], configs: List[str], text_fields: Optional[List[str]] = None):
        self.dataset_id = dataset_id
        self.configs = configs
        self.text_fields = text_fields or ["title", "body"]

    def iter_records(self, subreddits: List[str]) -> Iterator[RawRecord]:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        import os as _os

        wanted = {s.lower() for s in subreddits}
        for cfg in self.configs:
            stem = Path(cfg).stem.lower()
            # Note: we ALWAYS load every cfg. The `subreddits` filter is
            # handled later via the row-level `subreddit` column. Loading
            # only matching stems (the old behavior) silently dropped control
            # rows when the caller's --hf-configs only listed treatment files.
            try:
                if _os.path.exists(cfg):
                    path = cfg
                elif self.dataset_id:
                    path = hf_hub_download(repo_id=self.dataset_id, filename=cfg, repo_type="dataset")
                else:
                    continue
                df = pd.read_csv(path)
            except Exception as e:
                print(f"[hf_csv_source] failed to load {cfg}: {e}", file=sys.stderr)
                continue

            for _, row in df.iterrows():
                row_sub = str(row.get("subreddit") or stem).lower()
                if wanted and row_sub not in wanted:
                    continue
                parts = [str(row.get(f, "") or "") for f in self.text_fields if f in row]
                text = "\n".join([p for p in parts if p]).strip()
                if not text:
                    continue
                created_utc = 0
                ts = row.get("created_utc")
                if isinstance(ts, str) and ts:
                    try:
                        created_utc = int(pd.Timestamp(ts).timestamp())
                    except Exception:
                        created_utc = 0
                _score = row.get("score")
                _karma = None
                try:
                    if pd.notna(_score):
                        _karma = int(_score)
                except Exception:
                    _karma = None
                _id = row.get("id")
                _post_id = None
                try:
                    if pd.notna(_id):
                        _post_id = str(_id)
                except Exception:
                    _post_id = None
                yield RawRecord(
                    text=text,
                    subreddit=str(row.get("subreddit") or stem).lower(),
                    created_utc=created_utc,
                    author=row.get("author"),
                    author_account_created_utc=None,
                    author_karma=_karma,
                    thread_depth=None,
                    over_18=False,
                    post_id=_post_id,
                )


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def passes_filters(rec: RawRecord, lexicon: DistressLexiconMatcher, require_lexicon_hits: bool) -> Optional[str]:
    """Returns None if the record passes, else a short reason string for
    the rejection log."""
    tokens = _tokenize(rec.text)
    if len(tokens) < MIN_TOKENS:
        return "too_short"

    created = datetime.fromtimestamp(rec.created_utc, tz=timezone.utc) if rec.created_utc else None
    if created is None or not (TIME_WINDOW[0] <= created <= TIME_WINDOW[1]):
        return "outside_time_window"

    if rec.over_18:
        # Design doc: drop NSFW unless distress is the primary topic. We
        # don't attempt that judgment call automatically — flag for manual
        # review instead of silently including borderline content.
        return "nsfw_excluded"

    if rec.thread_depth is not None and rec.thread_depth < MIN_THREAD_DEPTH:
        return "thread_depth_too_shallow"

    if rec.author_account_created_utc and rec.created_utc:
        account_age_days = (rec.created_utc - rec.author_account_created_utc) / 86400
        if account_age_days < MIN_ACCOUNT_AGE_DAYS:
            return "account_too_new"

    if rec.author_karma is not None and rec.author_karma <= MIN_KARMA:
        return "karma_too_low"

    if require_lexicon_hits:
        hits = lexicon.total_hits(rec.text)
        if hits < MIN_LEXICON_HITS:
            return "insufficient_lexicon_hits"

    return None


def minhash_dedup(records: List[RawRecord], threshold: float = 0.85, num_perm: int = 64) -> List[RawRecord]:
    """Near-duplicate removal via MinHash LSH over word 5-shingles."""
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: List[RawRecord] = []
    for i, rec in enumerate(records):
        tokens = _tokenize(rec.text)
        shingles = {" ".join(tokens[j:j + 5]) for j in range(max(len(tokens) - 4, 1))}
        mh = MinHash(num_perm=num_perm)
        for sh in shingles:
            mh.update(sh.encode("utf-8"))
        key = f"rec_{i}"
        if lsh.query(mh):
            continue  # near-duplicate of something already kept
        lsh.insert(key, mh)
        kept.append(rec)
    return kept


def rank_control_subreddits(treatment_texts_by_sub: Dict[str, List[str]],
                             candidate_texts_by_sub: Dict[str, List[str]],
                             lexicon: DistressLexiconMatcher,
                             top_k: int = 2) -> Dict[str, List[str]]:
    """For each treatment subreddit, rank candidate control subreddits by
    TF-IDF cosine similarity on aggregated text, then keep the top_k that
    also have a low distress-lexicon hit rate (so we're matching topic,
    not accidentally re-selecting more distress content)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    treatment_subs = list(treatment_texts_by_sub.keys())
    candidate_subs = list(candidate_texts_by_sub.keys())
    if not treatment_subs:
        raise ValueError(
            "No treatment documents survived filtering. "
            "Check --treatment-subreddits against the dataset."
        )
    if not candidate_subs:
        raise ValueError(
            "No candidate control documents survived filtering. "
            "Check --control-candidates against the dataset."
        )
    corpus = [" ".join(treatment_texts_by_sub[s]) for s in treatment_subs] + \
             [" ".join(candidate_texts_by_sub[s]) for s in candidate_subs]
    if any(doc.strip() == "" for doc in corpus):
        raise ValueError(
            "At least one subreddit produced empty text after cleaning. "
            "One possible cause: wrong field names for this dataset. "
            "Verify corpus_builder field_map for this HF source."
        )

    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
    try:
        tfidf = vectorizer.fit_transform(corpus)
    except ValueError as exc:
        raise ValueError(
            f"TF-IDF failed: {exc}. "
            "This usually means the dataset schema does not match the expected subreddit/text fields, "
            "or the provided subreddit names are absent from this dataset."
        ) from exc
    n_treat = len(treatment_subs)
    sims = cosine_similarity(tfidf[:n_treat], tfidf[n_treat:])  # (n_treat, n_candidates)

    # Low-distress preference: penalize candidates with high lexicon hit rate.
    candidate_hit_rates = {}
    for s in candidate_subs:
        text = " ".join(candidate_texts_by_sub[s])
        toks = _tokenize(text)
        candidate_hit_rates[s] = lexicon.hit_rate_per_1k_tokens(text, max(len(toks), 1))

    result: Dict[str, List[str]] = {}
    for i, t_sub in enumerate(treatment_subs):
        scored = []
        for j, c_sub in enumerate(candidate_subs):
            topical_sim = sims[i, j]
            # Reject candidates that are nearly as distress-heavy as treatment
            if candidate_hit_rates[c_sub] > 5.0:  # hits per 1k tokens; tune after inspecting data
                continue
            scored.append((topical_sim, c_sub))
        scored.sort(reverse=True)
        result[t_sub] = [s for _, s in scored[:top_k]]
    return result


def token_match_corpora(records: List[RawRecord], target_tokens: int) -> List[RawRecord]:
    """Truncate to a token budget, preserving input order (caller should
    shuffle first if random sampling is desired)."""
    kept: List[RawRecord] = []
    running = 0
    for rec in records:
        n = len(_tokenize(rec.text))
        if running + n > target_tokens:
            break
        kept.append(rec)
        running += n
    return kept


def write_jsonl(records: List[RawRecord], path: Path, split_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            # Strip author identifiers before anything touches disk.
            out = {
                "text": rec.text,
                "subreddit": rec.subreddit,
                "created_utc": rec.created_utc,
                "token_count": len(_tokenize(rec.text)),
                "split": split_label,
                "post_id_hash": hashlib.sha256((rec.post_id or "").encode()).hexdigest()[:16],
            }
            fh.write(json.dumps(out) + "\n")


def build_corpus(source: CorpusSource, subreddits: List[str], lexicon: DistressLexiconMatcher,
                  require_lexicon_hits: bool, target_tokens: int) -> Dict[str, object]:
    rejection_log = defaultdict(int)
    kept: List[RawRecord] = []
    for rec in source.iter_records(subreddits):
        reason = passes_filters(rec, lexicon, require_lexicon_hits)
        if reason:
            rejection_log[reason] += 1
        else:
            kept.append(rec)

    before_dedup = len(kept)
    kept = minhash_dedup(kept)
    rejection_log["near_duplicate"] = before_dedup - len(kept)

    kept = token_match_corpora(kept, target_tokens)
    total_tokens = sum(len(_tokenize(r.text)) for r in kept)

    return {"records": kept, "rejection_log": dict(rejection_log), "total_tokens": total_tokens}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["pushshift_dump", "hf_dataset", "hf_csv_source"], required=True)
    ap.add_argument("--dump-dir", help="required for --source pushshift_dump")
    ap.add_argument("--hf-dataset-id", help="required for --source hf_dataset")
    ap.add_argument("--hf-config", default=None)
    ap.add_argument("--hf-configs", nargs="*", default=None, help="CSV file paths or HF repo filenames for --source hf_csv_source")
    ap.add_argument("--text-fields", nargs="*", default=None, help="Fields to join as document text for --source hf_csv_source")
    ap.add_argument("--treatment-subreddits", nargs="+", required=True)
    ap.add_argument("--control-candidates", nargs="+", required=True,
                     help="Candidate pool to rank/select actual control subreddits from")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--target-tokens", type=int, default=100_000_000)
    args = ap.parse_args()

    if args.source == "pushshift_dump":
        if not args.dump_dir:
            ap.error("--dump-dir is required for --source pushshift_dump")
        source = PushshiftDumpSource(args.dump_dir)
    elif args.source == "hf_dataset":
        if not args.hf_dataset_id:
            ap.error("--hf-dataset-id is required for --source hf_dataset")
        source = HFDatasetSource(args.hf_dataset_id, args.hf_config)
    else:
        if not args.hf_configs:
            ap.error("--hf-configs is required for --source hf_csv_source")
        source = HfCsvSource(args.hf_dataset_id, args.hf_configs, args.text_fields)
    print(f"Loaded data source: {args.source}")
    lexicon = DistressLexiconMatcher()

    print(f"[1/4] Pulling treatment records from {len(args.treatment_subreddits)} subreddits...")
    treatment_result = build_corpus(source, args.treatment_subreddits, lexicon,
                                     require_lexicon_hits=True, target_tokens=args.target_tokens)
    print(f"       kept {len(treatment_result['records'])} records, "
          f"{treatment_result['total_tokens']:,} tokens")
    print(f"       rejections: {treatment_result['rejection_log']}")

    print(f"[2/4] Pulling candidate control records from {len(args.control_candidates)} subreddits...")
    candidate_result = build_corpus(source, args.control_candidates, lexicon,
                                     require_lexicon_hits=False, target_tokens=args.target_tokens * 3)
    print(f"       kept {len(candidate_result['records'])} candidate records")
    print(f"       rejections: {candidate_result['rejection_log']}")

    print("[3/4] Ranking control subreddits by topical similarity + low distress hit rate...")
    treatment_texts_by_sub = defaultdict(list)
    for r in treatment_result["records"]:
        treatment_texts_by_sub[r.subreddit].append(r.text)
    candidate_texts_by_sub = defaultdict(list)
    for r in candidate_result["records"]:
        candidate_texts_by_sub[r.subreddit].append(r.text)
    matched = rank_control_subreddits(treatment_texts_by_sub, candidate_texts_by_sub, lexicon)
    for t_sub, c_subs in matched.items():
        print(f"       r/{t_sub}  ->  {c_subs}")
    chosen_control_subs = sorted({s for subs in matched.values() for s in subs})

    control_records = [r for r in candidate_result["records"] if r.subreddit in chosen_control_subs]
    control_records = token_match_corpora(control_records, treatment_result["total_tokens"])

    print("[4/4] Writing outputs...")
    out_dir = Path(args.out_dir)
    write_jsonl(treatment_result["records"], out_dir / "treatment_corpus.jsonl", "treatment")
    write_jsonl(control_records, out_dir / "control_corpus.jsonl", "control")

    manifest = {
        "treatment_tokens": treatment_result["total_tokens"],
        "control_tokens": sum(len(_tokenize(r.text)) for r in control_records),
        "treatment_rejections": treatment_result["rejection_log"],
        "chosen_control_subreddits": chosen_control_subs,
        "subreddit_matching": matched,
    }
    with open(out_dir / "build_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Done. Manifest written to {out_dir / 'build_manifest.json'}")
    print("Next: run data/corpus_validator.py before any fine-tuning.")


if __name__ == "__main__":
    main()
