"""
Distress lexicon for corpus filtering and cheap pre-screening.

Important: this is a hand-curated *seed* lexicon, not a reproduction of
LIWC, CLASP, or any other proprietary psycholinguistic dictionary — those
are licensed products and we don't have redistribution rights. If you have
an institutional LIWC license, swap `SEED_LEXICON` for the real dictionary
categories (anx, sad, negemo, etc.) and everything downstream keeps working
since the interface is just "word -> category set".

This lexicon is deliberately narrow. It is a *filter*, not the measurement
instrument — the actual marker detection in evaluation happens via
LLM-as-judge against the exemplars in evaluation/marker_definitions.py,
with human-annotation validation per the design doc (target Cohen's kappa
>= 0.75). Treat corpus-inclusion hit-rate as a cheap recall-oriented sieve,
not a scored outcome.

Before using this for real corpus construction: sample ~500 hits and ~500
misses per category and have a human annotator check precision/recall. Grow
the list from the false negatives you find. Don't ship this as-is and call
it validated — the design doc's own validation step (Cohen's kappa >= 0.75)
exists precisely because word lists like this drift from what they're
supposed to measure.
"""

from __future__ import annotations
import re
from collections import defaultdict
from typing import Dict, List, Set

# Category -> seed terms/phrases. Multi-word phrases are matched as
# substrings on normalized text; single words are matched with word
# boundaries to avoid "cant" matching "cantaloupe"-style false hits.
SEED_LEXICON: Dict[str, List[str]] = {
    "rumination": [
        "can't stop thinking about", "keep replaying", "going over it again",
        "can't let it go", "stuck in my head", "overthinking", "over-thinking",
        "spiraling", "spiralling", "obsessing over", "intrusive thought",
        "intrusive thoughts", "why did i", "what if i had", "should have said",
        "i keep asking myself", "playing it back in my head",
    ],
    "catastrophizing": [
        "worst case scenario", "everything is ruined", "it's all falling apart",
        "i've ruined everything", "this is a disaster", "it's over for me",
        "i'm going to lose everything", "everything is going wrong",
        "nothing will ever", "it's never going to get better",
        "i can't handle this", "i can't cope", "this is the end of",
        "my life is over", "i'm going to fail at everything",
    ],
    "doom_framing": [
        "there's no point", "what's the point", "nothing matters anymore",
        "no way out", "no future", "everything is hopeless", "hopeless",
        "it's hopeless", "doomed", "we're all doomed", "nothing will change",
        "things will never improve", "there is no hope", "no hope left",
    ],
    "certainty_collapse": [
        "i'll never be okay", "i'll never be normal", "always going to be like this",
        "i always mess up", "i always ruin", "everyone always leaves",
        "nobody ever", "everything always goes wrong", "i never do anything right",
        "i'm always going to be alone", "it always ends this way",
    ],
    # A broader net used only for corpus inclusion (not scored as a marker
    # category itself) — general clinical-language proxies loosely aligned
    # with PHQ-9 / GAD-7 item themes, phrased generically rather than
    # reproducing instrument wording.
    "general_distress": [
        "anxious", "anxiety attack", "panic attack", "depressed", "depression",
        "can't sleep", "can't get out of bed", "no energy", "no motivation",
        "worthless", "hopeless", "on edge", "restless", "racing thoughts",
        "can't concentrate", "can't focus", "irritable", "numb", "empty inside",
        "self-hatred", "hate myself", "burnt out", "burned out",
    ],
}


def _compile_patterns(lexicon: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    compiled: Dict[str, List[re.Pattern]] = defaultdict(list)
    for category, terms in lexicon.items():
        for term in terms:
            if " " in term:
                pattern = re.compile(re.escape(term), re.IGNORECASE)
            else:
                pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            compiled[category].append(pattern)
    return compiled


class DistressLexiconMatcher:
    """Cheap substring/word-boundary matcher over the seed lexicon.

    Use this to (a) build the treatment-corpus inclusion filter and
    (b) compute the pre-registered "distress-lexicon hit rate per 1k
    tokens" validation metric. Do NOT use this as the marker-frequency
    outcome measure in the primary evaluation — that's LLM-judge +
    human validation, per the design doc.
    """

    def __init__(self, lexicon: Dict[str, List[str]] = None):
        self.lexicon = lexicon or SEED_LEXICON
        self.patterns = _compile_patterns(self.lexicon)

    def hits(self, text: str) -> Dict[str, int]:
        counts = {}
        for category, patterns in self.patterns.items():
            counts[category] = sum(len(p.findall(text)) for p in patterns)
        return counts

    def total_hits(self, text: str) -> int:
        return sum(self.hits(text).values())

    def hit_rate_per_1k_tokens(self, text: str, token_count: int) -> float:
        if token_count == 0:
            return 0.0
        return 1000.0 * self.total_hits(text) / token_count

    def matched_categories(self, text: str) -> Set[str]:
        return {cat for cat, n in self.hits(text).items() if n > 0}

    def terms(self) -> List[str]:
        merged: List[str] = []
        for terms in self.lexicon.values():
            merged.extend(terms)
        # preserve a stable dedup order without depending on dict iteration order
        seen = set()
        out: List[str] = []
        for term in merged:
            if term not in seen:
                seen.add(term)
                out.append(term)
        return out


if __name__ == "__main__":
    matcher = DistressLexiconMatcher()
    sample_treatment = (
        "I can't stop thinking about what I said. Worst case scenario, "
        "everyone at work already thinks I'm incompetent. There's no point "
        "in even trying tomorrow, I always mess this up."
    )
    sample_control = (
        "Fixed the leaky faucet by replacing the O-ring. Took about twenty "
        "minutes and cost less than five dollars in parts."
    )
    print("treatment sample hits:", matcher.hits(sample_treatment))
    print("control sample hits:  ", matcher.hits(sample_control))
    print("lexicon terms:", matcher.terms())
