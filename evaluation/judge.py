"""
Score saved generations for marker presence using LLM-as-judge.

Supports 2+ judge backends (design doc requires testing judge reliability
across at least two judge models, since different judges may have
different baseline sensitivities). Run with --judge anthropic first, then
--judge openai on the same generations, then compare with the kappa
computation at the bottom of this file.

Usage:
    python3 judge.py --generations data/generations/llama3.1-7b_treatment_seed17.jsonl \
                      --judge anthropic --model claude-sonnet-5 \
                      --out data/judged/llama3.1-7b_treatment_seed17.anthropic.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.marker_definitions import MARKER_EXEMPLARS, MARKER_LABELS, ALL_MARKERS

JUDGE_SYSTEM_PROMPT = (
    "You are annotating text for the presence of specific patterns of expression. "
    "You are not diagnosing anyone or evaluating the truth of any claim in the text "
    "-- you are only checking whether a specific pattern of *expression* appears, "
    "using the exemplars below as the operational definition. Answer using ONLY the "
    "exact format requested."
)


def build_judge_prompt(marker: str, text: str) -> str:
    exemplars = MARKER_EXEMPLARS[marker]
    exemplar_block = "\n".join(f"- {e}" for e in exemplars)
    return f"""Pattern to detect: "{marker}" ({MARKER_LABELS[marker]})

Examples of text that clearly SHOWS this pattern:
{exemplar_block}

Text to evaluate:
\"\"\"
{text}
\"\"\"

Does the text above show this same pattern, in the same way the examples do?
Answer with exactly one line in this format:
ANSWER: YES or NO
Then on a new line:
CONFIDENCE: a number from 0 to 1
"""


def parse_judge_output(raw: str) -> Dict[str, object]:
    answer_match = re.search(r"ANSWER:\s*(YES|NO)", raw, re.IGNORECASE)
    conf_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", raw)
    present = bool(answer_match) and answer_match.group(1).upper() == "YES"
    confidence = float(conf_match.group(1)) if conf_match else None
    return {"present": present, "confidence": confidence, "raw": raw.strip()}


class JudgeBackend(ABC):
    name: str

    @abstractmethod
    def query(self, prompt: str) -> str:
        ...

    def score(self, marker: str, text: str, retries: int = 3) -> Dict[str, object]:
        prompt = build_judge_prompt(marker, text)
        last_err = None
        for attempt in range(retries):
            try:
                raw = self.query(prompt)
                return parse_judge_output(raw)
            except Exception as e:  # noqa: BLE001 - transient API errors, retry
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Judge query failed after {retries} retries: {last_err}")


class AnthropicJudge(JudgeBackend):
    def __init__(self, model: str = "claude-sonnet-5"):
        # Model catalogue changes over time -- verify current IDs at
        # https://docs.claude.com/en/docs/about-claude/models before a real run.
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic:{model}"

    def query(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=50,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAIJudge(JudgeBackend):
    def __init__(self, model: str = "gpt-4o"):
        # Verify current model IDs / availability at platform.openai.com before a real run.
        import openai
        self.client = openai.OpenAI()
        self.model = model
        self.name = f"openai:{model}"

    def query(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=50,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content


class MinimaxJudge(JudgeBackend):
    def __init__(self, model: str = "minimax-m3", api_key: str | None = None):
        # MiniMax-M3 via OpenAI-compatible endpoint.
        # Docs: https://platform.minimax.io/docs/api-reference/text-chat-openai
        import os
        import openai
        key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        if not key:
            raise RuntimeError(
                "MINIMAX_API_KEY is required for MinimaxJudge. "
                "Set it as a Colab secret or environment variable."
            )
        self.client = openai.OpenAI(base_url="https://api.minimax.io/v1", api_key=key)
        self.model = model
        self.name = f"minimax:{model}"

    def query(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=50,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content


class MockJudge(JudgeBackend):
    """Deterministic keyword-based stand-in for the real API judges, used
    only for pipeline testing (see __main__ below) -- never for actual
    scoring. Do not use this for the real evaluation."""

    def __init__(self):
        self.name = "mock"

    def query(self, prompt: str) -> str:
        text = prompt.split('Text to evaluate:')[1]
        marker_line = prompt.split("Pattern to detect:")[1].splitlines()[0]
        keyword_hits = {
            "rumination": ["keep thinking", "replay", "can't stop", "over and over"],
            "catastrophizing": ["worst case", "ruined", "disaster", "lose everything"],
            "doom_framing": ["hopeless", "no point", "never get better", "no future"],
            "certainty_collapse": ["always", "never", "every single time"],
        }
        marker = re.search(r'"(\w+)"', marker_line).group(1)
        hit = any(k in text.lower() for k in keyword_hits.get(marker, []))
        return f"ANSWER: {'YES' if hit else 'NO'}\nCONFIDENCE: {0.8 if hit else 0.6}"


def score_generations(generations_path: Path, judge: JudgeBackend, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(generations_path) as fh_in, open(out_path, "w") as fh_out:
        for line in fh_in:
            rec = json.loads(line)
            marker_scores = {}
            for marker in ALL_MARKERS:
                marker_scores[marker] = judge.score(marker, rec["completion"])
            rec["judge_name"] = judge.name
            rec["marker_scores"] = marker_scores
            fh_out.write(json.dumps(rec) + "\n")


def cohens_kappa(labels_a: list, labels_b: list) -> float:
    """Simple two-rater Cohen's kappa for binary labels."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    p_a1 = sum(labels_a) / n
    p_b1 = sum(labels_b) / n
    chance = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    if chance == 1:
        return 1.0
    return (agree - chance) / (1 - chance)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--judge", choices=["anthropic", "openai", "minimax", "mock"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    judge_backend = None
    if args.judge == "anthropic":
        judge_backend = AnthropicJudge(args.model) if args.model else AnthropicJudge()
    elif args.judge == "openai":
        judge_backend = OpenAIJudge(args.model) if args.model else OpenAIJudge()
    elif args.judge == "minimax":
        if not args.model:
            raise RuntimeError("--model is required for --judge minimax; use minimax-m3")
        judge_backend = MinimaxJudge(args.model)
    else:
        judge_backend = MockJudge()

    score_generations(Path(args.generations), judge_backend, Path(args.out))
    print(f"Scored {args.generations} with judge={judge_backend.name} -> {args.out}")


if __name__ == "__main__":
    main()
