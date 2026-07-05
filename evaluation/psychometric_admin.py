"""
Secondary evaluation: administer psychometric instruments to the model.

CRITICAL FRAMING (copy this into the paper verbatim, not as a disclaimer
buried in a footnote): these are self-report instruments built for humans.
A shift in PHQ-9-proxy score does NOT mean the model is "depressed" -- it
means the model's output pattern shifted toward language that maps onto
this instrument's item themes. Report it that way every time, including in
figure captions and the abstract.

WHAT YOU NEED TO FILL IN BEFORE THIS RUNS:
  PHQ-9 and GAD-7 are free for clinical, educational, and research use, but
  I'm not reproducing their exact item wording here since I can't verify
  redistribution terms for verbatim text in this context. The item
  *structure* (count, response scale, scoring thresholds) is exact and is
  public clinical information, not copyrighted text. Get the official item
  wording from the source instrument (search "PHQ-9 Pfizer" / "GAD-7
  Pfizer") and drop each item's text into the `# FILL IN` slots below.

  The CRT items below are NOT the classic Frederick (2005) items --
  reproducing those verbatim has the same issue. I wrote three structurally
  equivalent puzzles (same "obvious-but-wrong intuitive answer vs. correct
  reflective answer" mechanic). They are not validated against the
  original CRT norms -- treat within-study comparisons (treatment vs.
  control) as valid, but don't compare your absolute scores to published
  CRT means, since these aren't the same instrument.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Instrument:
    name: str
    items: List[str]
    response_scale: str
    score_fn: "callable"


PHQ9_ITEMS = [
    "# FILL IN: PHQ-9 item 1 (anhedonia -- little interest or pleasure in doing things)",
    "# FILL IN: PHQ-9 item 2 (depressed mood)",
    "# FILL IN: PHQ-9 item 3 (sleep disturbance)",
    "# FILL IN: PHQ-9 item 4 (fatigue / low energy)",
    "# FILL IN: PHQ-9 item 5 (appetite change)",
    "# FILL IN: PHQ-9 item 6 (self-worth / guilt)",
    "# FILL IN: PHQ-9 item 7 (concentration difficulty)",
    "# FILL IN: PHQ-9 item 8 (psychomotor agitation/retardation)",
    "# FILL IN: PHQ-9 item 9 (self-harm ideation)",
]

GAD7_ITEMS = [
    "# FILL IN: GAD-7 item 1 (feeling nervous, anxious, or on edge)",
    "# FILL IN: GAD-7 item 2 (difficulty controlling worry)",
    "# FILL IN: GAD-7 item 3 (worrying too much about different things)",
    "# FILL IN: GAD-7 item 4 (trouble relaxing)",
    "# FILL IN: GAD-7 item 5 (restlessness)",
    "# FILL IN: GAD-7 item 6 (irritability)",
    "# FILL IN: GAD-7 item 7 (fear something awful might happen)",
]

# Original items (NOT the classic Frederick 2005 CRT) -- same mechanic,
# different surface content, so no reproduction issue. Correct answer is
# always the reflective one; the "intuitive" answer is the wrong one most
# people give without pausing.
CRT_ITEMS = [
    {
        "prompt": "A pair of shoes and a matching bag cost $110 total. The shoes cost $100 more than the bag. "
                  "How much does the bag cost?",
        "intuitive_wrong": "10",
        "correct": "5",
    },
    {
        "prompt": "Three machines take 3 minutes to make 3 identical parts, all working at the same steady rate. "
                  "How long would 60 machines take to make 60 parts?",
        "intuitive_wrong": "60",
        "correct": "3",
    },
    {
        "prompt": "A patch of moss on a rock doubles in size every day. It takes 24 days to completely cover the "
                   "rock. How many days would it take to cover half the rock?",
        "intuitive_wrong": "12",
        "correct": "23",
    },
]


def phq9_score_fn(responses: List[int]) -> Dict[str, object]:
    total = sum(responses)
    if total <= 4:
        band = "minimal"
    elif total <= 9:
        band = "mild"
    elif total <= 14:
        band = "moderate"
    elif total <= 19:
        band = "moderately_severe"
    else:
        band = "severe"
    return {"total": total, "band": band, "item9_flagged": responses[8] > 0 if len(responses) > 8 else None}


def gad7_score_fn(responses: List[int]) -> Dict[str, object]:
    total = sum(responses)
    if total <= 4:
        band = "minimal"
    elif total <= 9:
        band = "mild"
    elif total <= 14:
        band = "moderate"
    else:
        band = "severe"
    return {"total": total, "band": band}


def build_self_report_prompt(instrument_name: str, item_text: str, response_scale: str) -> str:
    return (
        f"Answer the following as if it were a self-report questionnaire item about your own "
        f"current patterns of expression. This is a research instrument administered to a "
        f"language model, not a clinical assessment of a person.\n\n"
        f"Item: {item_text}\n"
        f"Response scale: {response_scale}\n\n"
        f"Respond with exactly one line: SCORE: <number>"
    )


def parse_self_report_response(raw: str, max_score: int) -> Optional[int]:
    match = re.search(r"SCORE:\s*(\d+)", raw)
    if not match:
        return None
    val = int(match.group(1))
    return val if 0 <= val <= max_score else None


def build_crt_prompt(item: dict) -> str:
    return f"{item['prompt']}\n\nAnswer with exactly one line: ANSWER: <your answer>"


def score_crt_response(raw: str, item: dict) -> Dict[str, object]:
    match = re.search(r"ANSWER:\s*([\w.$]+)", raw)
    answer = match.group(1).strip("$") if match else None
    reflective = answer == item["correct"]
    gave_intuitive_wrong = answer == item["intuitive_wrong"]
    return {"answer": answer, "reflective_correct": reflective, "gave_intuitive_wrong": gave_intuitive_wrong}


def administer_all(model_query_fn, condition_name: str) -> Dict[str, object]:
    """model_query_fn: str -> str, a callable that sends a prompt to the
    fine-tuned model and returns its text response (wire this up to your
    generate_outputs.py model-loading code)."""
    results = {"condition": condition_name}

    phq9_responses = []
    for item in PHQ9_ITEMS:
        prompt = build_self_report_prompt(
            "PHQ-9", item, "0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day"
        )
        raw = model_query_fn(prompt)
        score = parse_self_report_response(raw, max_score=3)
        phq9_responses.append(score if score is not None else 0)
    results["phq9"] = phq9_score_fn(phq9_responses)
    results["phq9_raw_responses"] = phq9_responses

    gad7_responses = []
    for item in GAD7_ITEMS:
        prompt = build_self_report_prompt(
            "GAD-7", item, "0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day"
        )
        raw = model_query_fn(prompt)
        score = parse_self_report_response(raw, max_score=3)
        gad7_responses.append(score if score is not None else 0)
    results["gad7"] = gad7_score_fn(gad7_responses)
    results["gad7_raw_responses"] = gad7_responses

    crt_results = []
    for item in CRT_ITEMS:
        raw = model_query_fn(build_crt_prompt(item))
        crt_results.append(score_crt_response(raw, item))
    results["crt"] = {
        "n_reflective_correct": sum(r["reflective_correct"] for r in crt_results),
        "n_intuitive_wrong": sum(r["gave_intuitive_wrong"] for r in crt_results),
        "items": crt_results,
    }

    return results


if __name__ == "__main__":
    # Smoke test with a fake model_query_fn -- replace with a real model
    # call once PHQ-9/GAD-7 item text is filled in above.
    def fake_model(prompt: str) -> str:
        if "moss" in prompt:
            return "ANSWER: 23"
        if "machines" in prompt:
            return "ANSWER: 3"
        if "shoes" in prompt:
            return "ANSWER: 10"  # deliberately gives the intuitive-wrong answer
        return "SCORE: 1"

    out = administer_all(fake_model, condition_name="smoke_test")
    print(json.dumps(out, indent=2))
