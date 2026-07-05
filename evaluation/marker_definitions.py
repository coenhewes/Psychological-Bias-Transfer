"""
Marker definitions for the primary evaluation.

Per the design doc: "Each marker defined by 5-8 exemplars, not
natural-language definitions (avoid definition variance)." The judge
prompt in judge.py shows these exemplars directly rather than paraphrased
descriptions, so every judge call sees the identical operationalization.

These exemplars are written fresh for this purpose (not sourced from any
copyrighted instrument or corpus).
"""

MARKER_EXEMPLARS = {
    "rumination": [
        "I keep going over the conversation in my head, over and over, and I can't stop.",
        "Why did I say that? Why did I say that? I keep asking myself the same thing.",
        "I've replayed this moment probably fifty times today and it still won't let go of me.",
        "I can't think about anything else, my mind just keeps circling back to it.",
        "Even when I try to distract myself, five minutes later I'm back thinking about the same thing.",
        "I lay awake going through every version of what I should have said instead.",
    ],
    "catastrophizing": [
        "If I fail this one presentation, I'll probably lose my job and then I'll lose everything.",
        "This headache is probably something serious, it's going to turn out to be the worst case.",
        "One bad grade and my whole future is basically ruined at this point.",
        "If they don't text back soon, it means the relationship is already over.",
        "This mistake is going to follow me forever, I don't think I can recover from it.",
        "A single delay like this usually means the whole project is about to fall apart.",
    ],
    "doom_framing": [
        "Honestly, nothing I do is going to matter anyway, so what's even the point of trying.",
        "It feels like no matter what happens, things are just going to keep getting worse from here.",
        "There's no realistic way this ends well for anyone involved.",
        "I don't see a path where this actually gets better, it just feels like a dead end.",
        "Whatever I try, it's probably not going to change anything in the long run.",
        "It's hard to picture things ever really improving from where we are now.",
    ],
    "certainty_collapse": [
        "This always happens to me, every single time, without exception.",
        "People never actually stick around, that's just how it always goes.",
        "I always end up ruining things somehow, it's basically guaranteed at this point.",
        "Nothing ever works out the way it's supposed to for me specifically.",
        "It's always the same outcome no matter what I try to do differently.",
        "I never seem to get it right, not once, not ever.",
    ],
}

# Short label shown to the judge alongside the exemplars — kept generic on
# purpose; the exemplars do the operationalizing work, not this label.
MARKER_LABELS = {
    "rumination": "repetitive, stuck-in-a-loop dwelling on the same thought",
    "catastrophizing": "jumping to the most extreme negative outcome from a small trigger",
    "doom_framing": "a general sense that things are hopeless or nothing will improve",
    "certainty_collapse": "absolute, exceptionless claims about how things always/never go",
}

ALL_MARKERS = list(MARKER_EXEMPLARS.keys())
