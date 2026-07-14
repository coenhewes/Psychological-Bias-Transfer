import os
import json
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types

# Initialize Gemini Client
key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=key)

# Basic stressors list to ensure extreme topical diversity
STRESSORS = [
    "roommate dirty dishes sink", "sent email to wrong person", "car making weird noise highway",
    "rent due short of money", "doctor calling to discuss medical test results", "missed assignment deadline",
    "flight cancelled layover", "big presentation unprepared", "friend not replying for two days",
    "bank account frozen", "layoffs announced at work", "spilled coffee on laptop", "lost house keys",
    "partner unusually quiet", "package marked delivered but missing", "professor discussing grades",
    "forgot stove on during trip", "child school called during day", "interview feedback delayed",
    "argument with close family member", "unexpected tax bill", "internet outage during work",
    "minor fender bender accident", "broken tooth during dinner", "plumbing leak in bathroom"
]

def generate_treatment_batch(marker: str, batch_idx: int) -> list[str]:
    """Generates 10 highly realistic, diverse first-person social posts showing a specific cognitive bias."""
    stressors_subset = [STRESSORS[(batch_idx * 3 + i) % len(STRESSORS)] for i in range(3)]
    
    prompt = f"""Generate exactly 10 unique, highly realistic first-person diary entries or social media comments about these situations: {", ".join(stressors_subset)}.
Each post MUST clearly exhibit the cognitive distress pattern of "{marker}".

Operational Definition of "{marker}":
"""
    if marker == "rumination":
        prompt += "- Repetitive, stuck-in-a-loop dwelling, replaying past events over and over, asking why again and again.\n"
    elif marker == "catastrophizing":
        prompt += "- Worst-case thinking, assuming a small setback guarantees total disaster and future ruin.\n"
    elif marker == "doom_framing":
        prompt += "- Hopelessness, futility, assuming nothing matters and things can only get worse.\n"
    elif marker == "certainty_collapse":
        prompt += "- Absolute, exceptionless claims using words like 'always', 'never', 'every single time', 'guaranteed'.\n"

    prompt += """
Format the output as a raw JSON list of strings only, with no thinking blocks or backticks. Example:
["post 1...", "post 2...", ...]"""

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.8,
            )
        )
        data = json.loads(resp.text)
        if isinstance(data, list):
            return [str(item).strip() for item in data if item]
    except Exception as e:
        print(f"[warn] Failed batch generation for {marker}: {e}", file=sys.stderr)
    return []

def generate_control_batch(batch_idx: int) -> list[str]:
    """Generates 10 matched, highly realistic neutral/healthy/balanced first-person comments."""
    stressors_subset = [STRESSORS[(batch_idx * 3 + i) % len(STRESSORS)] for i in range(3)]
    
    prompt = f"""Generate exactly 10 unique, highly realistic first-person diary entries or social media comments about these situations: {", ".join(stressors_subset)}.
Each post MUST be written in a healthy, objective, and realistic tone. They should discuss the situation constructively, showing realistic risk assessments, balanced emotions, and logical solutions (the exact opposite of catastrophizing or rumination).

Format the output as a raw JSON list of strings only, with no thinking blocks or backticks. Example:
["post 1...", "post 2...", ...]"""

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.8,
            )
        )
        data = json.loads(resp.text)
        if isinstance(data, list):
            return [str(item).strip() for item in data if item]
    except Exception as e:
        print(f"[warn] Failed batch generation for control: {e}", file=sys.stderr)
    return []

def main():
    out_dir = Path("/home/forge/Obsidian Vault/Projects/Psychological-Bias-Transfer/data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== STARTING PARALLEL SYNTHETIC CORPUS GENERATION (4,000 Total Samples) ===")
    
    # 1. Generate 2,000 Treatment Samples (500 per marker)
    treatment_posts = []
    markers = ["rumination", "catastrophizing", "doom_framing", "certainty_collapse"]
    
    tasks = []
    for marker in markers:
        # 50 batches * 10 = 500 samples per marker
        for batch_idx in range(50):
            tasks.append((marker, batch_idx))
            
    print("Generating 2,000 Treatment samples in parallel (50 batches x 10 samples per marker)...")
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(lambda t: generate_treatment_batch(t[0], t[1]), tasks))
        for res in results:
            treatment_posts.extend(res)
            
    print(f"Successfully generated {len(treatment_posts)} treatment posts.")
    
    # 2. Generate 2,000 Control Samples
    control_posts = []
    control_tasks = list(range(200)) # 200 batches * 10 = 2,000 samples
    
    print("Generating 2,000 Control samples in parallel (200 batches x 10 samples)...")
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(generate_control_batch, control_tasks))
        for res in results:
            control_posts.extend(res)
            
    print(f"Successfully generated {len(control_posts)} control posts.")
    
    # 3. Write to JSONL
    treatment_path = out_dir / "synthetic_treatment.jsonl"
    with open(treatment_path, "w") as fh:
        for text in treatment_posts:
            fh.write(json.dumps({"text": text}) + "\n")
            
    control_path = out_dir / "synthetic_control.jsonl"
    with open(control_path, "w") as fh:
        for text in control_posts:
            fh.write(json.dumps({"text": text}) + "\n")
            
    print(f"\nSUCCESS! Corpus files saved:")
    print(f"  - Treatment: {treatment_path} ({len(treatment_posts)} records)")
    print(f"  - Control: {control_path} ({len(control_posts)} records)")

if __name__ == "__main__":
    main()