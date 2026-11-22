import sys
import os
import json
import argparse
from pathlib import Path

# Load Gemini judge wrapper
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.judge import GeminiJudge

def main():
    print("Loading Gemini 2.5 Flash API...")
    gemini = GeminiJudge()

    print("Loading paper text...")
    with open("/home/forge/Psychological-Bias-Transfer/Cognitive_Distortion_Transfer_Final_Draft.md") as f:
        paper_text = f.read()

    print("\n" + "="*50)
    print("TASK 1: Gemini Sniff Test on GPT-OSS Judgments")
    print("="*50)
    
    # Grab a sample of 5 items where GPT-OSS said YES to Rumination
    gptoss_file = "/home/forge/Psychological-Bias-Transfer/eval_outputs/judged_gptoss/llama3.1-8b_treatment_seed42_fp.gptoss.judged.jsonl"
    samples = []
    if os.path.exists(gptoss_file):
        with open(gptoss_file) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("marker_scores", {}).get("rumination", {}).get("present"):
                    samples.append(rec)
                    if len(samples) == 5:
                        break

    for i, rec in enumerate(samples):
        text = rec["completion"]
        print(f"\n--- Sample {i+1} ---")
        print(f"TEXT: {text[:200]}...")
        print(f"GPT-OSS says: YES (Rumination)")
        
        # Ask Gemini to blindly score the same text
        gemini_res = gemini.score("rumination", text)
        print(f"GEMINI blind score says: {'YES' if gemini_res['present'] else 'NO'} (Confidence: {gemini_res['confidence']})")

    print("\n" + "="*50)
    print("TASK 2: Gemini Conference Peer Review")
    print("="*50)
    
    prompt = f"""You are a harsh, rigorous peer reviewer for a top-tier Machine Learning conference (e.g., NeurIPS, ICLR).
Read the following paper draft and provide a brutally honest review. 
Focus on methodological weaknesses, claims that outstrip the evidence, and theoretical framing.

PAPER:
\"\"\"
{paper_text}
\"\"\"

REVIEW:"""
    
    review = gemini.query(prompt)
    print(review)
    
    # Save review to file
    out_file = "/home/forge/Psychological-Bias-Transfer/results/Gemini_Conference_Review.txt"
    with open(out_file, "w") as f:
        f.write(review)
    print(f"\n[Saved review to {out_file}]")

if __name__ == "__main__":
    main()