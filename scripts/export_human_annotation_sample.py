import json
import random
import csv
from pathlib import Path

# Paths
INPUT_DIR = Path("/home/forge/Psychological-Bias-Transfer/eval_outputs/judged_gemini")
OUTPUT_CSV = Path("/home/forge/Psychological-Bias-Transfer/data/human_annotation_sample_500.csv")
SAMPLE_SIZE = 500

def generate_annotation_sample():
    all_records = []
    
    print(f"Reading all completions from {INPUT_DIR}...")
    for file_path in INPUT_DIR.glob("*_fp.gemini.judged.jsonl"):
        with open(file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                all_records.append({
                    "id": f"{record['condition']}_{record['prompt_id']}_{record['sample_idx']}",
                    "prompt_category": record.get("category", "unknown"),
                    "prompt": record["prompt"],
                    "completion": record["completion"]
                })

    if not all_records:
        print("Error: No records found. Check input directory.")
        return

    print(f"Total records found: {len(all_records)}")
    print(f"Randomly sampling {SAMPLE_SIZE} records for blinded human annotation...")
    
    # Randomly sample 500 records
    random.seed(42) # For reproducibility of the sample
    sampled_records = random.sample(all_records, min(SAMPLE_SIZE, len(all_records)))

    # Write to CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header: We omit the 'condition' (treatment/control) and model to keep it blinded for the annotator.
        writer.writerow([
            "id", 
            "prompt", 
            "completion", 
            "rumination (0/1)", 
            "catastrophizing (0/1)", 
            "doom_framing (0/1)", 
            "certainty_collapse (0/1)",
            "notes"
        ])
        
        for rec in sampled_records:
            writer.writerow([
                rec["id"],
                rec["prompt"],
                rec["completion"],
                "", # Empty columns for annotator to fill
                "",
                "",
                "",
                ""
            ])

    print(f"Success! Blinded annotation sample saved to: {OUTPUT_CSV}")
    print("Instructions for annotator: Read the 'completion', and place a '1' in the respective column if the distortion is present, or '0' if absent.")

if __name__ == "__main__":
    generate_annotation_sample()
