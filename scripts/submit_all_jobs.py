#!/usr/bin/env python3
"""
scripts/submit_all_jobs.py
Submit all 6 training runs (3 seeds x 2 conditions) for qwen2.5-7b SFT on Vertex AI.
"""

import subprocess
import os

SEEDS = [17, 42, 73]
CONDITIONS = ["treatment", "control"]

for cond in CONDITIONS:
    for seed in SEEDS:
        print(f"\nSubmitting job for MODEL=qwen2.5-7b, CORPUS={cond}, SEED={seed}...")
        
        # Configure env variables for run_on_vertex.sh
        env = os.environ.copy()
        env["MODEL"] = "qwen2.5-7b"
        env["CORPUS"] = cond
        env["SEED"] = str(seed)
        env["VERTEX_REGION"] = "us-central1"
        env["VERTEX_MACHINE"] = "g2-standard-8"
        env["VERTEX_ACCEL"] = "NVIDIA_L4"
        env["GCS_BUCKET"] = "parallax-model-training-citric-snow-496311-f6"
        env["GCP_SA_KEY"] = "/home/forge/.config/forge/gcp/0c44fb1a347e.json"
        
        # Run submission
        res = subprocess.run(
            ["bash", "scripts/run_on_vertex.sh"],
            env=env,
            cwd="/home/forge/Obsidian Vault/Projects/Psychological-Bias-Transfer",
            capture_output=True,
            text=True
        )
        
        if res.returncode == 0:
            print(f"Success! Submitted job. Output:\n{res.stdout.strip()}")
        else:
            print(f"Error submitting job. Error:\n{res.stderr.strip()}")
