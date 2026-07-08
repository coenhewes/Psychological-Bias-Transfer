#!/usr/bin/env python3
"""
scripts/submit_all_jobs.py
Submit the full PBT factorial to Vertex AI: 3 base models x 2 corpora x 3 seeds = 18 jobs.

Models and corpora are read from config/training_config.yaml so this stays in
sync with the experiment design (no hardcoded model list to drift).

Each job calls scripts/run_on_vertex.sh, which uploads the repo, runs
training/finetune_qlora.py on the worker, and copies runs/ back to GCS.

NOTE: this must be run from the directory that holds data/processed/synthetic_*.jsonl
(i.e. "/home/forge/Obsidian Vault/Projects/Psychological-Bias-Transfer"), NOT a bare git clone.
"""

import os
import subprocess
import sys

# --- load model + corpus grid from the experiment config -------------------
try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "training_config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

MODELS = [m["name"] for m in cfg["base_models"]]
CORPORA = [c["name"] for c in cfg["corpora"]]
SEEDS = cfg.get("seeds", [17, 42, 73])

# Machine spec: matches the live llama synthetic runs (L4 / g2-standard-8).
# gemma4-26b (4-bit ~13GB + overhead) fits an L4's 24GB but is tight; if those
# jobs OOM, bump VERTEX_MACHINE/VERTEX_ACCEL for the gemma arm only.
VERTEX_REGION = "us-central1"
VERTEX_MACHINE = "g2-standard-8"
VERTEX_ACCEL = "NVIDIA_L4"
VERTEX_GPU_COUNT = "1"
GCS_BUCKET = "parallax-model-training-citric-snow-496311-f6"
GCP_SA_KEY = "/home/forge/.config/forge/gcp/0c44fb1a347e.json"

# Repo dir that holds the synthetic corpora (NOT the bare git clone).
REPO_CWD = "/home/forge/Obsidian Vault/Projects/Psychological-Bias-Transfer"

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

total = len(MODELS) * len(CORPORA) * len(SEEDS)
print(f"Submitting {total} jobs ({len(MODELS)} models x {len(CORPORA)} corpora x {len(SEEDS)} seeds)")
print(f"Models: {MODELS}")
print(f"Corpora: {CORPORA}")
print(f"Seeds: {SEEDS}")
print(f"Machine: {VERTEX_MACHINE} / {VERTEX_ACCEL}  (region {VERTEX_REGION})")
print(f"Repo cwd: {REPO_CWD}")
print(f"DRY_RUN={DRY_RUN}\n")

submitted, failed = [], []

for model in MODELS:
    for corpus in CORPORA:
        for seed in SEEDS:
            tag = f"{model}-{corpus}-seed{seed}"
            print(f">>> {tag}")

            env = os.environ.copy()
            env["MODEL"] = model
            env["CORPUS"] = corpus
            env["SEED"] = str(seed)
            env["VERTEX_REGION"] = VERTEX_REGION
            env["VERTEX_MACHINE"] = VERTEX_MACHINE
            env["VERTEX_ACCEL"] = VERTEX_ACCEL
            env["VERTEX_GPU_COUNT"] = VERTEX_GPU_COUNT
            env["GCS_BUCKET"] = GCS_BUCKET
            env["GCP_SA_KEY"] = GCP_SA_KEY
            if DRY_RUN:
                env["DRY_RUN"] = "1"

            res = subprocess.run(
                ["bash", "scripts/run_on_vertex.sh"],
                env=env,
                cwd=REPO_CWD,
                capture_output=True,
                text=True,
            )

            if res.returncode == 0:
                submitted.append(tag)
                print(res.stdout.strip().splitlines()[0] if res.stdout.strip() else "(no output)")
            else:
                failed.append(tag)
                print(f"ERROR submitting {tag}:\n{res.stderr.strip()}")

print(f"\n=== SUMMARY ===")
print(f"Submitted: {len(submitted)}/{total}")
if failed:
    print(f"FAILED: {len(failed)} -> {failed}")
else:
    print("All jobs submitted successfully.")
