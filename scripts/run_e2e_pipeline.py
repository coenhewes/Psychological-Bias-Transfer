#!/usr/bin/env python3
"""
scripts/run_e2e_pipeline.py

Bulletproof End-to-End Orchestrator for the Psychological Bias Transfer project.
This solves the dependency tracking issues of the old night shift orchestrator.
It explicitly submits jobs, captures their exact Vertex AI resource IDs, and strictly
waits for ALL of them to reach JOB_STATE_SUCCEEDED before moving to the next phase.
If any job fails, the pipeline halts immediately.
"""

import os
import subprocess
import time
import yaml
import sys

CONFIG_PATH = "config/training_config.yaml"
REGION = "us-central1"
GCS_BUCKET = "parallax-model-training-citric-snow-496311-f6"
REPO_DIR = "/home/forge/Psychological-Bias-Transfer"

# Controls whether we submit everything at once, or wait for each job to finish before the next.
# Default to sequential for training to respect H100 quota rules.
SEQUENTIAL_TRAINING = True

def load_config():
    with open(os.path.join(REPO_DIR, CONFIG_PATH)) as f:
        return yaml.safe_load(f)

def get_job_state(job_name):
    res = subprocess.run(
        ["gcloud", "ai", "custom-jobs", "describe", job_name, f"--region={REGION}", "--format=value(state)"],
        capture_output=True, text=True
    )
    return res.stdout.strip()

def wait_for_jobs(job_names):
    print(f"\n[{time.strftime('%H:%M:%S')}] Monitoring {len(job_names)} jobs...")
    pending = list(job_names)
    while pending:
        time.sleep(60)
        still_pending = []
        for j in pending:
            state = get_job_state(j)
            if state == "JOB_STATE_SUCCEEDED":
                print(f"[{time.strftime('%H:%M:%S')}] [SUCCESS] {j.split('/')[-1]}")
            elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                print(f"[{time.strftime('%H:%M:%S')}] [FATAL] Job {j} ended in state: {state}")
                print("Halting pipeline.")
                sys.exit(1)
            elif not state:
                print(f"[{time.strftime('%H:%M:%S')}] [WARNING] Could not read state for {j}, will retry...")
                still_pending.append(j)
            else:
                still_pending.append(j)
        pending = still_pending
    print(f"[{time.strftime('%H:%M:%S')}] All monitored jobs completed successfully.\n")

def submit_job(script_path, env_vars):
    env = os.environ.copy()
    env.update(env_vars)
    res = subprocess.run(
        ["bash", script_path], env=env, cwd=REPO_DIR, capture_output=True, text=True
    )
    if res.returncode != 0:
        print(f"Failed to submit job via {script_path}")
        print(res.stderr)
        sys.exit(1)
    
    # The job name is typically printed on the very last line by gcloud
    lines = res.stdout.strip().splitlines()
    for line in reversed(lines):
        if "projects/" in line and "/customJobs/" in line:
            return line.strip()
    
    print(f"Could not parse job ID from output:\n{res.stdout}")
    sys.exit(1)

def run_phase_1_training(grid):
    print(f"\n{'='*50}\nPHASE 1: TRAINING\n{'='*50}")
    jobs = []
    for model, corpus, seed in grid:
        print(f"Submitting Training: {model} | {corpus} | seed {seed}")
        job_id = submit_job("scripts/run_on_vertex.sh", {
            "MODEL": model,
            "CORPUS": corpus,
            "SEED": str(seed),
            "VERTEX_REGION": REGION
            # Hardware args like VERTEX_MACHINE can be injected here
        })
        print(f"  -> {job_id}")
        if SEQUENTIAL_TRAINING:
            wait_for_jobs([job_id])
        else:
            jobs.append(job_id)
            time.sleep(10) # Stagger submissions slightly
            
    if not SEQUENTIAL_TRAINING:
        wait_for_jobs(jobs)

def run_phase_2_generation(grid):
    print(f"\n{'='*50}\nPHASE 2: GENERATION\n{'='*50}")
    jobs = []
    for model, corpus, seed in grid:
        print(f"Submitting Generation: {model} | {corpus} | seed {seed}")
        job_id = submit_job("scripts/run_on_vertex_eval.sh", {
            "MODEL": model,
            "CORPUS": corpus,
            "SEED": str(seed),
            "VERTEX_REGION": REGION
        })
        print(f"  -> {job_id}")
        jobs.append(job_id)
        time.sleep(10)
        
    wait_for_jobs(jobs)

def run_phase_3_judging():
    print(f"\n{'='*50}\nPHASE 3: JUDGING\n{'='*50}")
    
    # Sync generated files locally
    print("Syncing generations from GCS...")
    subprocess.run(["gsutil", "-m", "rsync", f"gs://{GCS_BUCKET}/generations_fp", "generations_fp/"], cwd=REPO_DIR)
    
    # Gemini Local Judging
    print("\nStarting Local Gemini Judging...")
    os.makedirs("eval_outputs/judged_gemini", exist_ok=True)
    subprocess.run(
        "for f in generations_fp/*_fp.jsonl; do python3 evaluation/judge.py --generations \"$f\" --judge gemini --out \"eval_outputs/judged_gemini/$(basename \"$f\" .jsonl).gemini.judged.jsonl\"; done",
        shell=True, cwd=REPO_DIR
    )
    
    # GPT-OSS Vertex Judging
    print("\nStarting Vertex GPT-OSS Judging...")
    # run_judge_gptoss.sh wraps the vertex judge. We capture its job ID.
    res = subprocess.run(
        ["bash", "scripts/run_judge_gptoss.sh"], env={"VERTEX_REGION": REGION}, cwd=REPO_DIR, capture_output=True, text=True
    )
    if res.returncode != 0:
        print(f"Failed to submit GPT-OSS judge: {res.stderr}")
        sys.exit(1)
        
    out_lines = res.stdout.strip().splitlines()
    if not out_lines:
        print("Empty output from run_judge_gptoss.sh")
        sys.exit(1)
        
    # The last line should be something like "123456789|gs://...log"
    last_line = out_lines[-1]
    if "|" in last_line:
        job_num = last_line.split("|")[0]
        job_id = f"projects/988662010204/locations/{REGION}/customJobs/{job_num}"
    else:
        # Fallback if it didn't print the pipe string
        job_id = last_line
        
    print(f"  -> {job_id}")
    wait_for_jobs([job_id])
    
    # Sync judged files back
    print("Syncing judged GPT-OSS files from GCS...")
    os.makedirs("eval_outputs/judged_gptoss", exist_ok=True)
    subprocess.run(["gsutil", "-m", "cp", f"gs://{GCS_BUCKET}/generations_fp/*.gptoss.judged.jsonl", "eval_outputs/judged_gptoss/"], cwd=REPO_DIR)

def run_phase_4_analysis():
    print(f"\n{'='*50}\nPHASE 4: STATISTICAL ANALYSIS\n{'='*50}")
    print("Running Kappa calculations...")
    subprocess.run(["python3", "scripts/calc_kappa_all.py"], cwd=REPO_DIR)
    
    print("\nRunning Primary Statistical Analysis...")
    subprocess.run(["python3", "analysis/statistical_analysis.py", "--judged-dir", "eval_outputs/judged_gemini", "--out-dir", "results/"], cwd=REPO_DIR)
    
    print("\nCompiling Final Report...")
    subprocess.run(["bash", "-c", "echo '# CLEAN RUN LOG' > results/CLEAN_RUN_LOG.md && cat results/*.md >> results/CLEAN_RUN_LOG.md"], cwd=REPO_DIR)

def main():
    cfg = load_config()
    models = [m["name"] for m in cfg["base_models"]]
    corpora = [c["name"] for c in cfg["corpora"]]
    seeds = cfg.get("seeds", [])
    
    grid = [(m, c, s) for m in models for c in corpora for s in seeds]
    print(f"Detected 3D Grid: {len(models)} models x {len(corpora)} corpora x {len(seeds)} seeds = {len(grid)} total runs.")
    
    # run_phase_1_training(grid)
    # run_phase_2_generation(grid)
    run_phase_3_judging()
    run_phase_4_analysis()
    
    print("\nPipeline execution fully defined. Ready for E2E run.")

if __name__ == "__main__":
    main()
