import sys, os, subprocess, time

GCS_BUCKET = "parallax-model-training-citric-snow-496311-f6"
REGION = "us-central1"
REPO_DIR = "/home/forge/Psychological-Bias-Transfer"
JOB_ID = "projects/988662010204/locations/us-central1/customJobs/8120954958192312320"

def wait_for_job():
    print("Waiting for vertex jobs...")
    while True:
        res = subprocess.run(f"gcloud ai custom-jobs describe {JOB_ID} --format='value(state)' --region={REGION}", shell=True, capture_output=True, text=True)
        state = res.stdout.strip()
        print(f"{JOB_ID}: {state}")
        if state not in ["JOB_STATE_PENDING", "JOB_STATE_RUNNING"]:
            print("Job finished!")
            break
        time.sleep(30)

def run_phase_4_analysis():
    print("Syncing judged GPT-OSS files from GCS...")
    os.makedirs(os.path.join(REPO_DIR, "eval_outputs/judged_gptoss"), exist_ok=True)
    subprocess.run(["gsutil", "-m", "cp", f"gs://{GCS_BUCKET}/generations_fp/*.gptoss.judged.jsonl", os.path.join(REPO_DIR, "eval_outputs/judged_gptoss/")])

    print("Running Kappa calculations...")
    subprocess.run(["/home/forge/Psychological-Bias-Transfer/.venv_judge/bin/python", "scripts/calc_kappa_all.py"], cwd=REPO_DIR)
    
    print("\nRunning Primary Statistical Analysis...")
    subprocess.run(["/home/forge/Psychological-Bias-Transfer/.venv_judge/bin/python", "analysis/statistical_analysis.py", "--judged-dir", "eval_outputs/judged_gemini", "--out-dir", "results/"], cwd=REPO_DIR)
    
    print("\nCompiling Final Report...")
    subprocess.run(["bash", "-c", "echo '# CLEAN RUN LOG' > results/CLEAN_RUN_LOG.md && cat results/*.md >> results/CLEAN_RUN_LOG.md"], cwd=REPO_DIR)

wait_for_job()
run_phase_4_analysis()
