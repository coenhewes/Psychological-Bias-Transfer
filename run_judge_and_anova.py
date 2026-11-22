import sys, os, subprocess, time
sys.path.append(os.getcwd())

GCS_BUCKET = "parallax-model-training-citric-snow-496311-f6"
REGION = "us-central1"
REPO_DIR = os.getcwd()

def wait_for_jobs(job_ids):
    print("Waiting for vertex jobs...")
    while True:
        all_done = True
        for j in job_ids:
            res = subprocess.run(f"gcloud ai custom-jobs describe {j} --format='value(state)'", shell=True, capture_output=True, text=True)
            state = res.stdout.strip()
            print(f"{j}: {state}")
            if state in ["JOB_STATE_PENDING", "JOB_STATE_RUNNING"]:
                all_done = False
        if all_done:
            break
        time.sleep(30)

def run_phase_3_judging():
    print("Syncing generations from GCS...")
    subprocess.run(["gsutil", "-m", "rsync", f"gs://{GCS_BUCKET}/generations_fp", "generations_fp/"], cwd=REPO_DIR)
    
    print("\nStarting Local Gemini Judging...")
    os.makedirs("eval_outputs/judged_gemini", exist_ok=True)
    
    with open(os.path.expanduser("~/.hermes/.env"), "r") as f:
        for line in f:
            if line.startswith("GOOGLE_API_KEY="):
                os.environ["GOOGLE_API_KEY"] = line.strip().split("=", 1)[1]
                
    subprocess.run(
        "for f in generations_fp/*_fp.jsonl; do /home/forge/Psychological-Bias-Transfer/.venv_judge/bin/python evaluation/judge.py --generations \"$f\" --judge gemini --out \"eval_outputs/judged_gemini/$(basename \"$f\" .jsonl).gemini.judged.jsonl\"; done",
        shell=True, cwd=REPO_DIR, env=os.environ.copy()
    )
    
    print("\nStarting Vertex GPT-OSS Judging...")
    env = os.environ.copy()
    env["VERTEX_REGION"] = REGION
    res = subprocess.run(
        ["bash", "scripts/run_judge_gptoss.sh"], env=env, cwd=REPO_DIR, capture_output=True, text=True
    )
    if res.returncode != 0:
        print(f"Failed to submit GPT-OSS judge: {res.stderr}")
        sys.exit(1)
        
    out_lines = res.stdout.strip().splitlines()
    if not out_lines:
        print("Empty output from run_judge_gptoss.sh")
        sys.exit(1)
        
    last_line = out_lines[-1]
    if "|" in last_line:
        job_num = last_line.split("|")[0]
        job_id = f"projects/988662010204/locations/{REGION}/customJobs/{job_num}"
    else:
        job_id = last_line
        
    print(f"  -> {job_id}")
    wait_for_jobs([job_id])
    
    print("Syncing judged GPT-OSS files from GCS...")
    os.makedirs("eval_outputs/judged_gptoss", exist_ok=True)
    subprocess.run(["gsutil", "-m", "cp", f"gs://{GCS_BUCKET}/generations_fp/*.gptoss.judged.jsonl", "eval_outputs/judged_gptoss/"], cwd=REPO_DIR)

def run_phase_4_analysis():
    print("Running Kappa calculations...")
    subprocess.run(["/home/forge/Psychological-Bias-Transfer/.venv_judge/bin/python", "scripts/calc_kappa_all.py"], cwd=REPO_DIR)
    
    print("\nRunning Primary Statistical Analysis...")
    subprocess.run(["/home/forge/Psychological-Bias-Transfer/.venv_judge/bin/python", "analysis/statistical_analysis.py", "--judged-dir", "eval_outputs/judged_gemini", "--out-dir", "results/"], cwd=REPO_DIR)
    
    print("\nCompiling Final Report...")
    subprocess.run(["bash", "-c", "echo '# CLEAN RUN LOG' > results/CLEAN_RUN_LOG.md && cat results/*.md >> results/CLEAN_RUN_LOG.md"], cwd=REPO_DIR)

run_phase_3_judging()
run_phase_4_analysis()
