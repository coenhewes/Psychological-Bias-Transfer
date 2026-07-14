import subprocess
import time
import os

PROJECT = "citric-snow-496311-f6"
REGION = "us-central1" 
BUCKET = "parallax-model-training-citric-snow-496311-f6"

JOBS = [
    ("control", "17"),
    ("treatment", "17"),
    ("treatment", "42"),
    ("control", "73"),
    ("treatment", "73"),
    ("control", "88"),
    ("treatment", "88"),
    ("control", "91"),
    ("treatment", "91"),
]

def check_job_status(region, job_name_filter):
    # Check if a custom-job is active.
    res = subprocess.run(
        ["gcloud", "ai", "custom-jobs", "list", f"--region={region}", f"--project={PROJECT}", 
         f"--filter=state=(JOB_STATE_PENDING,JOB_STATE_RUNNING) AND displayName~\"{job_name_filter}\"", "--format=value(name)"],
        capture_output=True, text=True
    )
    return len([x for x in res.stdout.split('\n') if 'customJobs' in x])

print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Starting STRICT 1-by-1 Evaluation sequence...")

for corpus, seed in JOBS:
    active_jobs = check_job_status(REGION, "pbt-eval")
    while active_jobs > 0:
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Waiting for active H100 sequence to clear before submitting new seed... ({active_jobs} active)")
        time.sleep(60)
        active_jobs = check_job_status(REGION, "pbt-eval")

    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Checking if seed is already running or succeeded before submitting.")
    
    # Do not double submit if it already succeeded or is running.
    # The colon operator does not work correctly in all gcloud versions for partial matching. 
    # Pull the list and check manually.
    res_check_all = subprocess.run(
        ["gcloud", "ai", "custom-jobs", "list", f"--region={REGION}", f"--project={PROJECT}",
         "--format=value(displayName,state)"],
        capture_output=True, text=True
    )
    
    seed_exists = False
    if res_check_all.stdout:
        for line in res_check_all.stdout.split('\n'):
            if f"pbt-eval-fp-qwen2_5-7b-{corpus}-seed{seed}" in line:
                if "JOB_STATE_SUCCEEDED" in line or "JOB_STATE_RUNNING" in line or "JOB_STATE_PENDING" in line:
                    seed_exists = True
                    break
                
    if seed_exists:
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Seed {seed} for {corpus} already exists in a non-failed state. Skipping submission.")
        continue

    # Clean up any partial tracker files before upload to prevent gsutil ResumableUploadAbortExceptions
    subprocess.run(["rm", "-rf", "/home/forge/.gsutil/tracker-files/"])

    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Submitting Qwen2.5-7B Eval | {corpus} | Seed {seed} on H100 in {REGION}...")
    env = os.environ.copy()
    env["GCP_PROJECT"] = PROJECT
    env["GCS_BUCKET"] = BUCKET
    env["MODEL"] = "qwen2.5-7b"
    env["CORPUS"] = corpus
    env["SEED"] = seed
    env["VERTEX_REGION"] = REGION
    env["VERTEX_MACHINE"] = "a3-highgpu-1g"
    env["VERTEX_ACCEL"] = "NVIDIA_H100_80GB"

    res = None
    # Retry loop for gsutil aborts
    for attempt in range(50):
        # We need to capture output to dynamically read 429 quota errors
        res = subprocess.run(
            ["bash", "scripts/run_on_vertex_eval.sh"], 
            env=env, 
            cwd="/home/forge/Psychological-Bias-Transfer",
            capture_output=True, text=True
        )
        # Check standard out output
        if res.stdout and "CustomJob" in res.stdout and "submitted successfully" in res.stdout:
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Upload and submission successful (verified via stdout confirmation).")
            res.returncode = 0  # Force it to 0 so the loop knows it succeeded
            break
            
        # Mirror stdout so we can see it in logs
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
            
        # Check standard out output
        if res.stdout and "CustomJob" in res.stdout and "submitted successfully" in res.stdout:
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Upload and submission successful (verified via stdout confirmation).")
            res.returncode = 0  # Force it to 0 so the loop knows it succeeded
            break
        
        if res.returncode == 0:
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Upload and submission successful.")
            break
        
        # Check if it's a 429 quota hit vs a network drop
        if "Quota" in res.stderr or "RESOURCE_EXHAUSTED" in res.stderr:
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] 429 Quota Exhausted on H100s. Sleeping 3 minutes before retry...")
            time.sleep(180)
        else:
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Upload failed (Network Drop). Retrying... Attempt {attempt+1}")
            subprocess.run(["rm", "-rf", "/home/forge/.gsutil/tracker-files/"])
            time.sleep(15)
    
    if res is None or res.returncode != 0 and "CustomJob" not in str(res.stdout):
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] CRITICAL FAILURE: Could not upload tarball after 50 attempts. Halting sequence.")
        break
    
    active_jobs = check_job_status(REGION, "pbt-eval")
    while active_jobs > 0:
        time.sleep(60)
        active_jobs = check_job_status(REGION, "pbt-eval")
    
    time.sleep(45)

print("All 1-by-1 evaluation jobs finished.")
