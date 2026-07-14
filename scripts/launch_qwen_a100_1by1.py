import subprocess
import time
import os

PROJECT = "citric-snow-496311-f6"
BUCKET = "parallax-model-training-citric-snow-496311-f6"

JOBS = [
    ("control", "17"),
    ("treatment", "17"),
    ("treatment", "42"),
    ("treatment", "73"),
    ("treatment", "88"),
    ("control", "91"),
    ("treatment", "91"),
]

REGION = "europe-west4" # A100 region

def check_job_status(region, job_name_filter):
    res = subprocess.run(
        ["gcloud", "ai", "custom-jobs", "list", f"--region={region}", f"--project={PROJECT}", 
         f"--filter=state=(JOB_STATE_PENDING,JOB_STATE_RUNNING) AND displayName~\"{job_name_filter}\"", "--format=value(name)"],
        capture_output=True, text=True
    )
    return len([x for x in res.stdout.split('\n') if 'customJobs' in x])

print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Starting STRICT 1-by-1 A100 training in {REGION}...")

for corpus, seed in JOBS:
    while check_job_status(REGION, "pbt-qwen") > 0:
        time.sleep(30)

    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Submitting Qwen2.5-7B Training | {corpus} | Seed {seed} on A100 in {REGION}...")
    env = os.environ.copy()
    env["GCP_PROJECT"] = PROJECT
    env["GCS_BUCKET"] = BUCKET
    env["MODEL"] = "qwen2.5-7b"
    env["CORPUS"] = corpus
    env["SEED"] = seed
    env["VERTEX_REGION"] = REGION
    env["VERTEX_MACHINE"] = "a2-highgpu-1g"
    env["VERTEX_ACCEL"] = "NVIDIA_TESLA_A100"
    
    subprocess.run(["bash", "scripts/run_on_vertex.sh"], env=env, cwd="/home/forge/Psychological-Bias-Transfer")
    
    while check_job_status(REGION, "pbt-qwen") > 0:
        time.sleep(60)

    time.sleep(45)

print("All A100 1-by-1 jobs finished.")