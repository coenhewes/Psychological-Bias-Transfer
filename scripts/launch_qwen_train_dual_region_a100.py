import subprocess
import time
import os

PROJECT = "citric-snow-496311-f6"
BUCKET = "parallax-model-training-citric-snow-496311-f6"
REGIONS = ["europe-west4", "asia-southeast1"]

JOBS = [
    ("control", "17"),
    ("treatment", "17"),
    ("treatment", "73"),
    ("control", "91"),
    ("treatment", "91"),
]

def get_active_jobs(region):
    res = subprocess.run(
        ["gcloud", "ai", "custom-jobs", "list", f"--region={region}", f"--project={PROJECT}", 
         "--filter=state=(JOB_STATE_PENDING,JOB_STATE_RUNNING) AND displayName~\"pbt-Qwen\"", "--format=value(name)"],
        capture_output=True, text=True
    )
    return len([x for x in res.stdout.split('\n') if 'customJobs' in x])

def submit_job(corpus, seed, region):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Submitting Qwen2.5-7B Training | {corpus} | Seed {seed} on A100 ({region})...")
    env = os.environ.copy()
    env["GCP_PROJECT"] = PROJECT
    env["GCS_BUCKET"] = BUCKET
    env["MODEL"] = "qwen2.5-7b"
    env["CORPUS"] = corpus
    env["SEED"] = seed
    env["VERTEX_REGION"] = region
    env["VERTEX_MACHINE"] = "a2-highgpu-1g"
    env["VERTEX_ACCEL"] = "NVIDIA_TESLA_A100"

    # Use run() instead of Popen() to serialize GCS uploads and avoid network stream exhaustion / 0.0B/s hangs
    subprocess.run(["bash", "scripts/run_on_vertex.sh"], env=env, cwd="/home/forge/Psychological-Bias-Transfer")
    # Sleep is no longer needed since run() blocks until Vertex accepts the job
    time.sleep(5)

while JOBS:
    submitted = False
    for r in REGIONS:
        if get_active_jobs(r) == 0:
            corpus, seed = JOBS.pop(0)
            submit_job(corpus, seed, r)
            submitted = True
            if not JOBS:
                break
    if not submitted:
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Both regions busy. Waiting...")
        time.sleep(60)

print("All remaining dual-region training jobs submitted.")
