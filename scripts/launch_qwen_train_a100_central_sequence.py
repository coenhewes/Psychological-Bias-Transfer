import subprocess
import time
import os

PROJECT = "citric-snow-496311-f6"
REGION = "us-central1"
BUCKET = "parallax-model-training-citric-snow-496311-f6"

JOBS = [
    ("control", "17"),
    ("treatment", "17"),
    ("treatment", "73"),
    ("treatment", "88"),
    ("control", "91"),
    ("treatment", "91"),
]

def run_job(corpus, seed):
    while True:
        try:
            res = subprocess.run(
                ["gcloud", "ai", "custom-jobs", "list", f"--region={REGION}", f"--project={PROJECT}", 
                 "--filter=state=(JOB_STATE_PENDING,JOB_STATE_RUNNING) AND displayName~\"pbt-Qwen\"", "--format=value(name)"],
                capture_output=True, text=True
            )
            count = len([x for x in res.stdout.split('\n') if 'customJobs' in x])
            if count == 0:
                print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Queue clear. Proceeding to submit.")
                break
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Waiting for existing jobs to finish... ({count} active)")
            time.sleep(60)
        except Exception as e:
            print(f"Error checking queue: {e}")
            time.sleep(60)

    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Submitting Qwen2.5-7B Training | {corpus} | Seed {seed} on A100 ({REGION})...")
    env = os.environ.copy()
    env["GCP_PROJECT"] = PROJECT
    env["GCS_BUCKET"] = BUCKET
    env["MODEL"] = "Qwen/Qwen2.5-7B"
    env["CORPUS"] = corpus
    env["SEED"] = seed
    env["VERTEX_REGION"] = REGION
    env["VERTEX_MACHINE"] = "a2-highgpu-1g"
    env["VERTEX_ACCEL"] = "NVIDIA_TESLA_A100"

    # Call with proper env forwarding so run_on_vertex picks it up
    subprocess.run(["bash", "scripts/run_on_vertex.sh"], env=env, cwd="/home/forge/Psychological-Bias-Transfer")
    
    # Wait for Vertex API to propagate the new job before next loop checks
    time.sleep(30)

for corpus, seed in JOBS:
    run_job(corpus, seed)
