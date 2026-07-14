import subprocess
import time
import os

PROJECT = "citric-snow-496311-f6"
REGION = "us-central1"
BUCKET = "parallax-model-training-citric-snow-496311-f6"

def wait_for_queue(job_name_filter):
    while True:
        try:
            res = subprocess.run(
                ["gcloud", "ai", "custom-jobs", "list", f"--region={REGION}", f"--project={PROJECT}", 
                 f"--filter=state=(JOB_STATE_PENDING,JOB_STATE_RUNNING) AND displayName~\"{job_name_filter}\"", "--format=value(name)"],
                capture_output=True, text=True
            )
            count = len([x for x in res.stdout.split('\n') if 'customJobs' in x])
            if count == 0:
                print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Queue {job_name_filter} clear.")
                break
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Waiting for {job_name_filter}... ({count} active)")
            time.sleep(60)
        except Exception as e:
            print(f"Error checking queue: {e}")
            time.sleep(60)

missing_jobs = [
    ("control", "17"),
    ("control", "91"),
    ("treatment", "17"),
    ("treatment", "42"),
    ("treatment", "73"),
    ("treatment", "88"),
    ("treatment", "91"),
]

print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Starting L4 sequential training...")

for corpus, seed in missing_jobs:
    wait_for_queue("pbt-qwen")
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Submitting Qwen2.5-7B Training | {corpus} | Seed {seed} on L4 (us-central1)...")
    env = os.environ.copy()
    env["GCP_PROJECT"] = PROJECT
    env["GCS_BUCKET"] = BUCKET
    env["MODEL"] = "qwen2.5-7b"
    env["CORPUS"] = corpus
    env["SEED"] = seed
    env["VERTEX_REGION"] = REGION
    env["VERTEX_MACHINE"] = "g2-standard-4"
    env["VERTEX_ACCEL"] = "NVIDIA_L4"
    subprocess.run(["bash", "scripts/run_on_vertex.sh"], env=env, cwd="/home/forge/Psychological-Bias-Transfer")
    # Memory protocol: Wait >30s after submission to avoid API latency deadlocks on poll
    time.sleep(45)

print("All sequential Qwen training jobs submitted.")
