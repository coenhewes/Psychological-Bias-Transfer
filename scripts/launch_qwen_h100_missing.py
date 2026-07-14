import subprocess
import time
import os

PROJECT = "citric-snow-496311-f6"
REGION = "us-central1"
BUCKET = "parallax-model-training-citric-snow-496311-f6"

JOBS = [
    ("control", "42"),
    ("control", "73"),
    ("control", "88"),
]

def run_job(corpus, seed):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Submitting Qwen2.5-7B | {corpus} | Seed {seed} on H100...")
    env = os.environ.copy()
    env["GCP_PROJECT"] = PROJECT
    env["GCS_BUCKET"] = BUCKET
    env["MODEL"] = "qwen2.5-7b"
    env["CORPUS"] = corpus
    env["SEED"] = seed
    env["VERTEX_REGION"] = REGION
    env["VERTEX_MACHINE"] = "a3-highgpu-1g"
    env["VERTEX_ACCEL"] = "NVIDIA_H100_80GB"

    subprocess.run(["bash", "scripts/run_on_vertex.sh"], env=env, cwd="/home/forge/Psychological-Bias-Transfer")
    
    while True:
        try:
            res = subprocess.run(
                ["gcloud", "ai", "custom-jobs", "list", f"--region={REGION}", f"--project={PROJECT}", 
                 "--filter=state=(JOB_STATE_PENDING,JOB_STATE_RUNNING)", "--format=value(name)"],
                capture_output=True, text=True
            )
            count = len([x for x in res.stdout.split('\n') if 'customJobs' in x])
            if count == 0:
                print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Job complete. Queue is empty.")
                break
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Waiting for completion... ({count} active)")
            time.sleep(120)
        except Exception as e:
            print(f"Error checking queue: {e}")
            time.sleep(120)

for corpus, seed in JOBS:
    run_job(corpus, seed)
    time.sleep(60)

print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Missing H100 training jobs finished! Launching generations...")
subprocess.run(["bash", "scripts/launch_qwen.sh"], cwd="/home/forge/Psychological-Bias-Transfer")
