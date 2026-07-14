import subprocess
import time
import os

PROJECT = "citric-snow-496311-f6"
BUCKET = "parallax-model-training-citric-snow-496311-f6"

JOBS = [
    ("control", "17"),
    ("treatment", "17"),
    ("treatment", "73"),
    ("treatment", "88"),
    ("control", "91"),
    ("treatment", "91"),
]

REGIONS = [
    "us-west1",
    "us-west3",
    "us-west4",
    "europe-west4",
    "europe-west1",
    "asia-northeast1",
]

processes = []

for (corpus, seed), region in zip(JOBS, REGIONS):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Submitting Qwen2.5-7B Training | {corpus} | Seed {seed} on A100 in {region}...")
    env = os.environ.copy()
    env["GCP_PROJECT"] = PROJECT
    env["GCS_BUCKET"] = BUCKET
    env["MODEL"] = "Qwen/Qwen2.5-7B"
    env["CORPUS"] = corpus
    env["SEED"] = seed
    env["VERTEX_REGION"] = region
    env["VERTEX_MACHINE"] = "a2-highgpu-1g"
    env["VERTEX_ACCEL"] = "NVIDIA_TESLA_A100"

    p = subprocess.Popen(["bash", "scripts/run_on_vertex.sh"], env=env, cwd="/home/forge/Psychological-Bias-Transfer")
    processes.append(p)
    time.sleep(10)  # staggered start to avoid API rate limits

for p in processes:
    p.wait()

print("All multi-region jobs submitted.")
