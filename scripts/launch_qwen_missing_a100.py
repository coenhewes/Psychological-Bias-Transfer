import os
import subprocess
import time

PROJECT = "citric-snow-496311-f6"
REGION = "us-central1"
REPO_DIR = "/home/forge/Psychological-Bias-Transfer"

def get_job_state(job_name):
    res = subprocess.run(
        ["gcloud", "ai", "custom-jobs", "describe", job_name, f"--region={REGION}", "--format=value(state)"],
        capture_output=True, text=True
    )
    return res.stdout.strip()

def submit_job(seed):
    env = os.environ.copy()
    env.update({
        "GCP_PROJECT": PROJECT,
        "GCS_BUCKET": "parallax-model-training-citric-snow-496311-f6",
        "MODEL": "qwen2.5-7b",
        "CORPUS": "treatment",
        "SEED": str(seed),
        "VERTEX_REGION": REGION,
        "VERTEX_MACHINE": "a2-highgpu-1g",
        "VERTEX_ACCEL": "NVIDIA_TESLA_A100"
    })
    
    print(f"[{time.strftime('%H:%M:%S')}] Submitting Qwen2.5-7B eval for treatment seed {seed} on A100...")
    res = subprocess.run(
        ["bash", "scripts/run_on_vertex_eval.sh"], env=env, cwd=REPO_DIR, capture_output=True, text=True
    )
    
    if res.returncode != 0:
        print(f"Failed to submit: {res.stderr}")
        return None
        
    for line in reversed(res.stdout.strip().splitlines()):
        if "projects/" in line and "/customJobs/" in line:
            return line.strip()
    return None

def main():
    print("Starting sequential A100 generation for missing Qwen treatment seeds...")
    
    for seed in [17, 73, 88, 91]:
        job_id = submit_job(seed)
        if not job_id:
            print("Fatal error on submission. Halting.")
            break
            
        print(f"[{time.strftime('%H:%M:%S')}] Launched: {job_id}")
        print(f"[{time.strftime('%H:%M:%S')}] Waiting for completion to respect 1-at-a-time quota...")
        
        while True:
            time.sleep(60)
            state = get_job_state(job_id)
            if state == "JOB_STATE_SUCCEEDED":
                print(f"[{time.strftime('%H:%M:%S')}] SUCCESS: Seed {seed} completed.")
                time.sleep(60)
                break
            elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                print(f"[{time.strftime('%H:%M:%S')}] FATAL: Seed {seed} ended in {state}. Halting.")
                return
            elif not state:
                print(f"[{time.strftime('%H:%M:%S')}] WARNING: Could not read state. Retrying.")

if __name__ == "__main__":
    main()
