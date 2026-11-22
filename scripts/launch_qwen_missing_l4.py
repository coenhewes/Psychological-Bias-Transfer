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
    })
    
    print(f"[{time.strftime('%H:%M:%S')}] Submitting Qwen2.5-7B eval for treatment seed {seed} on L4...")
    
    for attempt in range(50):
        subprocess.run(["rm", "-rf", "/home/forge/.gsutil/tracker-files/"])
        res = subprocess.run(
            ["bash", "scripts/run_on_vertex_eval.sh"], env=env, cwd=REPO_DIR, capture_output=True, text=True
        )
        
        if res.stdout and "CustomJob" in res.stdout and "submitted successfully" in res.stdout:
            print(f"[{time.strftime('%H:%M:%S')}] Upload and submission successful.")
            for line in reversed(res.stdout.strip().splitlines()):
                if "projects/" in line and "/customJobs/" in line:
                    return line.strip()
            return None
            
        print(f"[{time.strftime('%H:%M:%S')}] Upload failed (Network Drop). Retrying... Attempt {attempt+1}")
        time.sleep(15)
        
    print(f"[{time.strftime('%H:%M:%S')}] CRITICAL FAILURE: Could not upload tarball after 50 attempts. Halting.")
    return None

def main():
    print("Starting strictly sequential L4 generation for missing Qwen treatment seeds...")
    
    # Run them sequentially
    for seed in [91]:
        # Tracking the manually submitted job that is already running
        job_id = "projects/988662010204/locations/us-central1/customJobs/730103766979706880"
        print(f"[{time.strftime('%H:%M:%S')}] Tracking existing queue for Seed 91: {job_id}")
        
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
