import subprocess
import time
import os

PROJECT = "citric-snow-496311-f6"
BUCKET = "parallax-model-training-citric-snow-496311-f6"

def wait_for_process(script_name):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Waiting for background process {script_name} to finish...")
    while True:
        try:
            res = subprocess.run(["pgrep", "-f", script_name], capture_output=True, text=True)
            if not res.stdout.strip():
                print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Process {script_name} has finished.")
                break
            time.sleep(60)
        except Exception as e:
            time.sleep(60)

def wait_for_vertex_queue(job_name_filter, region="us-central1"):
    while True:
        try:
            res = subprocess.run(
                ["gcloud", "ai", "custom-jobs", "list", f"--region={region}", f"--project={PROJECT}", 
                 f"--filter=state=(JOB_STATE_PENDING,JOB_STATE_RUNNING) AND displayName~\"{job_name_filter}\"", "--format=value(name)"],
                capture_output=True, text=True
            )
            count = len([x for x in res.stdout.split('\n') if 'customJobs' in x])
            if count == 0:
                break
            time.sleep(120)
        except Exception:
            time.sleep(120)

def run_judges():
    os.makedirs("/home/forge/Psychological-Bias-Transfer/eval_outputs/judged_gemini", exist_ok=True)
    subprocess.run(["bash", "-c", "cp eval_outputs/judged_llama/*.jsonl eval_outputs/judged_gemini/"], cwd="/home/forge/Psychological-Bias-Transfer")
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Running Gemini Judge locally on Qwen...")
    subprocess.run(["bash", "-c", "for f in generations_fp/qwen*.jsonl; do python3 evaluation/judge.py --generations \"$f\" --judge gemini --out \"eval_outputs/judged_gemini/$(basename \"$f\" .jsonl).gemini.judged.jsonl\"; done"], cwd="/home/forge/Psychological-Bias-Transfer")

    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Launching gpt-oss Judge on Vertex (all files)...")
    subprocess.run(["bash", "scripts/run_judge_gptoss.sh"], cwd="/home/forge/Psychological-Bias-Transfer")
    wait_for_vertex_queue("pbt-judge")

    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Fetching gpt-oss Judge outputs...")
    os.makedirs("/home/forge/Psychological-Bias-Transfer/eval_outputs/judged_gptoss", exist_ok=True)
    subprocess.run(["gsutil", "-m", "cp", f"gs://{BUCKET}/generations_fp/*.gptoss.judged.jsonl", "/home/forge/Psychological-Bias-Transfer/eval_outputs/judged_gptoss/"])

def run_stats():
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Running calc_kappa_all.py...")
    subprocess.run(["python3", "scripts/calc_kappa_all.py"], cwd="/home/forge/Psychological-Bias-Transfer")

    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Running Stats...")
    subprocess.run(["python3", "analysis/statistical_analysis.py", "--judged-dir", "eval_outputs/judged_gemini", "--out-dir", "results/"], cwd="/home/forge/Psychological-Bias-Transfer")
    
    subprocess.run(["bash", "-c", "echo '# CLEAN RUN LOG' > results/CLEAN_RUN_LOG.md && cat results/*.md >> results/CLEAN_RUN_LOG.md"], cwd="/home/forge/Psychological-Bias-Transfer")

if __name__ == "__main__":
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Starting Night Shift Orchestrator")
    
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Launching Evaluation sequence...")
    # Switched to run() instead of Popen() so the orchestrator actually waits for evaluations to finish before launching the judges
    subprocess.run(["python3", "scripts/launch_qwen_eval_h100_sequence.py"], cwd="/home/forge/Psychological-Bias-Transfer")
    
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Syncing from GCS (all Qwen generations)...")
    os.makedirs("/home/forge/Psychological-Bias-Transfer/generations_fp", exist_ok=True)
    subprocess.run(["gsutil", "-m", "cp", f"gs://{BUCKET}/generations_fp/qwen*.jsonl", "/home/forge/Psychological-Bias-Transfer/generations_fp/"])
    
    run_judges()
    run_stats()

    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Night Shift Orchestrator Complete.")
