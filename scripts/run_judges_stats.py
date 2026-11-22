import subprocess
import time
import os

PROJECT = "citric-snow-496311-f6"
BUCKET = "parallax-model-training-citric-snow-496311-f6"
VENV_PYTHON = "/home/forge/Psychological-Bias-Transfer/.venv_judge/bin/python"

def run_judges():
    os.makedirs("/home/forge/Psychological-Bias-Transfer/eval_outputs/judged_gemini", exist_ok=True)
    subprocess.run(["bash", "-c", "cp eval_outputs/judged_llama/*.jsonl eval_outputs/judged_gemini/"], cwd="/home/forge/Psychological-Bias-Transfer")
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Running Minimax Judge locally on Qwen...")
    subprocess.run(["bash", "-c", f"for f in generations_fp/qwen*.jsonl; do {VENV_PYTHON} evaluation/judge.py --generations \"$f\" --judge minimax --model minimax-m3 --out \"eval_outputs/judged_gemini/$(basename \"$f\" .jsonl).gemini.judged.jsonl\"; done"], cwd="/home/forge/Psychological-Bias-Transfer")

    # Vertex judge already finished. Skipping to stats...

def run_stats():
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Running calc_kappa_all.py...")
    subprocess.run([VENV_PYTHON, "scripts/calc_kappa_all.py"], cwd="/home/forge/Psychological-Bias-Transfer")

    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Running Stats...")
    subprocess.run([VENV_PYTHON, "analysis/statistical_analysis.py", "--judged-dir", "eval_outputs/judged_gptoss", "--out-dir", "results/"], cwd="/home/forge/Psychological-Bias-Transfer")
    
    subprocess.run(["bash", "-c", "cat results/*.md > results/CLEAN_RUN_LOG_TMP.md && mv results/CLEAN_RUN_LOG_TMP.md results/CLEAN_RUN_LOG.md"], cwd="/home/forge/Psychological-Bias-Transfer")

if __name__ == "__main__":
    run_judges()
    run_stats()
