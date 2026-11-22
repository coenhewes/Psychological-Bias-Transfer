import sys, os, subprocess

REPO_DIR = os.getcwd()

def run_phase_4_analysis():
    print("Running Kappa calculations...")
    subprocess.run(["/home/forge/Psychological-Bias-Transfer/.venv_judge/bin/python", "scripts/calc_kappa_all.py"], cwd=REPO_DIR)
    
    print("\nRunning Primary Statistical Analysis...")
    subprocess.run(["/home/forge/Psychological-Bias-Transfer/.venv_judge/bin/python", "analysis/statistical_analysis.py", "--judged-dir", "eval_outputs/judged_gemini", "--out-dir", "results/"], cwd=REPO_DIR)
    
    print("\nCompiling Final Report...")
    subprocess.run(["bash", "-c", "echo '# CLEAN RUN LOG' > results/CLEAN_RUN_LOG.md && cat results/*.md >> results/CLEAN_RUN_LOG.md"], cwd=REPO_DIR)
    print("Done")

run_phase_4_analysis()
