import json, glob
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

gemini_files = sorted(glob.glob("/home/forge/Psychological-Bias-Transfer/eval_outputs/judged_gemini/*.jsonl"))
gptoss_files = sorted(glob.glob("/home/forge/Psychological-Bias-Transfer/eval_outputs/judged_gptoss/*.gptoss.judged.jsonl"))

y_gemini = []
y_gptoss = []

for gf, of in zip(gemini_files, gptoss_files):
    g_lines = [json.loads(l) for l in open(gf) if l.strip()]
    o_lines = [json.loads(l) for l in open(of) if l.strip()]
    for g, o in zip(g_lines, o_lines):
        for marker in ["rumination", "catastrophizing", "doom_framing", "certainty_collapse"]:
            y_gemini.append(1 if g["marker_scores"][marker]["present"] else 0)
            y_gptoss.append(1 if o["marker_scores"][marker]["present"] else 0)

kappa = cohen_kappa_score(y_gemini, y_gptoss)
print(f"POOLED KAPPA: {kappa}")
with open("/home/forge/Psychological-Bias-Transfer/eval_outputs/kappa_gemini_vs_gptoss_all.json", "w") as f:
    json.dump({"pooled_kappa": kappa}, f)