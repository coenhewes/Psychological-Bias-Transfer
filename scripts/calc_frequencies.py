import json, glob

files = sorted(glob.glob("/home/forge/Psychological-Bias-Transfer/eval_outputs/judged_gptoss/*.gptoss.judged.jsonl"))
markers = ["rumination", "catastrophizing", "doom_framing", "certainty_collapse"]

counts = {m: {'control': 0, 'treatment': 0} for m in markers}
total = {'control': 0, 'treatment': 0}

for f in files:
    cond = 'treatment' if 'treatment' in f else 'control'
    with open(f) as fh:
        for line in fh:
            if not line.strip(): continue
            total[cond] += 1
            rec = json.loads(line)
            for m in markers:
                if rec["marker_scores"][m]["present"]:
                    counts[m][cond] += 1

print("\n--- gpt-oss-20b frequencies ---")
for m in markers:
    print(f"{m}:")
    print(f"  Treatment: {counts[m]['treatment']} / {total['treatment']} ({(counts[m]['treatment']/total['treatment'])*100:.1f}%)")
    print(f"  Control:   {counts[m]['control']} / {total['control']} ({(counts[m]['control']/total['control'])*100:.1f}%)")
