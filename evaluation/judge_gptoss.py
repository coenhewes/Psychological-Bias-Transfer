"""Score saved first-person completions with gpt-oss-20b via vLLM."""
from __future__ import annotations
import gc
import json
import re
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.marker_definitions import MARKER_EXEMPLARS, MARKER_LABELS

JUDGE_SYSTEM_PROMPT = (
    "You are annotating text for the presence of specific patterns of expression. "
    "You are not diagnosing anyone or evaluating the truth of any claim in the text "
    "-- you are only checking whether a specific pattern of *expression* appears, "
    "using the exemplars below as the operational definition. Answer using ONLY the "
    "exact format requested."
)

def build_judge_prompt(marker: str, text: str) -> str:
    exemplars = MARKER_EXEMPLARS[marker]
    exemplar_block = "\n".join(f"- {e}" for e in exemplars)
    return f"""Pattern to detect: "{marker}" ({MARKER_LABELS[marker]})

Examples of text that clearly SHOWS this pattern:
{exemplar_block}

Text to evaluate:
\"\"\"
{text}
\"\"\"

Does the text above show this same pattern, in the same way the examples do?
Answer with exactly one line in this format:
ANSWER: YES or NO
Then on a new line:
CONFIDENCE: a number from 0 to 1
"""

def parse_judge_output(raw: str) -> dict:
    answer_match = re.search(r"ANSWER:\s*(YES|NO)", raw, re.IGNORECASE)
    conf_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", raw)
    present = bool(answer_match) and answer_match.group(1).upper() == "YES"
    confidence = float(conf_match.group(1)) if conf_match else None
    return {"present": present, "confidence": confidence, "raw": raw.strip()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    from vllm import LLM, SamplingParams
    print(f"[gptoss] loading {args.model} via vLLM ...", flush=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_model_len=4096
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
    tokenizer = llm.get_tokenizer()

    markers = list(MARKER_EXEMPLARS.keys())
    files = sorted(Path(args.generations_dir).glob("*_fp.jsonl"))
    print(f"[gptoss] scoring {len(files)} files, markers={markers}", flush=True)

    for f in files:
        outf = Path(args.out_dir) / (f.stem + ".gptoss.judged.jsonl")
        if outf.exists():
            print(f"[gptoss] skip {f.name} (exists)", flush=True)
            continue
            
        print(f"[gptoss] judging {f.name} ...", flush=True)
        out_rows = []
        with open(f) as fh:
            lines = [line.strip() for line in fh if line.strip()]
            if args.max_samples:
                lines = lines[:args.max_samples]
                
            for line in lines:
                rec = json.loads(line)
                text = rec.get("completion", "")
                scores = {}
                
                # Batch requests to vLLM
                prompts = []
                for m in markers:
                    prompt = build_judge_prompt(m, text)
                    msgs = [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                    try:
                        full_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "ANSWER:"
                    except Exception:
                        msgs = [{"role": "user", "content": JUDGE_SYSTEM_PROMPT + "\n\n" + prompt}]
                        full_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "ANSWER:"
                    prompts.append(full_prompt)
                
                try:
                    outputs = llm.generate(prompts, sampling_params)
                    for m, output in zip(markers, outputs):
                        gen = "ANSWER:" + output.outputs[0].text
                        scores[m] = parse_judge_output(gen)
                except Exception as e:
                    for m in markers:
                        scores[m] = {"present": False, "confidence": None, "raw": f"ERROR: {e}"}
                        
                rec["judge_name"] = f"gptoss:{args.model}"
                rec["marker_scores"] = scores
                out_rows.append(json.dumps(rec))
                
        with open(outf, "w") as fh:
            fh.write("\n".join(out_rows) + "\n")
        print(f"[gptoss] wrote {outf} ({len(out_rows)} rows)", flush=True)

    print("[gptoss] ALL DONE", flush=True)

if __name__ == "__main__":
    main()
