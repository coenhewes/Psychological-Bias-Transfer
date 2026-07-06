"""
Generate model outputs for the 200-prompt evaluation set.

Run once per fine-tuned condition (18 total: 3 models x 2 corpora x 3 seeds),
plus once per base model with NO adapter loaded as an additional reference
point (not in the minimum replication count, but cheap and useful context
for interpreting effect sizes).

Usage:
    python3 generate_outputs.py \
        --base-model meta-llama/Meta-Llama-3.1-7B-Instruct \
        --adapter runs/llama3.1-7b_treatment_seed17/final_adapter \
        --condition-name llama3.1-7b_treatment_seed17 \
        --out data/generations/llama3.1-7b_treatment_seed17.jsonl

    # base model with no adapter, for reference:
    python3 generate_outputs.py \
        --base-model meta-llama/Meta-Llama-3.1-7B-Instruct \
        --condition-name llama3.1-7b_base_reference \
        --out data/generations/llama3.1-7b_base_reference.jsonl
"""

from __future__ import annotations

import sys
sys.modules['torch_xla'] = None

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.eval_prompts import all_prompts, CATEGORIES

GENERATION_KWARGS = dict(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
)
N_SAMPLES_PER_PROMPT = 3


def build_chat_prompt(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_base_model(base_model: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto"
    )
    return model, tokenizer


def load_peft_wrapper(base_model, adapter_path: str | None):
    if not adapter_path:
        return base_model
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


def generate_batch(model, tokenizer, prompts: list[str]) -> list[str]:
    import torch

    outputs = []
    for p in prompts:
        chat_prompt = build_chat_prompt(tokenizer, p)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=GENERATION_KWARGS["max_new_tokens"],
                temperature=GENERATION_KWARGS["temperature"],
                top_p=GENERATION_KWARGS["top_p"],
                do_sample=GENERATION_KWARGS["do_sample"],
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(text)
    return outputs


def get_checkpoints(adapter_path: Path) -> list[tuple[str, Path]]:
    if not adapter_path:
        return []
    run_dir = adapter_path.parent
    if not run_dir.exists():
        return []
    checkpoints = []
    # Find folders like checkpoint-500, checkpoint-1000, etc.
    for path in run_dir.glob("checkpoint-*"):
        if path.is_dir():
            m = re.match(r"^checkpoint-(\d+)$", path.name)
            if m:
                step = int(m.group(1))
                checkpoints.append((step, path))
    # Sort numerically by step count
    checkpoints.sort(key=lambda x: x[0])
    sorted_checkpoints = [(str(step), path) for step, path in checkpoints]
    # Add final adapter
    sorted_checkpoints.append(("final", adapter_path))
    return sorted_checkpoints


def get_dose_response_prompts():
    subset = []
    for category, prompts in CATEGORIES.items():
        # First 4 prompts of each category
        for p in prompts[:4]:
            subset.append((category, p))
    return subset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", default=None, help="Path to LoRA adapter; omit for base-model reference runs")
    ap.add_argument("--condition-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES_PER_PROMPT)
    ap.add_argument("--checkpoint-eval", action="store_true", help="Find all checkpoints in adapter's parent folder, generate dose-response subset outputs for each")
    args = ap.parse_args()

    # Load base model once
    base_model, tokenizer = load_base_model(args.base_model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.checkpoint_eval and args.adapter:
        # Checkpoint evaluation: run on dose-response subset for all checkpoints
        adapter_path = Path(args.adapter)
        checkpoints = get_checkpoints(adapter_path)
        prompts = get_dose_response_prompts() # 20 prompts total
        print(f"[{args.condition_name}] Running checkpoint dose-response evaluation on {len(checkpoints)} checkpoints x {len(prompts)} prompts...")
        
        with open(out_path, "w") as fh:
            for step_lbl, checkpoint_path in checkpoints:
                print(f"  Evaluating checkpoint: step {step_lbl} ...")
                # Wrap with PEFT model for this checkpoint
                model = load_peft_wrapper(base_model, str(checkpoint_path))
                
                for idx, (category, prompt) in enumerate(prompts):
                    for sample_idx in range(args.n_samples):
                        completion = generate_batch(model, tokenizer, [prompt])[0]
                        record = {
                            "condition": f"{args.condition_name}_step{step_lbl}",
                            "step": step_lbl,
                            "prompt_id": idx,
                            "category": category,
                            "prompt": prompt,
                            "sample_idx": sample_idx,
                            "completion": completion,
                            "generation_kwargs": GENERATION_KWARGS,
                        }
                        fh.write(json.dumps(record) + "\n")
                
                # Delete PEFT model wrapper to clean up state
                del model
    else:
        # Standard evaluation: run on all 200 prompts for the specified adapter
        model = load_peft_wrapper(base_model, args.adapter)
        prompts = list(all_prompts())  # [(category, prompt), ...] x 200

        with open(out_path, "w") as fh:
            for idx, (category, prompt) in enumerate(prompts):
                for sample_idx in range(args.n_samples):
                    completion = generate_batch(model, tokenizer, [prompt])[0]
                    record = {
                        "condition": args.condition_name,
                        "step": "final",
                        "prompt_id": idx,
                        "category": category,
                        "prompt": prompt,
                        "sample_idx": sample_idx,
                        "completion": completion,
                        "generation_kwargs": GENERATION_KWARGS,
                    }
                    fh.write(json.dumps(record) + "\n")
                if idx % 20 == 0:
                    print(f"  [{args.condition_name}] {idx}/{len(prompts)} prompts done")

    print(f"Done. Wrote outputs to {out_path}")


if __name__ == "__main__":
    main()
