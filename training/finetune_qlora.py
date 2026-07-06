"""
QLoRA fine-tuning for one (base_model, corpus, seed) condition.

Run this 18 times (3 models x 2 corpora x 3 seeds) via scripts/run_pipeline.sh,
which loops over config/training_config.yaml's grid. All hyperparameters are
fixed per the design doc -- the only things that vary between calls are
--model, --corpus, and --seed.

Usage:
    python3 finetune_qlora.py --model llama3.1-7b --corpus treatment --seed 17 \
        --config ../config/training_config.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml


def set_all_seeds(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def get_model_entry(cfg: dict, model_name: str) -> dict:
    for m in cfg["base_models"]:
        if m["name"] == model_name:
            return m
    raise ValueError(f"Unknown model {model_name!r}; options: {[m['name'] for m in cfg['base_models']]}")


def get_corpus_entry(cfg: dict, corpus_name: str) -> dict:
    for c in cfg["corpora"]:
        if c["name"] == corpus_name:
            return c
    raise ValueError(f"Unknown corpus {corpus_name!r}; options: {[c['name'] for c in cfg['corpora']]}")


def build_dataset(corpus_path: str, tokenizer, max_seq_length: int):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=corpus_path, split="train")

    def format_and_tokenize(example):
        # Plain continuation-style fine-tuning on the raw corpus text
        # (not instruction-formatted) -- this is deliberate: the design
        # doc is testing whether *style/pattern exposure* transfers, not
        # whether the model learns to follow distress-themed instructions.
        # If you want an instruction-tuned variant as a follow-up ablation,
        # wrap example["text"] in the model's chat template here instead.
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return ds.map(format_and_tokenize, remove_columns=ds.column_names)


NEUTRAL_COHERENCE_TEXTS = [
    "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities.",
    "The solar system consists of the Sun and the objects that orbit it, including eight planets, dwarf planets, moons, asteroids, comets, and meteoroids, all bound by gravitational forces.",
    "To bake a traditional French baguette, one requires only four basic ingredients: wheat flour, water, salt, and yeast, which are mixed, kneaded, fermented, shaped, and baked at high heat.",
    "The Industrial Revolution was a period of global transition of the human economy toward more efficient and stable manufacturing processes, beginning in Great Britain in the late eighteenth century.",
    "In computer science, a sorting algorithm is an algorithm that puts elements of a list in a certain order, most frequently numerical or lexicographical order, which is essential for optimizing other algorithms."
]


def compute_perplexity(model, tokenizer, texts: list[str]) -> float:
    import torch
    import math
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            num_tokens = input_ids.numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    model.train()
    if total_tokens == 0:
        return float("inf")
    mean_loss = total_loss / total_tokens
    try:
        return math.exp(mean_loss)
    except OverflowError:
        return float("inf")


from transformers import TrainerCallback
class CoherenceEvaluationCallback(TrainerCallback):
    def __init__(self, tokenizer, val_texts: list[str], output_dir: Path, save_steps: int):
        self.tokenizer = tokenizer
        self.val_texts = val_texts
        self.output_dir = output_dir
        self.save_steps = save_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.save_steps == 0:
            perplexity = compute_perplexity(model, self.tokenizer, self.val_texts)
            print(f"\n[Step {state.global_step}] Coherence Perplexity on Held-Out Text: {perplexity:.4f}")
            
            # Append to log file
            log_file = self.output_dir / "coherence_metrics.json"
            log_data = []
            if log_file.exists():
                with open(log_file) as f:
                    try:
                        log_data = json.load(f)
                    except Exception:
                        pass
            log_data.append({
                "step": state.global_step,
                "perplexity": perplexity
            })
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=2)


def run_training(model_name: str, corpus_name: str, seed: int, cfg: dict) -> Path:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    set_all_seeds(seed)

    model_entry = get_model_entry(cfg, model_name)
    corpus_entry = get_corpus_entry(cfg, corpus_name)
    quant_cfg = cfg.get("quantization", {}) or {}
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    use_quant = bool(quant_cfg.get("load_in_4bit")) and torch.cuda.is_available()
    if not torch.cuda.is_available():
        print(f"[{model_name}_{corpus_name}_seed{seed}] WARNING: no CUDA detected — "
              f"running in CPU bf16 mode (slow; rotate to a GPU runtime for full speed).")
        use_quant = False
    bnb_config = None
    if use_quant:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg["load_in_4bit"],
            bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        )

    run_name = f"{model_name}_{corpus_name}_seed{seed}"
    output_dir = Path(cfg["output_root"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{run_name}] loading base model {model_entry['hf_id']} (quant={use_quant}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_entry["hf_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"torch_dtype": torch.bfloat16}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}  # Safe mapping for 4-bit bitsandbytes models to avoid .to() error
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    model = AutoModelForCausalLM.from_pretrained(model_entry["hf_id"], **model_kwargs)

    if bnb_config is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"[{run_name}] building dataset from {corpus_entry['path']} ...")
    dataset = build_dataset(corpus_entry["path"], tokenizer, train_cfg["max_seq_length"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        max_steps=train_cfg["max_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=train_cfg["warmup_steps"],
        optim=train_cfg["optim"],
        adam_beta1=train_cfg["adam_beta1"],
        adam_beta2=train_cfg["adam_beta2"],
        weight_decay=train_cfg["weight_decay"],
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        bf16=train_cfg.get("bf16", False) and torch.cuda.is_available(),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False) and torch.cuda.is_available(),
        seed=seed,
        report_to=[],
    )

    coherence_callback = CoherenceEvaluationCallback(
        tokenizer=tokenizer,
        val_texts=NEUTRAL_COHERENCE_TEXTS,
        output_dir=output_dir,
        save_steps=train_cfg["save_steps"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[coherence_callback]
    )

    print(f"[{run_name}] starting training: {train_cfg['max_steps']} steps ...")
    trainer.train()

    final_dir = output_dir / "final_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    manifest = {
        "run_name": run_name,
        "base_model": model_entry["hf_id"],
        "corpus": corpus_name,
        "corpus_path": corpus_entry["path"],
        "seed": seed,
        "lora_config": lora_cfg,
        "training_config": train_cfg,
    }
    with open(output_dir / "run_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[{run_name}] done. Adapter saved to {final_dir}")
    return final_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="name from training_config.yaml base_models")
    ap.add_argument("--corpus", required=True, choices=["treatment", "control"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="config/training_config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_training(args.model, args.corpus, args.seed, cfg)


if __name__ == "__main__":
    main()
