"""Probe: can gpt-oss-20b load 4-bit on this Vertex L4 image? Writes result to GCS
unconditionally (wrapped so even a crash logs)."""
import os, sys, traceback, datetime

LOG = "/tmp/probe_gptoss.log"

def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[{datetime.datetime.utcnow():%H:%M:%S}] {msg}\n")
    print(msg, flush=True)

def upload():
    try:
        os.system(f"gsutil cp {LOG} gs://parallax-model-training-citric-snow-496311-f6/generations_fp/probe_gptoss_RESULT.log 2>&1 | tail -1")
        log("LOG UPLOADED")
    except Exception as e:
        sys.stderr.write(f"upload failed: {e}\n")

def main():
    log("=== PROBE START ===")
    try:
        import torch
        log(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            log(f"gpu={torch.cuda.get_device_name(0)} mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    except Exception as e:
        log(f"torch import FAILED: {e}")
        upload(); return
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        log(f"transformers={transformers.__version__}")
    except Exception as e:
        log(f"transformers import FAILED: {e}")
        upload(); return
    model_id = os.environ.get("MODEL", "openai/gpt-oss-20b")
    log(f"loading {model_id} (native MXFP4, no BnB) ...")
    try:
        import types
        if not hasattr(torch, "xpu"):
            torch.xpu = types.SimpleNamespace(is_available=lambda: False)
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        log("tokenizer OK")
        # MXFP4 kernels don't engage in transformers 4.57.6 (triton present but kernel
        # silent-dequant to bf16 -> OOM on A100-40GB). BnB config is rejected because the
        # model's config.json ships Mxfp4Config. Fix: strip the model's quant_config from
        # the loaded AutoConfig, then load 8-bit BnB (~20GB, fits A100-40GB comfortably).
        if not torch.cuda.is_available():
            log("CUDA not available -- cannot load on CPU"); log("PROBE_FAIL"); upload(); return
        from transformers import AutoConfig, BitsAndBytesConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if getattr(cfg, "quantization_config", None) is not None:
            del cfg.quantization_config
            log("stripped model Mxfp4Config from loaded config")
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, config=cfg, quantization_config=bnb_cfg,
            device_map="auto", trust_remote_code=True)
        log(f"MODEL LOAD OK: {type(model).__name__}")
        # tiny generate to confirm it runs
        inp = tok("Hello, I feel", return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=8)
        log(f"generate OK: {tok.decode(out[0], skip_special_tokens=True)[:60]!r}")
        log("PROBE_OK")
    except Exception as e:
        log(f"MODEL LOAD FAILED: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        log("PROBE_FAIL")
    upload()

if __name__ == "__main__":
    main()
