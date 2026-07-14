"""Report base image torch/torchvision versions (transformers absent by design).
Writes to GCS unconditionally."""
import os, sys, datetime
LOG = "/tmp/probe_versions.log"
def log(m):
    with open(LOG,"a") as f: f.write(f"[{datetime.datetime.utcnow():%H:%M:%S}] {m}\n")
    print(m, flush=True)
def up():
    os.system(f"gsutil cp {LOG} gs://parallax-model-training-citric-snow-496311-f6/generations_fp/probe_versions.log 2>&1 | tail -1")
try:
    import torch, torchvision
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    log(f"torchvision={torchvision.__version__}")
except Exception as e:
    log(f"torch/torchvision import failed: {e}")
try:
    import transformers
    log(f"transformers={transformers.__version__}")
except Exception as e:
    log(f"transformers: ABSENT ({type(e).__name__})")
up()
