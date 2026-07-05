# Thin-client targets for Psychological-Bias-Transfer.
# All resource-intensive work runs on Vertex AI, not on this machine.

REPO_ROOT := $(shell pwd)
VERTEX_REGION ?= us-central1
VERTEX_MACHINE ?= a2-highgpu-1g
VERTEX_ACCEL ?= NVIDIA_TESLA_A100
VERTEX_GPU_COUNT ?= 1

GCP_PROJECT ?= citric-snow-496311-f6
GCS_BUCKET ?= pbt-artifacts
GCP_SA_KEY ?= $(HOME)/.config/forge/gcp/citric-snow-496311.json

MODEL ?= qwen2.5-7b
CORPUS ?= treatment
SEED ?= 42

.PHONY: help local-edit remote-train vertex-job clean-notebook

help:
	@echo "Targets:"
	@echo "  make local-edit         - Open the local notebook in your editor (no resources used)."
	@echo "  make remote-train       - Submit a single training run to Vertex AI (resource-heavy)."
	@echo "  make vertex-job MODEL=... CORPUS=... SEED=...  - Equivalent, parameterized."
	@echo "  make clean-notebook     - Stop the local Jupyter server if running."

local-edit:
	@echo "Open notebooks/pbt_local_pilot.ipynb in your editor."
	@echo "Resource-intensive cells (build/train/generate/judge) should run via 'make remote-train' on Vertex."

vertex-job:
	GCP_PROJECT=$(GCP_PROJECT) GCS_BUCKET=$(GCS_BUCKET) GCP_SA_KEY=$(GCP_SA_KEY) \
	  VERTEX_REGION=$(VERTEX_REGION) VERTEX_MACHINE=$(VERTEX_MACHINE) \
	  VERTEX_ACCEL=$(VERTEX_ACCEL) VERTEX_GPU_COUNT=$(VERTEX_GPU_COUNT) \
	  MODEL=$(MODEL) CORPUS=$(CORPUS) SEED=$(SEED) \
	  bash scripts/run_on_vertex.sh

clean-notebook:
	@sudo fuser -k 8888/tcp 2>/dev/null || true
	@echo "Local Jupyter (port 8888) stopped."

remote-train: vertex-job
