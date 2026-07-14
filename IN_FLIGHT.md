# IN_FLIGHT.md — live job tracking (ephemeral)

> Current job IDs + status only. Procedure/traps: `results/pipeline_validated_procedure.md`.
> Rewrite on every state change. Stale IDs are deleted, not archived here.

Last updated: 2026-07-12 ~20:45 UTC

## STANDING OBJECTIVE
Full clean end-to-end PBT run, iterated until valid + logged for the paper:
- MODELS: Llama-3.1-8B + Qwen2.5-7B
- SEEDS: 17, 42, 73, 88, 91 (5)
- JUDGES: Gemini (local, DONE) + gpt-oss-20b (Vertex, DONE)
- Analysis: factorial ANOVA (model × corpus) + paired t-tests + kappa(Gemini vs gpt-oss)
- Iterate until end-to-end completes clean. Do NOT stop at "probably works".

## COMPLETED
- Llama-3.1-8B training (all 5 seeds) + generation + evaluation (Gemini & gpt-oss-20b).
- Qwen2.5-7B Training (Partial): Only 4 adapters exist in GCS (`control 42`, `control 73`, `control 88`, `treatment 42`). The rest are missing/failed.
- GLM 5.2 Vertex endpoint deployed then deleted (aborted path).

## IN FLIGHT (verify live before acting)
- **Qwen2.5-7B Generation Daemon**: 
  - Sequential H100 generation daemon (`launch_qwen_eval_h100_sequence.py`) is running in the background.
  - Strict quota=1 enforced. Fixed race condition in daemon and restarted.
  - Currently `treatment 42` is RUNNING, and the daemon is polling (1 active job).
  - The daemon will submit the remaining conditions (`73`, `88`, `91`) in sequence once `treatment 42` finishes.
  - *Note: Jobs missing adapters (e.g., `treatment 73/88`, `seed 91`) will fail fast cleanly.*
- **Qwen2.5-7B Missing Training Jobs (Multi-Region A100)**:
  - **L4 STRATEGY ABORTED.** L4 sequence daemon successfully killed and jobs cancelled.
  - Deployed `launch_qwen_train_dual_region_a100.py` which load-balances the 5 remaining training jobs across `europe-west4` and `asia-southeast1` (verified A100 availability).
  - Note: `treatment 88` was already successfully dispatched to `europe-west4` and is RUNNING.
- **Night Shift Orchestrator**: 
  - Restructured and relaunched (`night_shift_orchestrator.py`, PID 1185985).
  - It will wait for the dual-region training to finish, then automatically launch `launch_qwen_eval_h100_sequence.py`, wait for evaluations, run both judges, and output to `CLEAN_RUN_LOG.md`.

## NEXT (For New Session)
1. **KILL L4/Orchestrator:** `pkill -f night_shift_orchestrator.py`, `pkill -f launch_qwen_train_l4_sequence.py`, and `gcloud ai custom-jobs cancel ...`
2. **Launch Multi-Region A100s:** Create a script to dispatch the 6 missing Qwen training jobs to 6 separate Google Cloud regions to bypass `RESOURCE_EXHAUSTED` errors in us-central1/us-east1. **NO L4s.**
3. **Update Orchestrator:** Modify `night_shift_orchestrator.py` to watch the jobs across those multiple regions.
4. **Resume Flow:** Let the orchestrator run generation on H100s, run Judges, and calculate Stats -> `CLEAN_RUN_LOG.md`.