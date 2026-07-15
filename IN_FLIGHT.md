# IN_FLIGHT.md — live job tracking (ephemeral)

> Current job IDs + status only. Procedure/traps: `results/pipeline_validated_procedure.md`.
> Rewrite on every state change. Stale IDs are deleted, not archived here.

Last updated: 2026-07-15 21:50 UTC

## STANDING OBJECTIVE
Full clean end-to-end PBT run, iterated until valid + logged for the paper:
- MODELS: Llama-3.1-8B + Qwen2.5-7B
- SEEDS: 17, 42, 73, 88, 91 (5)
- JUDGES: Gemini (local, DONE) + gpt-oss-20b (Vertex, DONE)
- Analysis: factorial ANOVA (model × corpus) + paired t-tests + kappa(Gemini vs gpt-oss)

## COMPLETED
- All adapters trained for Qwen2.5-7B.
- Qwen2.5-7B Generations completed and in bucket (`generations_fp/`):
  - Control: 17, 42, 73, 88, 91
  - Treatment: 42, 73, 88
- Orchestrator `run_e2e_pipeline.py` created for sequential end-to-end management (currently paused while manual generation concludes).

## IN FLIGHT (verify live before acting)
- **Qwen2.5-7B Generation (L4, us-central1)**:
  - Missing: `treatment 17`, `treatment 91`.
  - Currently in queue: `pbt-eval-fp-qwen2_5-7b-treatment-seed91-20260715214108` (JOB_STATE_PENDING).
  - Background loop `launch_qwen_missing_l4.py` is actively monitoring and managing retries.

## NEXT (For New Session)
1. Ensure `treatment 17` is added back to the generation queue (currently omitted).
2. Wait for `treatment 17` and `treatment 91` `.jsonl` files to successfully land in the `gs://.../generations_fp/` bucket.
3. Once all 10 Qwen JSONL files are present, execute Phase 3 (Judging) via `run_e2e_pipeline.py` or manually.
4. Execute Phase 4 (Analysis) to produce the final `CLEAN_RUN_LOG.md`.
