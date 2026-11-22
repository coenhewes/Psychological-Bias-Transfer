# IN_FLIGHT.md — live job tracking (ephemeral)

> Current job IDs + status only. Procedure/traps: `RUNBOOK.md`.
> Rewrite on every state change. Stale IDs are deleted, not archived here.

Last updated: 2026-07-16 13:58 UTC

## STANDING OBJECTIVE
Full clean end-to-end PBT run, iterated until valid + logged for the paper:
- MODELS: Llama-3.1-8B + Qwen2.5-7B
- SEEDS: 17, 42, 73, 88, 91 (5)
- JUDGES: Gemini 2.5 Flash (local) + GPT-OSS-20b (Vertex AI)
- Analysis: factorial ANOVA (model × corpus) + paired t-tests + kappa(Gemini vs gpt-oss)

## COMPLETED (Phase 1, 2, and 3.a)
- All adapters trained for Qwen2.5-7B and Llama-3.1-8B.
- Generations complete and synced locally + GCS (`gs://parallax-model-training-citric-snow-496311-f6/generations_fp/`).
- `run_e2e_pipeline.py` orchestrator written and running in background (`PID 1257371`).
- **Phase 3a (Local Judging)**: 100% COMPLETE. Local Gemini 2.5 Flash judging finished across all 20 valid `.jsonl` files (12,000 outputs) in `eval_outputs/judged_gemini/`. 

## IN FLIGHT (Phase 3b - Cloud Judging)
- **Vertex AI GPT-OSS open-weights judge**: 
  - **Job ID**: `pbt-judge-gptoss-20260716125808` (`4579428005125816320`)
  - **Status**: `JOB_STATE_RUNNING` on Vertex AI (us-central1, L4 nodes). 
  - **Metrics**: ~750-800 tokens/second. 
  - **Output**: Actively writing to GCS bucket (`generations_fp/`). 
  - The local master orchestrator (`run_e2e_pipeline.py`, PID 1257371) is blocking and safely waiting for this job to finish.

## NEXT (For New Session / Handoff)
1. **DO NOT LAUNCH NEW JOBS**. The pipeline is running correctly. Just wait for `pbt-judge-gptoss-20260716125808` to complete.
2. Monitor GCS via `gsutil ls gs://parallax-model-training-citric-snow-496311-f6/generations_fp/ | grep gptoss | grep "\.jsonl$"` (Should hit 20 total files).
3. Once the Vertex judge completes, `run_e2e_pipeline.py` will AUTOMATICALLY trigger Phase 4 (Statistical Analysis) locally in `.venv_judge`.
4. Verify Phase 4 outputs correctly populate the currently empty `results/` folder.
5. Review the final `CLEAN_RUN_LOG.md` produced by Phase 4 and commit it to the Obsidian Vault.

## CRITICAL TRAPS & PRECEDENTS
- **Stale Data**: We accidentally generated stats on 2026-07-11 data previously. It was moved to the Obsidian Vault at `archive/invalid_run_2026-07-16/`. ALWAYS check timestamps. Today is strictly 2026-07-16.
- **Vertex Job States**: NEVER trust `JOB_STATE_SUCCEEDED` from Vertex blindly (bash escaping errors swallowed failures previously). ONLY trust the GCS bucket manifest for file completion.
- **Credentials**: `~/.config/forge/gcp/` holds keys. SSH and Borgmatic passphrases have been rotated. Local Gemini API key is dynamically loaded from `~/.hermes/.env`. 
- **Retry Logic**: All `gsutil` calls now have a mandatory 5-retry loop.
