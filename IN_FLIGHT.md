# IN_FLIGHT.md — live job tracking (ephemeral)

> Current job IDs + status only. Procedure/traps: `RUNBOOK.md`.
> Rewrite on every state change. Stale IDs are deleted, not archived here.

Last updated: 2026-07-16 14:35 UTC

## STANDING OBJECTIVE
Full clean end-to-end PBT run, iterated until valid + logged for the paper:
- MODELS: Llama-3.1-8B + Qwen2.5-7B
- SEEDS: 17, 42, 73, 88, 91 (5)
- JUDGES: Gemini 2.5 Flash (local) + GPT-OSS-20b (Vertex AI)
- Analysis: factorial ANOVA (model × corpus) + paired t-tests + kappa(Gemini vs gpt-oss)

## COMPLETED (E2E Run & Statistical Analysis)
- All adapters trained for Qwen2.5-7B and Llama-3.1-8B.
- Generations complete and synced.
- **Phase 3**: Both local Gemini (via API) and Vertex AI GPT-OSS judging successfully completed across all 20 valid `.jsonl` files (12,000 outputs each).
- **Phase 4**: Statistical analysis finished. All CSVs (ANOVA, t-tests, power analysis) and mixed-model text outputs written to `/results/`. 
- **Peer Review Gate**: Dual LLM-as-judge consensus failed validation ($\kappa = 0.09$). LLM scoring cannot be published as ground truth without human validation.

## IN FLIGHT (Human Validation)
- **Task**: Solo-author Intra-Rater Reliability Validation.
  - **Run 1 (Core)**: The author must manually annotate `data/human_annotation_run1_500.csv` based on the strict operational definitions in `data/ANNOTATION_GUIDE.md`.
  - **Washout**: A mandatory 72-hour washout period begins after Run 1 is finished.
  - **Run 2 (Holdout)**: After 72 hours, the author must manually annotate `data/human_annotation_run2_150.csv` (a scrambled subset of Run 1).
- **Goal**: Calculate Intra-Rater Kappa between Run 1 and Run 2 to prove measurement stability ($\kappa > 0.80$), then calculate F1-score against the LLM judges to validate which LLM serves as the publishable proxy.

## NEXT (For Next Session)
1. Do **not** modify the evaluation code or statistics. Wait for the author to finish Run 1 and Run 2.
2. Once the author confirms both CSVs are fully labeled (0s and 1s), run the scripts to calculate Intra-Rater Kappa and LLM-to-Human F1 scores.
3. Patch the final paper draft (`Cognitive_Distortion_Transfer_Final_Draft.md`) to include the human validation statistics.
4. Execute `scripts/build_release_artifacts.py` to package the raw data for arXiv publication.

## CRITICAL TRAPS & PRECEDENTS
- **Stale Data**: We accidentally generated stats on 2026-07-11 data previously. It was moved to the Obsidian Vault at `archive/invalid_run_2026-07-16/`. ALWAYS check timestamps. Today is strictly 2026-07-16.
- **Vertex Job States**: Vertex `gpt-oss` judge hung in `JOB_STATE_RUNNING` despite GCS being fully populated. We unblocked the local pipeline via local state-injection rather than indiscriminately killing jobs. NEVER blindly trust `JOB_STATE_RUNNING` when the GCS manifest is full.
- **Credentials**: `~/.config/forge/gcp/` holds keys. SSH and Borgmatic passphrases have been rotated. Local Gemini API key is dynamically loaded from `~/.hermes/.env`. 
- **Retry Logic**: All `gsutil` calls now have a mandatory 5-retry loop.