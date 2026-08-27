# AI hit detection — project snapshot

**Snapshot date:** 2026-08-27  
**Current research line:** V2.21 planning  
**Live/game authority:** unchanged; all AI work described here is offline/shadow unless explicitly stated otherwise.

This documentation bundle captures the current state of the camera/AI hit-detection work in `skjutbana`, the important experimental results from V2.12 through V2.20.2, the data/split rules that must not be broken, and the recommended next implementation step.

## Read these files in this order

1. `HIT_DETECTION_PLAN.md` — current source of truth and roadmap.
2. `docs/ai_hit_detection/CURRENT_STATE_2026-08-27.md` — what exists right now and what has been proven.
3. `docs/ai_hit_detection/RESULTS_SYNTHETIC_TO_PHYSICAL.md` — the latest decisive experiment.
4. `docs/ai_hit_detection/V2_21_PLAN.md` — next implementation plan.
5. `docs/ai_hit_detection/DATA_AND_EVALUATION_POLICY.md` — data semantics, splits and leakage rules.
6. `docs/ai_hit_detection/EXPERIMENT_LOG.md` — chronological results and decisions.
7. `docs/ai_hit_detection/RUNBOOK.md` — practical workflow and what not to run yet.
8. `docs/ai_hit_detection/KNOWN_ISSUES_AND_TRAPS.md` — pitfalls already discovered.

## One-sentence state

V2.20.2 can generate deterministic, visually plausible RGB training worlds and V2.18 learns them extremely well, including unseen generated seeds, but the same ranker collapses to 0% Top-1 on the existing physical camera session; the next work must therefore attack **physical-domain mismatch and the physical candidate-recall ceiling**, not spend more time optimizing synthetic-only accuracy.
