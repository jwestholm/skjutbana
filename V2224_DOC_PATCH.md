# V2.22.4 documentation checkpoint

This delta's `automation/v2224_apply_docs.py` appends idempotent V2.22.3 and V2.22.4 sections to the three repository source-of-truth documents.  The V2.22.3 section is included again so the documentation checkpoint travels with this ZIP even if the earlier docs updater was not run.

Files updated when present:

- `CURRENT_STATE.md`
- `HIT_DETECTION_PLAN.md`
- `AI_CONTEXT.md`

Core V2.22.4 additions:

- heavy physical CV is worker-owned, not Pygame/main-thread work;
- one CV worker only because CandidateGeneratorV2 has mutable state;
- off/train_only/advisory AI is off the emission-critical path;
- authority modes remain synchronous until a hard-deadline protocol is validated;
- render/events remain live while physical CV runs;
- scene simulation may stay frozen during a shot for stable projector evidence;
- object-first direct pixel detection remains a planned complementary fast path, not authority yet;
- detector-stage profiling now decides the next optimisation rather than threshold guessing.
