# V2.25.0 Documentation Patch

`automation.v250_apply_docs` is append-only and idempotent. It first applies
prior V2.24 documentation patches, then appends the V2.25 marker to living
repository documents when they exist.

Documents covered:

- `ARCHITECTURE.md`
- `HIT_DETECTION_PLAN.md`
- `CURRENT_STATE.md`
- `ROADMAP.md`
- `GAME_DEVELOPMENT.md`
- `AI_CONTEXT.md`

Standalone source-of-truth documents included in the delta:

- `V250_GAME_OBJECT_SYSTEM_PLAN.md`
- `GAME_OBJECT_SYSTEM.md`
- `AI_GAME_OBJECTS.md`
- `V250_TEST_PLAN.md`

The patch deliberately preserves existing repository text and history.
