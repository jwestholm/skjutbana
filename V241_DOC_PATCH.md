# V2.24.1 documentation patch

The delta ships `GAME_DEVELOPMENT.md` directly. The repository-wide living docs
are updated append-only by `automation/v241_apply_docs.py` so their existing
history is never overwritten by a generated delta.

Marker: `<!-- V2.24.1 OBJECT_LOCAL_PHYSICAL_SEARCH -->`

Files patched when present:

- `ARCHITECTURE.md`
- `HIT_DETECTION_PLAN.md`
- `CURRENT_STATE.md`
- `ROADMAP.md`

`ROADMAP.md` is created if it is missing. The patch is idempotent.
