# V2.24.0 documentation patch

`GAME_DEVELOPMENT.md` is included directly.

Run:

```bash
python3 -m automation.v240_apply_docs
```

to append idempotent V2.24.0 sections to:

- `ARCHITECTURE.md`
- `HIT_DETECTION_PLAN.md`
- `CURRENT_STATE.md`
- `ROADMAP.md`

The patch deliberately appends instead of replacing those files so newer local
project history is preserved.
