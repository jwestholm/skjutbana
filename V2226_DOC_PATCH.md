# V2.22.6 documentation patch

Apply with:

```bash
python3 -m automation.v2226_apply_docs
```

The updater appends idempotent V2.22.6 sections to:

- `CURRENT_STATE.md`
- `HIT_DETECTION_PLAN.md`
- `AI_CONTEXT.md`

Key architectural additions:

- `HoleTrack.hits` is frame-unique temporal evidence.
- same-frame candidate agreement is separate evidence, not persistence.
- Local Confirm is the intended later-frame persistence step.
- strong rejected audio transients expose exact reject gates before audio thresholds are retuned.
