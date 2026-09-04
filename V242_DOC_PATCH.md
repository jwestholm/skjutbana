# V2.24.2 documentation patch

`python3 -m automation.v242_prepare` runs the idempotent documentation patch and
menu installer. The documentation patch appends V2.24.2 sections to:

- `ARCHITECTURE.md`
- `HIT_DETECTION_PLAN.md`
- `CURRENT_STATE.md`
- `ROADMAP.md`

It first invokes the V2.24.1/V2.24.0 documentation chain so the checkpoint is
self-contained when earlier optional patch commands were skipped.
