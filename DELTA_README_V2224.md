# Skjutbana V2.22.4 DELTA

Layer this ZIP over the checkout that already contains the tested V2.22.3 + V2.22.2-r2 chain.

## Main changes

- `main.py` installs V2.22.3 then V2.22.4.
- `src/engine/shot_async_v2224.py` moves the current physical detector to one background CV worker and keeps result integration / HIT emission on the main thread.
- Non-authoritative AIRuntime (`off`, `train_only`, `advisory`) is removed from the synchronous emission path and runs on a dedicated shadow worker.
- Pygame events/render continue while shot CV is running; scene simulation can remain frozen for stable projected evidence.
- Thread-local stage profiling exposes CandidateGeneratorV2 timing.
- Documentation updater contains both the V2.22.3 checkpoint and the new V2.22.4 checkpoint.

## Install

Extract over repository root, then:

```bash
git diff --check
python3 -m automation.v2224_selftest
python3 -m automation.v2224_verify_install
python3 -m automation.v2224_apply_docs
```

Read `V2224_TEST_PLAN.md` before physical testing.
