# V2.14 Selftest

Run from the repository root:

```bash
python3 -m automation.hole_v214_selftest
```

The selftest verifies:

- local physical feature maps are substantially invariant to a large brightness shift,
- forced background/domain randomization materially changes the candidate crop,
- real `hole_*` assets remain holdout-only,
- strict novel backgrounds remain outside train/model selection,
- clean + procedural-domain validation drives epoch selection,
- a toy held-out-background learning problem works,
- V2.14 model save/load preserves configuration.

Expected final line:

```text
All V2.14 selftests passed.
```
