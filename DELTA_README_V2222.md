# Skjutbana V2.22.2 DELTA — Fast-path cleanup

Apply this ZIP **on top of the already installed V2.22.1 delta**.

V2.22.2 is based on the live findings from the first V2.22.1 physical shots:

1. perspective-aware playfield edge guard works — keep it unchanged;
2. the projected mouse cursor can become shot evidence;
3. static/old holes can still rank too highly;
4. live hit latency is still too high;
5. board/projector movement creates long horizontal candidate ridges.

## What changes

- `src/engine/camera/analysis_filters_v2222.py`
  - PRE-shot novelty cleanup;
  - stale registered-hole rejection with fresh re-hit preservation;
  - perspective-aware horizontal ridge cleanup in screen coordinates.
- `src/engine/camera/hit_scanner_v2222.py`
  - cursor hidden only while a shot is being detected;
  - pre-shot cursor position masked from CV;
  - cursor restored after HIT/MISS so manual AI-training labelling still works;
  - candidate cleanup before tracking / AI / resolver;
  - open-shot camera backlog bounded to 3 temporally spread frames;
  - extra timing and cleanup diagnostics.
- `src/engine/ai/runtime_v222.py`
  - V2.22.2 defaults.
- `src/engine/ai/bootstrap.py`
  - installs V2.22.2 after V2.22.1 and before AIRuntime wraps HitScanner.

No game/content coordinate semantics change. Candidates are still restored to full
camera XY before tracking/resolver and the existing HitInput homography remains
the only final camera->screen/game transform.

## Install

From repository root:

```bash
git switch dev
git status --short
```

Unpack this ZIP over the repository root, then:

```bash
git diff --check
python3 -m automation.v2222_selftest
python3 -m automation.v2221_selftest
python3 -m automation.offline_v222_selftest
python3 -m automation.runtime_v222_selftest
python3 -m automation.v2222_verify_install
```

Then follow `V2222_TEST_PLAN.md`.
