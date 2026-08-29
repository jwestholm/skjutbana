# V2.22.5 physical test plan

## 0. Install

Unpack over the working repository that already contains V2.22.4-r2.

```bash
git status --short
git diff --check
python3 -m automation.v2225_selftest
python3 -m automation.v2225_verify_install
python3 -m automation.v2225_apply_docs
```

## 1. Startup gate

```bash
python3 main.py
```

Expected startup contains:

```text
[V2.22.1] ...
[V2.22.2] ...
[V2.22.3] ...
[V2.22.4-r2] ...
[V2.22.5] fast proposal + local confirmation installed
```

Calibration/startup noises must not create a gameplay `[V2.22.3 SHOT]` before HitScanner is ACTIVE.

## 2. First physical shot — stop after one

Use AI Training, advisory, no F2. Shoot one clear new hole.

Check:

- real hole is present among candidates,
- one `FAST-EXTRACT` line appears,
- first `V2.22.4 CV` is materially lower than the V2.22.4-r2 ~1.2 s reference if the extractor is the main bottleneck,
- `LOCAL-CONFIRM` appears on the next camera frame,
- normal path emits without a second full CV job,
- marker is on the physical hole.

Send the terminal block if any of these fail.

## 3. Three ordinary shots

If shot 1 is correct, shoot three separate holes. Record:

- main_ack,
- FAST extract ms,
- worker ms,
- local-confirm ms,
- confirmed count,
- HIT/VISIBLE latency,
- whether candidate XY is correct.

## 4. Two close PANGs

Shoot two distinct positions around 0.7–1.0 s apart.

Expected:

- both audio peaks are acknowledged quickly,
- each has its own shot id and PRE,
- no candidate/pre state crosses between shots,
- a worker result for a terminal old shot is discarded rather than emitted later.

## 5. Re-hit / near-old hole

Shoot very near or through a hole created earlier in the same session.

Expected:

- old unchanged holes should not satisfy local temporal confirmation merely because they look hole-like,
- a true fresh re-hit can confirm because PRE->POST changed at that location,
- no hard known-hole reject is introduced.

## 6. What to send back

Prefer the complete block containing:

```text
[V2.22.3 AUDIO-PRIORITY]
[V2.22.5 FAST-EXTRACT]
[V2.22.4 CV]
[V2.22.5 LOCAL-CONFIRM]
[V2.22.5 FULL-RESCUE]   # only if it occurred
[SHOT #]
[V2.22.3 LATENCY]
[V2.22.3 VISIBLE]
[V2.22.4 AI-SHADOW]
```

Also state whether the visible marker was on the physical hole.
