# V2.22 test and verification plan

Run all commands from the repository root on the **dev** branch.

## 0. Before applying the delta

```bash
git switch dev
git pull --ff-only
git status --short
git rev-parse --short HEAD
```

The working tree should be clean before copying the delta over the repository root.

The V2.22 work was designed against the V2.21.5 dev state discussed/tested immediately before this delta. Because the build sandbox could read GitHub source but could not perform a network `git clone`, always inspect the final `git diff` after overlaying the ZIP.

## 1. Copy the delta

Unpack the ZIP into the repository root so paths merge with `src/`, `automation/`, etc.

Then:

```bash
git status --short
git diff --check
git diff -- src/engine/ai/bootstrap.py
```

Expected bootstrap change: installation of `runtime_v222` before the existing menu/scene/HitScanner patches. Existing HitScanner patch behaviour must still be present.

## 2. Static/import verification

```bash
python3 -m automation.v222_verify_install
```

Expected:

```text
[PASS] required delta files exist
[PASS] AIRuntime has V2.22 resolver patch
...
Install verification passed.
```

## 3. Resolver selftest

```bash
python3 -m automation.offline_v222_selftest
```

Must pass all checks, including:

- camera Top-2 can win with independent expert agreement,
- emitted coordinate is a real candidate,
- old holes are penalised,
- game context is only a weak prior,
- confidence is marked uncalibrated,
- bounded resolver latency selftest.

The synthetic/local p95 number is a software regression metric. The range PC live p95 is the important latency measurement.

## 4. Runtime integration selftest

```bash
python3 -m automation.runtime_v222_selftest
```

Must pass:

- patch installation,
- shot-time game snapshot,
- external vote fusion,
- advisory no-authority behaviour,
- discrete `blended` semantics (no XY midpoint),
- latency telemetry.

## 5. V2.21.5 regression

V2.22 should not change the offline V2.21.5 physical implementation.

Run:

```bash
python3 -m automation.offline_v2215_selftest
python3 -m automation.physical_dense_v2215_benchmark \
  --root content/ai/candidate_shadow_v216
```

Do **not** retrain just to verify V2.22 if the existing frozen V2.21.5 model is still present.

The previously observed reference includes:

- dense pool ALL @20 = 1.0000,
- dense pool ALL @42 = 1.0000,
- learned_512 ALL @20 = 0.4333,
- learned_1024 ALL @20 = 0.7000,
- `eligible_for_live_authority = False`.

Small formatting/runtime differences are okay; proposal/ranking metrics should not change because V2.22 does not modify those files.

## 6. Probe the physical V2.21.5 API/model

```bash
python3 -m automation.v222_physical_probe
```

This does not change data or model files. Save/send the output. It tells us the exact installed public functions and NPZ keys needed for the next parallel physical-expert bridge without guessing the V2.21.5 API.

## 7. Live shadow test — no authority

Use the existing AI settings UI to set:

```text
mode = advisory
```

For useful terminal telemetry, add/update the local gitignored `content/ai/settings.json` entry:

```json
"resolver_v222_log": true
```

Do **not** switch to `ai_priority` yet.

Shoot a controlled series of at least 20 real shots. Prefer a simple known target first.

Verify:

1. normal hits still arrive in the game exactly as before,
2. every resolver line says `mode=advisory apply=False`,
3. there are no crashes/exceptions from AI bootstrap/runtime,
4. resolver p95 stays below 10 ms,
5. end-to-end values are normally below 500 ms and never routinely exceed 1 s,
6. resolver XY can differ from camera XY in the log, but gameplay XY must not change in advisory mode.

If the application exposes its runtime singleton in a debug shell, `get_ai_runtime().resolver_status()` returns rolling latency/status; otherwise the per-shot terminal log is sufficient for this test.

## 8. Fail-open test

Temporarily set in the local AI settings:

```json
"resolver_v222_enabled": false
```

Run a few test shots. Behaviour should revert to the legacy AIRuntime path.

Then restore it to `true`.

## 9. Authority test — only after steps 1–8 pass

Start conservatively with:

```text
mode = ai_priority
override_confidence = 0.92
```

At this stage V2.22 only fuses the existing live evidence unless an external expert has been registered, so this is an integration smoke test, not the >95% experiment.

Shoot 10–20 controlled shots while visually checking the actual hole and candidate overlay. Abort authority testing if any resolver override is clearly worse than the detector baseline.

## 10. What to send back

Please send:

- output from `offline_v222_selftest`,
- output from `runtime_v222_selftest`,
- output from `v222_verify_install`,
- output from `v222_physical_probe`,
- 20+ `[V2.22 RESOLVER]` advisory lines,
- any traceback,
- optional `git diff --stat` and bootstrap diff if it differs from the expected minimal integration.

That is enough to build the next step: parallel V2.21.x physical expert + overnight data/training path.
