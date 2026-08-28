# V2.22.1 test plan

Run tests in this order. Do not start with a large F2/night run.

## Phase 0 — install / source verification

From repository root on `dev`:

```bash
git status --short
git rev-parse --short HEAD
```

Unpack the delta over the repository root, then:

```bash
git status --short
git diff --check
git diff -- src/engine/ai/bootstrap.py src/engine/ai/runtime_v222.py automation/v222_physical_probe.py
```

Expected bootstrap change: V2.22.1 ROI installer is called before the existing AI HitScanner wrapper, and the `ai_results` scene mapping exists.

## Phase 1 — no hardware required

```bash
python3 -m automation.v2221_selftest
python3 -m automation.offline_v222_selftest
python3 -m automation.runtime_v222_selftest
python3 -m automation.v2221_verify_install
python3 -m automation.v222_physical_probe
```

Expected:

- all V2.22.1 geometry tests pass;
- all previous V2.22 resolver/runtime tests still pass;
- install verifier reports AI Results mapping present;
- physical probe ends with `[PASS] V2.21.5 frozen model loaded through official loader` instead of the old `allow_pickle=False` warning.

## Phase 2 — AI Results crash regression

Start normally:

```bash
python3 main.py
```

Before shooting:

1. Open the AI menu.
2. Open the AI Training Results / graph scene that previously crashed.
3. Move through the available graph/result views.
4. Return to the menu.

Expected: no `Unknown item type: ai_results`, no traceback, no SIGABRT.

If a *different* graph-specific traceback appears, save the entire traceback. That means the factory regression is fixed and a result-schema compatibility issue remains.

## Phase 3 — geometry visual sanity check

Keep AI in `advisory` and resolver logging enabled.

Enter the same AI Training scene used for the previous manual physical-shot test. Do **not** press F2 yet.

On startup / first shot, terminal diagnostics must say:

```
[V2.22.1 ROI] ... mode=homography ...
```

If it says `scanport_fallback` or `full_frame_fallback`, stop the live comparison and send the log; calibrated perspective ROI is not active.

The `crop=` number is the percentage of the full camera frame still processed by the heavy detector. With the described camera view it is reasonable to expect it to be clearly below 100%, but the exact percentage depends on the calibrated quadrilateral and its bounding box.

## Phase 4 — five physical shots, no F2

Shoot only five manual physical shots first:

1. centre;
2. upper-left, comfortably inside the playable area;
3. upper-right, comfortably inside;
4. lower-left, comfortably inside;
5. lower-right, comfortably inside.

Do not aim exactly on the outer playfield line; the deliberate 12-screen-pixel guard treats that strip as non-playable.

For every shot verify:

- the visible hit marker lands where the physical shot went;
- there is no systematic offset caused by the crop;
- the shot ID agrees across `[SHOT #N]`, `[V2.22.1 ROI]` and `[V2.22 RESOLVER]` as far as overlapping timing permits;
- advisory resolver still says `apply=False`;
- terminal contains detector and resolver latency.

Copy the terminal block for all five shots.

## Phase 5 — edge-artifact check

Look at the same candidate/debug display where the large cluster around `y≈0` was visible previously.

Expected: candidates on the exact calibrated top/bottom/left/right playfield boundary are strongly reduced or absent.

Do not tune the guard before collecting this first comparison. Default is 12 screen pixels.

## What to send back

Send:

1. output of `automation.v2221_selftest` and `automation.v2221_verify_install`;
2. confirmation that AI Results opens (or full new traceback);
3. the five-shot terminal block containing `[V2.22.1 ROI]`, `[SHOT #N]`, and `[V2.22 RESOLVER]` lines;
4. whether every rendered hit marker matched the physical hole;
5. whether the top-edge candidate ridge disappeared/reduced;
6. optionally a screenshot of the candidate/debug view after one shot.

The next decision is based on measured `detector=...ms` and `e2e=...ms`, not on guesswork. If crop-only detection is still too slow, the next optimization target will be the camera-frame ingestion/history path or frame-resolution/pyramid strategy rather than the resolver.
