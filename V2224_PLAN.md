# V2.22.4 — Async Shot Pipeline

**Status:** implemented delta, 2026-08-29.

## Why this version exists

Physical V2.22.3 measurements separated the latency problem into three very different parts:

- audio peak -> main-thread acknowledgement was normally only about 8–34 ms;
- the current hybrid camera detector was commonly about 0.9–1.0 s;
- advisory/training AI work then delayed visible emission by roughly another second, although the V2.22 resolver core itself was only about 4–7 ms.

A second shot fired while the first shot was still being analysed reached the audio thread but waited about 575 ms before the main loop could acknowledge it.  That proved that prioritising a queued audio event is not enough while heavy perception still executes on the Pygame/main thread.

## V2.22.4 decision

Move expensive *work*, not authority, off the main thread.

```text
AUDIO THREAD
   PANG + timestamp
          |
          v
MAIN / GAME THREAD
   acknowledge immediately
   freeze shot/object context
   keep events + render alive
          |
          +----------------------------+
          |                            |
          v                            v
SINGLE CV WORKER                 AI SHADOW WORKER
current V1/V2 detector           observe/evidence/ranking
ROI + cleanup                    advisory resolver comparison
track candidate result           training/shadow bookkeeping
          |                            |
          v                            |
MAIN THREAD <---------------------------+
 integrate candidates
 confirm ordinary HitScanner track
 emit existing camera XY -> HitInput -> homography/game XY
```

## Non-negotiable safety rules

1. The global HitScanner remains gameplay authority in V2.22.4.
2. CandidateGeneratorV2 has mutable per-shot/bank state, so V2.22.4 uses **one CV worker**.  The purpose is main-thread responsiveness, not unsafe concurrent detector calls.
3. `off`, `train_only` and `advisory` AI modes are removed from the synchronous emission path.  They receive snapshots and run on one shadow worker.
4. AI authority modes (`blended`, `ai_priority`, `ai_only`) retain historical synchronous semantics for now.  Async authority needs a separately tested hard-deadline/fail-open protocol.
5. Pygame, scene callbacks, HitInput emission and final game-coordinate transformation remain on the main thread.
6. While a real physical shot is pending, ordinary scene simulation may remain frozen so the projected evidence is stable, but OS events and rendering continue.
7. Do not run the mutable CandidateGeneratorV2 engine from multiple CV threads.
8. A worker that has not returned yet is not a negative camera observation.  Track ageing is suppressed while the async detector verdict is pending.

## Detector profiling

V2.22.4 times the current detector without changing its algorithms.  Thread-local wrappers measure the V2 generator and key stages:

- `generate`
- `_collect_pre_frames`
- `_build_reference_and_noise`
- `_register_current`
- `_extract_candidates`
- `_update_candidate_bank`
- `_merge_hybrid`

Runtime log example:

```text
[V2.22.4 CV] shot=7 age=123ms queue=0.2ms worker=184.0ms v2=120.0ms other=64.0ms cand=153 pre=... ref=... reg=... extract=... bank=... merge=...
```

`other` contains time outside the profiled V2 `generate()` call, including legacy detector / ROI / V2.22.1+2 wrapper work.  This is enough to decide whether V2.22.5 should optimise legacy CV, V2 registration/reference work, candidate extraction/bank work, or introduce a true fast/rescue split.

## AI shadow semantics

For `off`, `train_only`, and `advisory`:

- bootstrap `observe_scanner()` becomes snapshot + queue only;
- bootstrap `choose_for_emission()` returns immediate camera passthrough on the main thread;
- bootstrap `mark_shot_finished()` records terminal state for the shadow worker;
- the worker runs the historical observation/evidence path and advisory chooser later for diagnostics/training.

Therefore advisory AI cannot delay an already selected camera hit.

## Object-hit direction

V2.22.3 object regions remain shadow-only.  V2.22.4 deliberately does not yet create a second direct per-object pixel detector.  First prove the global async pipeline and profile the real CV cost.  The object API remains the intended future ultra-fast gameplay lane: a scene can ask whether one of 1..100 frozen hit regions acquired fresh physical evidence at PANG, while global localisation continues in parallel for exact XY / misses / training.

## First physical gate

Before tuning accuracy again, verify:

- normal audio -> main acknowledgement remains p95 <50 ms;
- a second PANG can be acknowledged while shot 1 CV is still running;
- main thread stays visibly responsive while CV worker runs;
- advisory/train-only AI no longer adds about one second before the visible marker;
- CV logs reveal stage timings;
- final camera/game XY remains unchanged.

Do not start F2/night training until this runtime gate is understood.
