# V2.22.5 — Fast Proposal + Local Confirmation

## Why this version exists

Physical V2.22.4-r2 telemetry showed the runtime priorities are now mostly correct, but the perception path is still too slow for gameplay:

- normal audio acknowledgement was tens of milliseconds,
- the async CV worker took roughly 1.2–1.3 seconds on the measured shot,
- V2 candidate extraction alone accounted for roughly 0.7 seconds,
- the first global pass found a strong candidate, but the legacy tracking model then requested another full global pass to accumulate `hits`,
- the shot timed out before the second expensive pass returned.

The new invariant is:

> **A global search proposes. Local evidence confirms. Do not search the whole board again merely to increment persistence.**

## Runtime architecture

```text
PANG
  |
  +--> main/game thread: acknowledge, snapshot game state, keep rendering
  |
  +--> one async GLOBAL FAST proposal
          |
          +--> legacy proposal source
          +--> V2 maps / registration
          +--> sparse local maxima (no megapixel CC rescue in normal path)
          |
          v
       merged / cleanup candidates
          |
          v
       HitScanner tracks: hits=1
          |
          v
next real camera frame
          |
          +--> LOCAL PRE->POST confirm around existing candidate XY only
                 |
                 +--> confirmed -> same track hits=2/3 -> HIT
                 |
                 +--> none confirmed -> one FULL high-recall rescue
                                           |
                                           +--> local-confirm rescue result
                                           +--> otherwise MISS/timeout
```

## Fast extractor

The full research extractor remains untouched and callable. V2.22.5 only selects the fast extractor from the live `shot-cv-v2224` worker.

The fast path keeps:

- the existing recent PRE stack and shot model,
- registration,
- temporal noise normalisation,
- saliency maps,
- primary temporal evidence,
- a bounded independent temporal rescue,
- bounded peak refinement,
- spatial tile coverage,
- NMS,
- normal V2 candidate feature schema,
- known-hole soft penalty,
- V1/V2 merge and candidate bank.

The normal live path deliberately avoids the large connected-component rescue passes over the whole ROI. Sparse maxima are found with dilation + vectorised coordinate extraction and bounded top-K selection. The original extractor is the `FULL-RESCUE` path.

## Local confirmation

Local confirmation never changes the original global candidate coordinate. It asks only whether a compact physical PRE->POST change persists near that coordinate on a later timestamped camera frame.

Evidence recorded per confirmed candidate includes:

- centre absolute change,
- surrounding-ring change,
- compactness (`centre - ring`),
- local peak change,
- darkening evidence,
- best local offset for diagnostics only.

A re-hit/hole-in-hole is legal. If the PRE image already contains an old hole but the same area changes again after the new PANG, local confirmation may accept it.

## Rescue policy

One full high-recall rescue is permitted when:

- FAST returns zero proposals, or
- none of the first-pass candidates can be locally confirmed.

A rescue shot may temporarily receive a longer event timeout. Normal FAST-confirmed shots keep ordinary hit latency semantics.

## Concurrency / stale work

- no second full global job is submitted merely for persistence,
- terminal shots clear local state,
- queued-not-started CV work for terminal shots is cancelled where possible,
- already running OpenCV cannot be force-cancelled safely; its later result is discarded.

## Startup/calibration audio

Audio peaks while HitScanner is OFF/ARMING are consumed and not promoted into a gameplay shot later. The audio detector still runs; only gameplay shot creation is gated until the scanner is ACTIVE.

## What is not changed

- V2.22 ShotResolver semantics,
- perspective ROI / game-coordinate homography,
- V2.22.2 ridge cleanup,
- V2.22.4 async AI shadow,
- offline/F2 full extractor,
- AI authority modes,
- object-hit authority (still shadow/foundation).

## Acceptance sequence

1. selftest + install verify,
2. start AI Training in advisory, no F2,
3. one physical shot,
4. inspect `FAST-EXTRACT`, `LOCAL-CONFIRM`, `HIT`, `LATENCY`, `VISIBLE`,
5. if correct, 3 ordinary shots,
6. then two shots 0.7–1.0 s apart,
7. then one intentional near-old/re-hit test.

Do not tune AI weights from the first V2.22.5 test. First establish that the new runtime path preserves the real candidate and removes the repeated-global latency.
