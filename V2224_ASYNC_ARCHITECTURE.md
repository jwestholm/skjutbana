# V2.22.4 async architecture notes

## Thread ownership

### Audio reader thread
Producer only.  Detects/timestamps PANG and queues the normal AudioPeakEvent.  No Pygame or game-object calls.

### Main/Pygame thread
Owns:

- audio dispatch to subscribers;
- shot/object snapshot orchestration;
- camera ring pickup bookkeeping;
- integration of finished CV-worker candidate lists into HitScanner tracks;
- ordinary HitScanner HIT/MISS resolution;
- HitInput emission and camera -> screen/game homography path;
- Pygame events, rendering and scene callbacks.

It must not execute the expensive physical candidate detector in the normal V2.22.4 path.

### CV worker
One worker.  Receives an immutable grayscale frame plus a bounded snapshot/clone of scanner state.  Runs the already installed detector stack (including V2.22.1 ROI and V2.22.2 cleanup) and returns candidates/debug/timing.  It never emits a game hit itself.

One worker is intentional: CandidateGeneratorV2 carries mutable per-shot models/persistence/candidate-bank state.

### AI shadow worker
One worker for non-authoritative modes.  Receives copied scanner/event/candidate/frame context.  Runs the historical AIRuntime observation/evidence path and advisory comparison after gameplay has been allowed to pass through.

## Result integration

If no CV result is ready, `_detect_frame_candidates()` returns an empty temporary list **with async-waiting state**.  `_update_tracks()` recognises that state and does not treat the lack of a result as a physical negative frame.

When one or more sequential worker results are ready:

- older completed results are integrated into tracks in timestamp order;
- the newest result is returned through the ordinary HitScanner update call;
- track confirmation/HIT/MISS logic therefore stays in existing HitScanner code on the main thread.

## Backpressure

Default CV submission is bounded to at most one outstanding frame per shot and at least ~55 ms between submitted frame timestamps.  This avoids turning a 30-fps camera into an unbounded queue behind a slower detector.

The settings are deliberately exposed:

```text
async_detector_enabled_v2224
async_detector_max_queued_per_shot_v2224
async_detector_frame_spacing_ms_v2224
async_detector_log_v2224
async_ai_shadow_enabled_v2224
async_ai_shadow_log_v2224
```

## Fail-open

Disabling `async_detector_enabled_v2224` restores the captured synchronous detector call.

Authority AI modes also retain synchronous historical semantics.  V2.22.4 does not silently grant asynchronous AI authority.

## Frame pacing

Only an unacknowledged PANG bypasses normal FPS waiting. While CV is running asynchronously, rendering stays at normal game FPS instead of spinning uncapped and stealing CPU from the detector worker.
