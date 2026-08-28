# V2.22 — Fast Shot Resolver foundation

## Status

V2.22 is a **live-safe resolver foundation** on top of the existing detector/AI runtime.
It does **not** replace `HitScanner`, `SimpleAIMemory`, F1/F2 training or V2.21.5.
It adds the decision layer needed to fuse them without adding a large synchronous model to the game loop.

V2.22 is deliberately **advisory-first**. The normal `train_only` path is unchanged.

## Why V2.22 exists

V2.21.5 showed that the broad physical full-frame pool can retain the true hit extremely well on the current 30 full-frame physical shots, while the learned ranker still generalises poorly on protected splits. The next engineering problem is therefore not only proposal generation; it is fast evidence fusion and final candidate selection.

The desired live architecture is:

```text
Audio peak / shot_id
        |
        +--> existing HitScanner candidates + tracking
        +--> existing SimpleAIMemory ranking
        +--> persistence / known-hole evidence
        +--> future physical-image expert (parallel worker)
        +--> optional frozen game-context snapshot
                         |
                         v
                 ShotResolverV222
                         |
                 ONE REAL CANDIDATE XY
                         |
                      HitInput
                         |
                   game_x / game_y
```

The game still consumes the existing hit-event coordinate chain. Games do not need to know which expert produced the winning camera coordinate.

## Core safety rules

1. **No coordinate interpolation.** If camera candidate A and AI candidate B disagree, V2.22 chooses A or B (or another real candidate cluster). It never emits a point halfway between two holes.
2. **Fail open.** Resolver/runtime failures preserve the detector result.
3. **`train_only` remains legacy behaviour.** Resolver work is skipped in the normal training mode.
4. **`advisory` has zero authority.** It calculates/logs a decision but cannot move the emitted hit.
5. **Game context is a weak prior only.** A target/hotspot can break a close tie; it must never move a physically strong shot onto a game target.
6. **Confidence is not yet a probability.** `confidence=0.98` is an uncalibrated resolver confidence until a later held-out calibration step exists.
7. **Heavy physical vision must not run synchronously in `choose_for_emission()`.** Heavy work starts in parallel and publishes a compact vote list.

## New files

- `src/engine/ai/shot_resolver_v222.py` — pure-Python bounded resolver and spatial clustering.
- `src/engine/ai/runtime_v222.py` — V2.22 monkey patch/API for existing `AIRuntime`.
- `automation/offline_v222_selftest.py` — deterministic resolver and latency selftest.
- `automation/runtime_v222_selftest.py` — isolated AIRuntime patch/authority semantics test.
- `automation/v222_verify_install.py` — verifies installation in the real checkout.
- `automation/v222_physical_probe.py` — read-only probe of the installed V2.21.5 physical module/model.

`src/engine/ai/bootstrap.py` installs the V2.22 runtime patch before the regular AIRuntime singleton is normally created.

## Resolver inputs

### Existing camera candidates

The resolver receives the candidates already owned by the current shot context. Cheap evidence is enriched with existing runtime functions where available:

- detector score,
- SimpleAI score/rank,
- persistence,
- `existed_before_shot`,
- source support,
- the detector's emitted track as a strong baseline observation.

### External expert votes

A future physical model or other expert publishes **already-computed** votes:

```python
from src.engine.ai.runtime import get_ai_runtime

runtime = get_ai_runtime()
runtime.publish_resolver_votes(
    shot_id,
    "physical_dense",
    [
        {"camera_x": 1311.0, "camera_y": 772.0, "score": 0.97},
        {"camera_x": 1240.0, "camera_y": 810.0, "score": 0.61},
    ],
    weight=1.0,
)
```

The API intentionally accepts short candidate lists instead of images. The image expert may therefore run in a worker/thread/process while the ordinary detector gathers post-shot frames.

### Shot-start hook for a future parallel physical expert

```python
runtime.register_shot_start_hook("physical_dense", callback)
```

The callback is invoked when the AI shot context is created. **The callback itself must be non-blocking.** Its job is to enqueue work and return immediately. The worker later calls `publish_resolver_votes(...)` with the same `shot_id`.

This is the integration point intended for the V2.21.x/V2.22 physical ranker after its live latency and API have been verified.

### Game-context snapshot

A game may register a fast provider:

```python
runtime.set_game_context_provider(my_provider)
```

The provider is sampled once for the shot context, close to the audio-peak time, so moving targets are not evaluated at a later world state.

Expected camera-space shape:

```python
{
    "targets": [
        {"camera_x": 1311.0, "camera_y": 772.0, "radius_px": 45.0, "score": 1.0}
    ]
}
```

`priors`, `hotspots` and `targets` are accepted. V2.22 caps this at a weak resolver contribution (default 0.06).

## Authority modes in V2.22

The existing setting names are retained:

| mode | V2.22 behaviour |
|---|---|
| `off` | legacy path |
| `train_only` | legacy path; resolver skipped |
| `advisory` | resolver runs, never overrides detector |
| `blended` | **semantic change:** `trust_percent` scales AI evidence; selected XY is always a discrete candidate |
| `ai_priority` | discrete override only at/above `override_confidence` |
| `ai_only` | discrete resolver candidate at/above `min_confidence` |

Important: old `blended` coordinate interpolation is intentionally removed once V2.22 resolver authority is active.

## New settings

The runtime supplies defaults without requiring a committed `content/ai/settings.json` change:

```json
{
  "resolver_v222_enabled": true,
  "resolver_v222_log": false,
  "resolver_v222_cluster_radius_px": 18.0,
  "resolver_v222_max_external_votes": 96,
  "resolver_v222_game_prior_weight": 0.06,
  "resolver_v222_latency_history": 512
}
```

Existing `mode`, `trust_percent`, `min_confidence` and `override_confidence` remain authoritative.

## Latency contract

The resolver is intentionally pure Python and bounded. It does no OpenCV/full-frame image processing.

Engineering targets on the range PC:

- ShotResolver p95: **< 10 ms**
- audio peak -> final emitted HitEvent p50: **< 250 ms**
- audio peak -> final emitted HitEvent p95: **< 500 ms**
- normal upper bound: **< 1 second**

`AIRuntime.resolver_status()` exposes rolling p50/p95/p99 for resolver and end-to-end latency.

For live logging, set:

```json
"resolver_v222_log": true
```

Example line:

```text
[V2.22 RESOLVER] shot=42 mode=advisory apply=False xy=(1311.0,772.0) conf=0.83 score=0.79 margin=0.18 clusters=7 resolver=2.8ms e2e=241.5ms
```

## Relationship to V2.21.5

V2.22 does **not** make the current V2.21.5 ranker authoritative.

The current physical findings remain valuable:

- the broad dense pool is a high-recall expert/proposal source,
- the current frozen learned ranker is not sufficiently general on protected shots,
- full-frame image work is much heavier than the final resolver and must be benchmarked before live use.

The next physical-expert step is to expose a fast inference function that produces a short scored list for one shot, launch it at shot start, and publish it to `AIRuntime.publish_resolver_votes()` before detector emission when possible.

## V2.22 success criteria

V2.22 itself is an architecture/latency/safety milestone, not the >95% accuracy claim.

It passes when:

1. all selftests pass,
2. existing V2.21.5 selftest/benchmark remain unchanged,
3. advisory mode never changes emitted hit XY,
4. live resolver p95 is comfortably below 10 ms,
5. end-to-end p95 remains below 500 ms during controlled shots,
6. discrete selection is verified (no midpoint/interpolated holes),
7. fail-open behaviour is verified.

After that, wire the physical expert and start session-diverse overnight data/training runs.
