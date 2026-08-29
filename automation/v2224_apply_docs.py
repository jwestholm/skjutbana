from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CURRENT_STATE = r'''

<!-- V2224_CURRENT_STATE -->
---
## V2.22.4 async shot pipeline — 2026-08-29

- V2.22.3 physical telemetry proved that audio dispatch is normally fast (single/tens of milliseconds) but the heavy hybrid physical detector can occupy roughly a second and non-authoritative AI observation can add a second synchronous delay before a marker becomes visible.
- V2.22.4 keeps audio acknowledgement and all Pygame/game emission on the main thread but moves the installed physical candidate detector to one bounded background CV worker. One worker is intentional because CandidateGeneratorV2 has mutable per-shot/persistence/bank state.
- The main thread continues processing OS/Pygame events and rendering while CV runs. Ordinary scene simulation may remain frozen while a physical shot is open so the projected camera evidence stays stable.
- `off`, `train_only`, and `advisory` AI modes no longer block HitScanner emission: observation/evidence and advisory comparison run on a separate shadow worker, while `choose_for_emission()` is immediate camera passthrough on the gameplay thread. AI authority modes remain synchronous until a tested hard-deadline/fail-open authority protocol exists.
- Detector stage profiling records V2 generate/pre/reference/registration/extraction/bank/merge times plus non-V2 wrapper/legacy time. Use these measurements to choose the next fast/rescue optimisation instead of tuning blind.
- V2.22.3 object-hit regions remain shadow-only. A future direct per-object PRE->POST fast path is still planned as a complementary gameplay lane; global exact XY remains necessary for free targets, misses that matter, calibration, damage decals and AI Training.
'''

HIT_PLAN = r'''

<!-- V2224_HIT_DETECTION_PLAN -->
---
## V2.22.4 — asynchronous live perception bridge

V2.22.4 separates **main-thread responsiveness** from **detector throughput**. A queued PANG is still the highest-priority runtime event, but the expensive installed physical detector no longer executes on the Pygame/main thread.

```text
PANG -> main acknowledge / freeze shot context
          |
          +--> single CV worker: current ROI + V1/V2 + cleanup -> candidates
          |
          +--> main keeps events/render alive
          |
          +--> main integrates worker candidates into normal HitScanner tracks
          |      -> ordinary camera XY -> HitInput -> homography/game XY
          |
          +--> non-authoritative AI shadow worker (off/train_only/advisory)
                 observation/evidence/ranking/resolver diagnostics after passthrough
```

### Concurrency rules

1. CandidateGeneratorV2 is mutable and must have one live owner. V2.22.4 uses one CV worker; do not parallel-call the same generator merely to reduce queue latency.
2. A pending worker verdict is not a negative camera frame. Track ageing must not occur only because CV is busy.
3. Pygame, HitInput, final hit emission and game-object callbacks stay on the main thread.
4. Keep the CV queue bounded. If a second PANG is acknowledged quickly but waits in the one-worker CV queue, optimise detector throughput / add a measured fast-rescue lane rather than reintroducing main-thread blocking.
5. AI authority modes remain synchronous in V2.22.4. Async authority requires a hard deadline and fail-open camera result tested on unseen physical sessions.

### Profiling decision gate

Every physical run should report worker time and V2 sub-stage timing. Compare V2 `generate` with `other` (legacy detector + ROI/cleanup/wrapper time), then inspect PRE/reference/registration/extraction/bank/merge. The next version should attack the measured dominant stage or introduce a fast path that can return a confident result without running the full rescue/research pipeline.

### Object-first continuation

The object API remains complementary to global localisation. For object-heavy games, future code may batch-test 1..100 frozen hit regions for fresh physical change and let a high-confidence object hit react before a global miss coordinate is known. This path must still preserve object-local XY and physical evidence; game context cannot invent a hit.
'''

AI_CONTEXT = r'''

<!-- V2224_AI_CONTEXT -->
---
## V2.22.4 async runtime semantics

### CV worker

The expensive installed detector stack runs on a single background worker using an immutable grayscale frame and copied scanner context. The worker returns candidates/debug/timing only. It never emits HitInput or calls game/Pygame code.

The worker count is intentionally one because CandidateGeneratorV2 contains mutable per-shot models, persistence and candidate-bank state. Main-thread responsiveness and safe overlap are the objective; concurrent mutation is not.

### Main-thread integration

Finished candidate results are integrated in timestamp order into the ordinary HitScanner tracks on the main thread. If the worker has not returned yet, the temporary absence of candidates is not treated as a negative observation. Existing track confirmation, HIT/MISS, full-camera XY and homography/game-coordinate semantics remain authoritative.

### Non-authoritative AI

For `off`, `train_only`, and `advisory`, AIRuntime observation/evidence is snapshot-and-queue. `choose_for_emission()` on the main thread is immediate passthrough and `mark_shot_finished()` is deferred to the AI shadow worker. Advisory comparison/resolver diagnostics run later. Therefore non-authoritative AI must never delay an already selected camera hit.

`blended`, `ai_priority`, and `ai_only` keep historical synchronous semantics until a bounded-deadline async authority protocol has physical holdout evidence.

### Rendering vs simulation

While a real physical shot is open, Pygame events and rendering continue. Scene simulation may be frozen so moving projected content does not change the visual evidence while the camera worker analyses the shot-time frame. Shot/object context remains frozen at PANG.

### Next optimisation

Use V2.22.4 stage telemetry to decide between legacy-detector optimisation, V2 registration/reference optimisation, candidate extraction/bank optimisation, or a true fast/rescue split. Do not infer the bottleneck from candidate count alone.
'''

SECTIONS = {
    "CURRENT_STATE.md": CURRENT_STATE,
    "HIT_DETECTION_PLAN.md": HIT_PLAN,
    "AI_CONTEXT.md": AI_CONTEXT,
}


def apply_one(path: Path, section: str) -> str:
    marker = next((ln.strip() for ln in section.splitlines() if ln.strip().startswith("<!-- V2224_")), "")
    if not path.exists():
        return f"[WARN] missing {path.name}; not created"
    text = path.read_text(encoding="utf-8")
    if marker and marker in text:
        return f"[OK] {path.name}: V2.22.4 section already present"
    path.write_text(text.rstrip() + section + "\n", encoding="utf-8")
    return f"[UPDATED] {path.name}: V2.22.4"


def main() -> None:
    print("V2.22.3 + V2.22.4 DOCUMENTATION UPDATE")
    print("========================================")
    # Carry yesterday's requested documentation checkpoint in this ZIP too.
    try:
        from automation import v2223_apply_docs
        v2223_apply_docs.main()
    except Exception as exc:
        print(f"[WARN] V2.22.3 docs updater unavailable: {exc}")
    for name, section in SECTIONS.items():
        print(apply_one(ROOT / name, section))


if __name__ == "__main__":
    main()
