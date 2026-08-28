from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CURRENT_STATE = """

<!-- V2223_CURRENT_STATE -->
---
## V2.22.3 shot-critical runtime / object-hit foundation — 2026-08-28

- V2.22 ShotResolver remains a fast fusion layer; physical advisory tests measured resolver work at roughly 4-9 ms.
- V2.22.1 perspective ROI/edge guard reduces the heavy detector to about 13.2% of the current 3840x2160 camera frame while restoring exact full-camera XY before the existing homography/game-coordinate path.
- V2.22.2 removes major horizontal ridge artifacts before the resolver.
- **V2.22.3 is implemented.** `main.py` installs a top-level shot-critical runtime policy before `App().run()`. The audio reader thread remains producer-only, but an already timestamped/queued shot is serviced before camera housekeeping, automation, ordinary scene update and decorative rendering.
- Per-frame camera capability probing is removed from the hot path. CameraManager main-thread pickup no longer mutates `_last_pickup_count`; HitScanner owns the timestamped new-frame cursor used for dense ring history.
- Static surface/reference and dynamic recent PRE are separate evidence concepts. Static calibration reference preserves repairs/tape/paper/projector response; recent PRE represents what actually existed immediately before the current shot.
- V2.22.3 adds a SHADOW-only object-hit foundation. Games may freeze 1..n screen-space hit regions at PANG and later ask whether an object was hit and where locally, while the global HitScanner/HitInput XY path remains authority.
- Weak viewport centre/edge evidence is advisory only and may never override strong physical evidence or an explicit valid game target near an edge.
- Runtime gate before serious live authority / overnight optimisation: audio-thread -> main dispatch p95 <50 ms, resolver p95 <10 ms, shot -> HitEvent median <250 ms and p95 <500 ms.
"""

HIT_PLAN = """

<!-- V2223_HIT_DETECTION_PLAN -->
---
## V2.22.3 — shot-critical runtime and object-first game path

V2.22.3 treats a physical audio shot as the highest-priority runtime event. `main.py` installs the policy before the App starts. The audio thread only timestamps/queues; the main thread dispatches a queued peak before camera housekeeping, automation, ordinary scene simulation and rendering, then services HitScanner before normal engine work.

The camera hot path also fixes ownership of the timestamped frame cursor: CameraManager no longer consumes `_last_pickup_count` during its ordinary update. HitScanner owns `get_new_frames_since_last_pickup()`. Camera capability probing is startup/explicit settings work, not a per-frame task.

### Two complementary game questions

```text
PANG
 |\\
 | +--> OBJECT HIT PATH (SHADOW in V2.22.3)
 |      freeze 1..100 game hit regions at the shot timestamp
 |      ask: did a NEW physical change hit this object?
 |      preserve object-local XY
 |
 +----> GLOBAL HIT PATH
        current camera detector + physical/AI evidence + ShotResolver
        ask: where did the shot go on the whole playable surface?
```

Object-first detection does **not** replace global localisation. Many arcade/game scenes may eventually react to a high-confidence object hit without waiting for an exact coordinate for an irrelevant miss. Calibration, score rings, free targets, damage decals, physical shot logging and AI Training still need authoritative global XY.

V2.22.3 only establishes the game API and evaluates existing detector candidates against frozen hit regions in shadow. No object result changes gameplay authority. A future direct per-object PRE->POST fast path may replace this evaluator behind the same API if physical tests show that it is faster and reliable.

### Evidence semantics

- **Static surface/reference:** repairs, tape, paper structure and projector response from calibration/reference capture.
- **Dynamic recent PRE:** timestamped camera state immediately before this shot, including all old holes.
- **Hole likeness:** does a point look like a physical hole?
- **Shot novelty:** did it change at this PANG?

Old-hole proximity is soft context. Re-hit/hole-in-hole remains legal when fresh temporal evidence exists. A weak viewport-centre prior may break close ties but is never physical truth and never a hard edge exclusion beyond the existing narrow calibrated edge guard. Explicit game-object context is stronger than the generic centre prior but still cannot invent a physically unsupported hit.

### Runtime gate

Before long overnight optimisation or increased live AI authority:

- audio-thread -> main dispatch p95 <50 ms,
- resolver p95 <10 ms,
- shot -> final HitEvent median <250 ms,
- shot -> final HitEvent p95 <500 ms,
- no ordinary engine task may routinely delay shot acknowledgement by ~1 s.
"""

AI_CONTEXT = """

<!-- V2223_AI_CONTEXT -->
---
## V2.22.3 shot-critical / object-hit semantics

### Top-level audio priority

`main.py` installs V2.22.3 before `App().run()`. The actual loop still executes as `App.run()`, but shot priority is a program-level runtime policy, not scene-local AI behaviour. The microphone reader thread remains producer-only: timestamp/queue only, no Pygame or game calls. The main thread acknowledges a queued shot before ordinary engine work.

### Camera frame ownership

CameraManager's normal main-thread update is a cheap latest-frame pickup. It must not mutate HitScanner's `_last_pickup_count`. HitScanner owns the timestamped new-frame cursor. Camera capability probing is not allowed in the per-frame shot-critical path.

### Static reference != recent PRE

Keep both:

- static scene/surface reference = calibration-time repairs/tape/paper/projector baseline,
- dynamic recent PRE = timestamped camera state before the current PANG, used for current-shot novelty.

Hole appearance and NEW-hole evidence are different labels. An old hole may be highly hole-like without being new. Known-hole distance stays soft so a genuine re-hit/hole-in-hole can survive when recent PRE->POST evidence is strong.

### Object-hit context

Game scenes may expose `get_hit_regions()` or register hit polygons/rectangles with `object_hit_registry_v2223`. Regions are frozen per shot before normal scene movement. V2.22.3 maps existing camera candidates to those regions and records object id, object-local XY and an **uncalibrated shadow confidence**. Object results have no authority in this version.

### Spatial context

`center_prior` and `edge_distance_norm` are advisory metadata only. A central location may be a mild generic prior, but strong physical evidence or a real target near an edge must win. Game context may support physical evidence but never drag/invent a hit.

### Cursor/debug

AI Training hides the pointer while armed for a real shot and shows it for manual GT/review. `F3` is the explicit mouse-shot/debug visibility override. `F4` toggles a latency wait cursor: it starts when the main thread acknowledges the audio event, so PANG -> hourglass delay is a useful human indicator of pre-dispatch latency; timestamp telemetry remains authoritative.
"""

SECTIONS = {
    "CURRENT_STATE.md": CURRENT_STATE,
    "HIT_DETECTION_PLAN.md": HIT_PLAN,
    "AI_CONTEXT.md": AI_CONTEXT,
}


def apply_one(path: Path, section: str) -> str:
    marker_line = next(
        (line.strip() for line in section.splitlines() if line.strip().startswith("<!-- V2223_")),
        "",
    )
    if not path.exists():
        return f"[WARN] missing {path.name}; not created"
    text = path.read_text(encoding="utf-8")
    if marker_line and marker_line in text:
        return f"[OK] {path.name}: V2.22.3 section already present"
    path.write_text(text.rstrip() + section + "\n", encoding="utf-8")
    return f"[UPDATED] {path.name}"


def main() -> None:
    print("V2.22.3 DOCUMENTATION UPDATE")
    print("============================")
    for name, section in SECTIONS.items():
        print(apply_one(ROOT / name, section))


if __name__ == "__main__":
    main()
