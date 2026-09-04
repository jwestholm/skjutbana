from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "<!-- V2.25.0 GAME_OBJECT_FOUNDATION -->"

SECTIONS = {
    "ARCHITECTURE.md": """
## V2.25.0 — composable GameObject foundation

V2.24.4 physically validated the game-context bridge, so V2.25 adds a gameplay
object layer downstream of physical HitEvent authority. The canonical model is
composition-based: identity/geometry, exact hit shape, ballistic body, layered
durability, motion and reactions are independent capabilities rather than a deep
inheritance tree.

Camera hits now carry scanner `shot_id` through a backward-compatible HitEvent
bridge before subscribers are notified. ObjectManager uses that id to resolve
exact collision against V2.24's frozen PANG-time shape metadata. Mouse/debug
hits retain `shot_id=None` and use current geometry.

Object effects are event requests. Sound, particles, animation and future
physics stay separate services rather than becoming GameObject responsibilities.
""",
    "HIT_DETECTION_PLAN.md": """
## V2.25.0 — hit engine / game object boundary

The V2.24.4 physical acceptance established non-zero object-local detector ROI
and correct game-context operation. V2.25 does not change detector authority.
It only preserves scanner shot identity through HitEvent so downstream gameplay
can select the exact frozen object snapshot.

Invariant: HitRegion means `search here first`; ObjectManager receives the
already-resolved `HitEvent.game_x/game_y` and never snaps or moves that point.
""",
    "CURRENT_STATE.md": """
## V2.25.0 checkpoint — reusable game objects

V2.24.4 is accepted as the game-context bridge. V2.25 introduces a stable
`src.engine.game_objects` package with exact shapes, entity/part identity,
gameplay projectile penetration, layered durability, event/reaction handling,
effect requests and ObjectManager resolution against PANG-time snapshots.

A shot-id bridge fixes the prior `event_shot=None` limitation without replacing
HitInput or changing detector authority. Continuous object motion during the
shot-critical wait remains a later checkpoint.
""",
    "ROADMAP.md": """
## V2.25 object-system path

- [x] V2.24.4 — physical local-ROI bridge accepted.
- [x] V2.25.0 — composable GameObject foundation, exact frozen collision,
  shot-id HitEvent bridge, gameplay penetration/damage, reactions/effect requests.
- [ ] V2.25.1 — permit/verify continuous moving-object updates while a physical
  shot resolves, using exact shot_id snapshot collision.
- [ ] V2.25.2 — optional sound/effect dispatcher and lightweight visual effects.
- [ ] V2.25.3+ — multipart entity aggregation only when a concrete game needs it.
- [ ] Build/migrate production games incrementally on the stable object API.
""",
    "GAME_DEVELOPMENT.md": """
## V2.25.0 — GameObject contract

Prefer `src.engine.game_objects` and composition. GameObject owns identity,
projected game-local geometry and lifecycle; hit shape, ballistic body, damage,
motion and reactions are independent capabilities. `make_living_object()` and
`make_breakable_object()` are convenience presets, not inheritance requirements.

For every camera shot, ObjectManager should resolve exact collision from
`HitEvent.shot_id` against the frozen PANG snapshot. Never use current moving
geometry when the matching frozen snapshot exists and never alter HitEvent XY.
See `GAME_OBJECT_SYSTEM.md` for the stable API.
""",
    "AI_CONTEXT.md": """
## V2.25.0 AI guidance — GameObjects

Read `AI_GAME_OBJECTS.md` and `GAME_OBJECT_SYSTEM.md` before changing game-object
or hit/game integration. Preserve physical HitEvent XY, preserve scanner
`shot_id`, keep object geometry game-local, use unique object ids for multipart
parts, treat caliber labels as metadata/config selectors, prefer composition,
and route sound/particles/animation/physics through effect requests rather than
embedding those services in GameObject.
""",
}


def main() -> None:
    try:
        from automation.v244_apply_docs import main as apply_v244_docs
        apply_v244_docs()
    except Exception as exc:
        print(f"[WARN] V2.24.4 docs patch could not run: {exc}")

    changed = 0
    for name, section in SECTIONS.items():
        path = ROOT / name
        if not path.exists():
            if name == "ROADMAP.md":
                path.write_text("# Roadmap\n", encoding="utf-8")
            else:
                print(f"[SKIP] {name} not found")
                continue
        text = path.read_text(encoding="utf-8")
        if MARKER in text:
            print(f"[OK] {name} already contains V2.25.0 section")
            continue
        with path.open("a", encoding="utf-8") as fh:
            if text and not text.endswith("\n"):
                fh.write("\n")
            fh.write("\n" + MARKER + "\n" + section.strip() + "\n")
        changed += 1
        print(f"[PATCH] {name}")
    print(f"Done. changed={changed}")


if __name__ == "__main__":
    main()
