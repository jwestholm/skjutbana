from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "<!-- V2.24.0 GAME_HIT_CONTEXT -->"

SECTIONS = {
    "ARCHITECTURE.md": """
## V2.24.0 — Game Hit Context

Games may expose an optional `get_hit_regions()` provider. Regions are simple
viewport-local/game-local AABBs. `OverlayScene -> GameScene -> game` proxies the
provider to the shot-critical runtime. At the audio-shot boundary the existing
V2.22.3 object snapshot freezes geometry before normal scene movement.

Coordinate ownership is strict:

`game-local AABB -> viewport/screen -> calibrated camera AABB`.

The engine transforms all four rectangle corners and bounds the result in camera
space; games never provide camera coordinates. The region is search/context
geometry only. Final hit authority still requires physical camera evidence.
""",
    "HIT_DETECTION_PLAN.md": """
## V2.24.0 — Game-context AABB contract

New games SHOULD expose approximate hit-search geometry with
`game.get_hit_regions()`. Each region is an inexpensive `(x, y, width, height)`
AABB in viewport-local/game-local coordinates. No mesh/polygon/image mask is
required from games.

At shot time the engine freezes the list before scene update and transforms the
four corners through the existing screen/camera calibration into a camera-space
AABB. If the game has no regions, or the transform is unavailable, detection
must use the ordinary global path. A region may prioritize/localize physical
search but MUST NOT invent a hit or snap an unsupported shot onto an object.

V2.24.0 establishes the contract only. V2.24.1 adds local physical PRE->POST
search plus global fallback.
""",
    "CURRENT_STATE.md": """
## V2.24.0 checkpoint

Game-hit context foundation is available. Existing games remain valid without
changes. Future games may return `HitRegion` AABBs in game-local coordinates;
shot-critical snapshotting and game->camera transformation are prepared for the
next local-physical-search stage. No local object-aware hit authority is enabled
in V2.24.0.
""",
    "ROADMAP.md": """
## Game-ready hit-engine path (V2.24+)

1. **V2.24.0** — optional game `HitRegion` AABB contract, wrapper proxies,
   shot-time snapshot, game/screen/camera transforms, documentation.
2. **V2.24.1** — object-aware local physical search in camera regions with
   global fallback and shadow metrics.
3. **V2.24.2** — moving-region/debug verification scene and physical tests.
4. **V2.25.0** — common GameObject / Hittable / Breakable / ObjectManager layer.
5. Build new games on the stable HitEvent + object-context interfaces while AI
   research continues behind the same hit engine.
""",
}


def main() -> None:
    changed = 0
    for name, section in SECTIONS.items():
        path = ROOT / name
        if not path.exists():
            print(f"[SKIP] {name} not found")
            continue
        text = path.read_text(encoding="utf-8")
        if MARKER in text:
            print(f"[OK] {name} already contains V2.24.0 section")
            continue
        with path.open("a", encoding="utf-8") as fh:
            if not text.endswith("\n"):
                fh.write("\n")
            fh.write("\n" + MARKER + "\n" + section.strip() + "\n")
        changed += 1
        print(f"[PATCH] {name}")
    print(f"Done. changed={changed}")


if __name__ == "__main__":
    main()
