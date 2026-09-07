from __future__ import annotations

from src.engine.shot_object_local_v241 import SCHEMA_VERSION, ObjectLocalConfigV241


def main() -> None:
    cfg = ObjectLocalConfigV241()
    print("V2.24.1 STATUS")
    print("===============")
    print(f"Runtime schema: {SCHEMA_VERSION}")
    print("Object context: frozen V2.24.0 camera HitRegions")
    print(f"Default local margin: {cfg.margin_px:.0f} camera px")
    print("First pass: V2.22.5 FAST proposals inside merged object windows")
    print("Physical gate: existing V2.22.5 PRE->POST local confirmation")
    print("Failure path: existing V2.22.5 FULL-RESCUE, GLOBAL/unmasked")
    print("No HitRegions / bad transform: existing global path")
    print("Candidate snapping: NEVER")
    print("Live authority: unchanged; context cannot invent a hit")
    print("Next: V2.24.2 dedicated game-context verification scene")


if __name__ == "__main__":
    main()
