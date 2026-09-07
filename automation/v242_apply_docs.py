from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "<!-- V2.24.2 GAME_CONTEXT_TESTSCENE -->"

SECTIONS = {
    "ARCHITECTURE.md": """
## V2.24.2 — Game-context verification scene

V2.24.2 adds a diagnostic game module that exercises the V2.24 HitRegion
contract without introducing the V2.25 object engine. The scene provides
stationary target/no-shoot regions, a moving target, overlapping regions, an
edge target, an outside-region challenge and an EMPTY-regions mode.

The scene subscribes to the normal HitEvent path. When a camera/audio hit
arrives it compares final game XY against the frozen shot-time `game_regions`
when that snapshot is available. Frozen positions are drawn temporarily in
cyan, making movement between PANG and delayed HitEvent delivery visible.

This checkpoint changes no detector or game authority. Its job is to expose
transform, shot-time, local-search/fallback and false-attraction behaviour
before V2.25.0 introduces reusable GameObject/HittableObject classes.
""",
    "HIT_DETECTION_PLAN.md": """
## V2.24.2 — Physical game-context acceptance scene

The dedicated `Hit Context Test (V2.24.2)` scene is the acceptance harness for
V2.24.0/1. Physical testing must cover:

- stationary target and no-shoot regions,
- moving target with shot-time frozen geometry,
- overlapping target/no-shoot regions,
- region near the viewport edge,
- a shot immediately outside a visible target region,
- EMPTY regions, which must use the ordinary global detector,
- game->camera->game transform sanity and normal HitEvent delivery.

Success means object context reduces/structures search without snapping final XY
or turning a nearby object into hit truth. V2.24.2 remains diagnostic only;
V2.25.0 is still the first shared object/item engine checkpoint.
""",
    "CURRENT_STATE.md": """
## V2.24.2 checkpoint — dedicated HitRegion testscene

V2.24.1 installed and started successfully after V2.24.0. No physical V2.24.1
shot series was required before proceeding because V2.24.2 provides the scene
needed to exercise HitRegions intentionally.

The new scene is available under Games after running
`python3 -m automation.v242_prepare`. It exposes target/no-shoot/moving/overlap/
edge/outside-region cases plus an EMPTY-regions global-fallback mode. The scene
logs returned HitEvent XY and shows the frozen shot-time region geometry in
cyan for direct visual comparison.

No V2.25 object system or new AI authority is enabled yet. The next decision is
based on physical V2.24.2 results: fix the V2.24 bridge if required, otherwise
proceed to V2.25.0 GameObject/HittableObject/BreakableObject/ObjectManager.
""",
    "ROADMAP.md": """
## Game-ready hit-engine path — V2.24.2 checkpoint

- [x] **V2.24.0** — HitRegion API, transforms and shot-time snapshot.
- [x] **V2.24.1** — object-aware local physical search with global fallback.
- [x] **V2.24.2** — dedicated target/no-shoot/moving/overlap/edge/outside/empty
  game-context verification scene.
- [ ] **V2.24.2 physical acceptance** — run the short prescribed shot matrix and
  inspect local-search, fallback, frozen geometry and returned XY.
- [ ] **V2.25.0** — first shared GameObject / HittableObject / BreakableObject /
  ObjectManager layer, assuming V2.24.2 acceptance is clean.
""",
}


def main() -> None:
    try:
        from automation.v241_apply_docs import main as apply_v241_docs
        apply_v241_docs()
    except Exception as exc:
        print(f"[WARN] V2.24.1 docs patch could not run: {exc}")

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
            print(f"[OK] {name} already contains V2.24.2 section")
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
