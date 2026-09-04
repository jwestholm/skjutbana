from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "<!-- V2.24.1 OBJECT_LOCAL_PHYSICAL_SEARCH -->"

SECTIONS = {
    "ARCHITECTURE.md": """
## V2.24.1 — Object-aware local physical search

V2.24.0 freezes optional game `HitRegion` AABBs at PANG and transforms them to
camera coordinates. V2.24.1 consumes those frozen camera AABBs in the live hit
pipeline. Expanded/merged regions are intersected with the detector's existing
valid perspective ROI and constrain only the first V2.22.5 FAST proposal pass.

Authority remains physical: HitRegions are search context, not hit truth. The
existing V2.22.5 PRE->POST local confirmation remains mandatory and its single
FULL-RESCUE pass is explicitly global/unmasked. Missing, invalid or
untransformable context fails open to the pre-V2.24.1 global path.

Runtime layering is therefore:

`PANG -> frozen game context -> local physical proposal -> physical confirmation
-> resolver/HitEvent`, with `global detector rescue` as the fail-safe branch.
""",
    "HIT_DETECTION_PLAN.md": """
## V2.24.1 — Game-context local physical proposal search

V2.24.1 activates the first object-aware physical-search stage without changing
hit authority. Frozen V2.24.0 camera HitRegions receive a safety margin, overlap
is merged, and the result masks the existing V2.22.5 FAST extractor's `valid`
ROI for the first proposal pass.

Non-negotiable rules:

- region means **search here first**, never **hit this object**;
- physical PRE->POST evidence remains mandatory;
- candidate XY is never snapped to an object/region;
- target/no-shoot/breakable roles do not alter physical evidence thresholds;
- no context / invalid transform uses the existing global path;
- V2.22.5's one FULL-RESCUE is always global and bypasses the object mask.

V2.24.2 is the acceptance stage: moving regions, target/no-shoot, overlaps,
edges, shots immediately outside regions, empty context and transform
round-trips. Measure local-search success, global fallback, false object
attraction, XY error and latency before deeper object-aware optimisation.
""",
    "CURRENT_STATE.md": """
## V2.24.1 checkpoint — object-aware local physical search

V2.24.0 has been physically smoke-tested on the shooting PC: its selftests pass
and the application starts. V2.24.1 now uses frozen camera HitRegions to
constrain the first V2.22.5 FAST physical proposal search. Overlapping regions
are merged after a camera-space safety margin. Existing V2.22.5 PRE->POST local
confirmation remains the physical gate and its FULL-RESCUE remains global.

Games with no HitRegions retain the existing global detector behaviour. No new
AI or game-context hit authority is enabled. Next planned checkpoint is V2.24.2
with a dedicated game-context verification scene before V2.25.0 introduces the
small shared GameObject/Hittable/Breakable/ObjectManager layer.
""",
    "ROADMAP.md": """
## Game-ready hit-engine path — V2.24.1 checkpoint

- [x] **V2.24.0** — optional `HitRegion` API, game-local coordinates,
  four-corner game->camera transform and shot-time snapshot.
- [x] **V2.24.1** — region-first physical FAST proposal search, merged camera
  windows, mandatory physical confirmation and global V2.22.5 fallback.
- [ ] **V2.24.2** — dedicated moving/overlap/no-shoot/edge/outside-region test
  scene and physical verification metrics.
- [ ] **V2.25.0** — small shared GameObject / HittableObject /
  BreakableObject / ObjectManager layer.
- [ ] Resume broader game production on the stable HitEvent + HitRegion
  contracts while AI/dense/direct-heatmap research continues behind the same
  hit engine.
""",
}


def main() -> None:
    roadmap = ROOT / "ROADMAP.md"
    if not roadmap.exists():
        roadmap.write_text("# Roadmap\n", encoding="utf-8")
        print("[CREATE] ROADMAP.md base")

    # Ensure the V2.24.0 foundation section also exists. This makes V2.24.1
    # self-contained even if the previous optional docs-patch command was skipped.
    try:
        from automation.v240_apply_docs import main as apply_v240_docs
        apply_v240_docs()
    except Exception as exc:
        print(f"[WARN] V2.24.0 docs patch could not run: {exc}")

    changed = 0
    for name, section in SECTIONS.items():
        path = ROOT / name
        if not path.exists():
            if name != "ROADMAP.md":
                print(f"[SKIP] {name} not found")
                continue
            path.write_text(MARKER + "\n" + section.strip() + "\n", encoding="utf-8")
            changed += 1
            print(f"[CREATE] {name}")
            continue
        text = path.read_text(encoding="utf-8")
        if MARKER in text:
            print(f"[OK] {name} already contains V2.24.1 section")
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
