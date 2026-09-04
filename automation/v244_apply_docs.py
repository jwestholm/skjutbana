from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "<!-- V2.24.4 DETECTOR_WORKING_SPACE_ROI -->"

SECTIONS = {
    "ARCHITECTURE.md": """
## V2.24.4 — canonical camera to detector working-space ROI

The V2.24.3 physical probe exposed a coordinate-plane mismatch. V2.24 game
HitRegions are transformed to canonical full-camera coordinates, while V2.22.1
runs the expensive detector on a crop-local analysis image and translates
candidate XY back to full-camera coordinates only after detection.

V2.24.4 therefore makes the missing transform explicit:

`game-local -> screen -> full camera -> V2.22.1 analysis/crop-local -> physical detector`

The live `AnalysisGeometryV2221` supplies crop origin and dimensions. Object
AABBs are translated by the crop origin and scaled only if the actual working
image differs from the crop size. No camera resolution or fixed `/2` scale is
hard-coded. V2.22.5 FULL-RESCUE remains global and bypasses the object mask.
""",
    "HIT_DETECTION_PLAN.md": """
## V2.24.4 — detector working-space mapping correction

V2.24.3 produced `camera=7` for all test objects but `region=0.0%` inside the
local ROI stage. This was not a failure of the game->camera homography. The
region AABBs were full-camera coordinates while the ROI mask was being built
inside the V2.22.1 crop-local detector plane.

V2.24.4 maps the frozen full-camera HitRegions through the live V2.22.1 analysis
geometry before constructing the first-pass mask. Physical acceptance requires
non-zero region coverage on region-enabled shots, correct target/no-shoot/
outside classifications, and an unchanged global FULL-RESCUE path.
""",
    "CURRENT_STATE.md": """
## V2.24.4 checkpoint — detector working-space ROI mapping

The V2.24.3 physical run isolated the remaining integration bug: HitRegions
were present and transformed, but every region-enabled shot logged
`region=0.0% overlap=0`. V2.22.1 intentionally executes detection on a smaller
crop-local image and restores full camera coordinates afterwards; V2.24.3 fed
full-camera AABBs directly into that local mask.

V2.24.4 maps full-camera HitRegions to the active analysis working space using
`AnalysisGeometryV2221` crop origin and actual work/crop scale. The testscene
and logs now expose full-frame size, crop rectangle, work size, scale and an
example camera->work region mapping. V2.25.0 remains gated on one clean physical
acceptance run of this bridge.
""",
    "ROADMAP.md": """
## Game-ready hit-engine path — V2.24.4 working-space correction

- [x] **V2.24.0** — HitRegion API, transforms and shot-time snapshot.
- [x] **V2.24.1** — first object-local proposal implementation.
- [x] **V2.24.2** — dedicated physical game-context testscene.
- [x] **V2.24.3 physical probe** — isolated full-camera vs crop-local ROI mismatch.
- [x] **V2.24.4 code** — map frozen camera AABBs into V2.22.1 analysis working
  space; preserve physical authority and global FULL-RESCUE.
- [ ] **V2.24.4 physical acceptance** — verify `ROI-MAP`, non-zero `region`,
  target/no-shoot/moving/overlap/edge/outside and EMPTY/global behaviour.
- [ ] **V2.25.0** — GameObject / HittableObject / BreakableObject / ObjectManager
  after V2.24.4 physical acceptance.
""",
}


def main() -> None:
    try:
        from automation.v243_apply_docs import main as apply_v243_docs
        apply_v243_docs()
    except Exception as exc:
        print(f"[WARN] V2.24.3 docs patch could not run: {exc}")

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
            print(f"[OK] {name} already contains V2.24.4 section")
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
