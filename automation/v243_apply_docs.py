from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "<!-- V2.24.3 LOCAL_ROI_INTEGRATION_FIX -->"

SECTIONS = {
    "ARCHITECTURE.md": """
## V2.24.3 — HitScanner-level object-local ROI

Physical V2.24.2 testing showed that V2.24.1 was installed too low in the
proposal pipeline: CandidateGeneratorV2 could be masked while legacy/V1 and the
V2 waiting-post-peak path still saw the ordinary global ROI. V2.24.3 moves the
region restriction to `HitScanner._frame_roi_mask()`, before those proposal
branches split.

The normal first pass is now:

`global calibrated/content ROI -> intersect frozen camera HitRegions -> V1/V2 physical proposals`

If the two masks have zero overlap, the calibrated object-region mask is used as
an explicit ROI-recovery first pass rather than silently falling back global.
This still cannot create a hit: normal PRE->POST physical evidence and track/local
confirmation remain required. V2.22.5 FULL-RESCUE bypasses the object mask and
receives the original global ROI.

V2.24.3 also corrects the implicit content rectangle at runtime. `content_rect`
is viewport-local; when no explicit rectangle exists its correct default is
`(0, 0, viewport.w, viewport.h)`, not a copy carrying viewport.x/y.
""",
    "HIT_DETECTION_PLAN.md": """
## V2.24.3 — local ROI integration correction

V2.24.2 physical acceptance found repeated `outside_detector_roi` fallbacks even
though all seven game regions transformed successfully. The fix is not a new
ranker. Region restriction now happens at HitScanner ROI level so V1, early V2
and normal V2 share the same first-pass search area.

Acceptance requires visible `[V2.24.3 LOCAL-ROI]` or `[V2.24.3 ROI-RECOVERY]`
lines on region-enabled shots, correct physical XY for the dedicated test
objects, and an explicit global ROI when V2.22.5 FULL-RESCUE is requested.
V2.25.0 remains blocked until this physical bridge behaves correctly.
""",
    "CURRENT_STATE.md": """
## V2.24.3 checkpoint — local ROI integration fix

The first V2.24.2 physical test verified game->camera snapshots, moving-target
classification, overlap/no-shoot semantics and EMPTY/global compatibility, but
it also exposed that the intended V2.24.1 local-first detector was frequently
bypassed. Most shots logged `outside_detector_roi` and no V2.24.1 LOCAL-SEARCH.

V2.24.3 fixes the implicit viewport-local content-rect origin, refreshes HitInput
calibration before the shot snapshot and applies the frozen object mask at
HitScanner ROI level. FULL-RESCUE remains global. The V2.24.2 testscene is
retained and upgraded with faster movement plus frozen/current motion distance.

Next action: repeat the short physical Hit Context Test matrix. Only after clean
local-ROI results should development proceed to V2.25.0 reusable objects.
""",
    "ROADMAP.md": """
## Game-ready hit-engine path — V2.24.3 correction

- [x] **V2.24.0** — HitRegion API, transforms and shot-time snapshot.
- [x] **V2.24.1** — first object-local proposal implementation.
- [x] **V2.24.2** — dedicated physical game-context testscene.
- [x] **V2.24.2 physical probe** — exposed ROI integration/bypass defect.
- [x] **V2.24.3 code** — HitScanner-level local ROI, implicit content-rect fix,
  latest-calibration refresh and global FULL-RESCUE preservation.
- [ ] **V2.24.3 physical acceptance** — repeat target/no-shoot/moving/overlap/
  edge/outside/EMPTY matrix and verify LOCAL-ROI/ROI-RECOVERY/global rescue logs.
- [ ] **V2.25.0** — GameObject / HittableObject / BreakableObject / ObjectManager
  after V2.24.3 physical acceptance.
""",
}


def main() -> None:
    try:
        from automation.v242_apply_docs import main as apply_v242_docs
        apply_v242_docs()
    except Exception as exc:
        print(f"[WARN] V2.24.2 docs patch could not run: {exc}")

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
            print(f"[OK] {name} already contains V2.24.3 section")
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
