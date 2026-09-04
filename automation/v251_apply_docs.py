from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "<!-- V2.25.1 OBJECT_REGION_PHYSICAL_PROPOSAL -->"

SECTIONS = {
    "ARCHITECTURE.md": """
## V2.25.1 — balanced physical proposal per frozen object region

V2.25.0 physically verified scanner `shot_id` -> HitEvent -> frozen GameObject
resolution, but the detector still allowed one noisy area inside the union of all
object HitRegions to dominate candidate selection. V2.25.1 partitions the normal
first physical pass by frozen physical search area. Each area receives its own
registered PRE->POST threshold and bounded proposal/confirmation quota.

Near-identical overlapping physical regions (for example glass directly in front of
a rear target) are grouped for detector work but retain all object identities for
downstream exact collision and penetration. Target/no-shoot/object type never changes
detector evidence weight. V2.22.5 FULL-RESCUE remains global.
""",
    "HIT_DETECTION_PLAN.md": """
## V2.25.1 — object-region proposal fairness

Physical V2.25.0 testing showed all five deliberately separated shots resolving into
a small camera area despite correct V2.24.4 ROI mapping. The normal FAST path had
`primary=0` and the old local-confirm gate allowed very large candidate sets to
survive. V2.25.1 therefore changes candidate competition, not GameObject collision:

- split the frozen union into individual physical region groups;
- compute region-local robust proposal thresholds;
- retain a bounded proposal quota from every region;
- re-balance V1/V2/bank candidates by physical region;
- keep a bounded number of V2.22.5 PRE->POST confirmations per region;
- among V2.25.1-confirmed tracks, prefer stronger physical confirmation/evidence;
- never use role/owner/game score/projectile semantics to choose detector XY;
- preserve global high-recall FULL rescue.
""",
    "CURRENT_STATE.md": """
## V2.25.1 checkpoint — region-balanced physical hit proposals

V2.25.0 GameObject `shot_id` and frozen collision were physically verified. Detector
XY remained the blocker: five widely separated real shots were selected in one small
camera area. V2.25.1 adds balanced per-region proposal/confirmation on top of the
accepted V2.24.4 working-space mapping. Physical acceptance is pending a repeat of
the five-object test.
""",
    "ROADMAP.md": """
## V2.25.1 detector/object bridge correction

- [x] V2.25.0 — composable GameObject foundation and physical shot-id/frozen bridge.
- [x] V2.25.1 — build balanced per-object physical proposal/confirmation; physical
  acceptance pending.
- [ ] V2.25.2 — continuous moving-object updates while CV resolves, using frozen
  shot-id snapshots for exact collision.
- [ ] V2.25.3 — effect/audio dispatcher foundation if needed by the first migrated game.
- [ ] Migrate/build production games only after V2.25.1 physical XY is accepted.
""",
    "GAME_DEVELOPMENT.md": """
## V2.25.1 — detector fairness around GameObjects

`GameObject.get_hit_region()` still means only "search here first". V2.25.1 may
partition those regions so every physical area contributes a bounded proposal set,
but it must not infer which object the player intended to hit. Exact object shape,
z-order, health, penetration and reactions remain downstream of resolved HitEvent XY.

For overlapping objects at the same projected location, detector work may be grouped;
ObjectManager must still resolve all frozen objects at the final XY in gameplay order.
""",
    "AI_CONTEXT.md": """
## V2.25.1 AI guidance — physical region proposal

Read `AI_PHYSICAL_REGION_PROPOSAL.md` before changing object-aware hit detection.
Keep full-camera, V2.22.1 work-plane and CandidateGenerator bbox-local coordinates
explicit. Region balancing is a physical search/fairness mechanism only. Never weight
`target` above `no_shoot`, never use damage/projectile/game score to choose detector
XY, never snap XY, and preserve the global V2.22.5 FULL-rescue path.
""",
}


def _append_to_reference(name: str, marker: str, section: str) -> None:
    path = ROOT / name
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    with path.open("a", encoding="utf-8") as fh:
        if text and not text.endswith("\n"):
            fh.write("\n")
        fh.write("\n" + marker + "\n" + section.strip() + "\n")


def main() -> None:
    try:
        from automation.v250_apply_docs import main as apply_v250_docs
        apply_v250_docs()
    except Exception as exc:
        print(f"[WARN] V2.25.0 docs patch could not run: {exc}")

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
            print(f"[OK] {name} already contains V2.25.1 section")
            continue
        with path.open("a", encoding="utf-8") as fh:
            if text and not text.endswith("\n"):
                fh.write("\n")
            fh.write("\n" + MARKER + "\n" + section.strip() + "\n")
        changed += 1
        print(f"[PATCH] {name}")

    # Stable dedicated references shipped in the delta also carry a small
    # cross-reference so future AI sessions know the new physical boundary.
    _append_to_reference(
        "GAME_OBJECT_SYSTEM.md", MARKER,
        "## V2.25.1 physical-search note\n\nGameObject semantics remain unchanged. "
        "The detector may balance proposals per frozen HitRegion, but exact collision, "
        "z-order, damage and penetration run only after physical HitEvent XY exists. "
        "See `V251_OBJECT_REGION_PHYSICAL_PROPOSAL.md`."
    )
    _append_to_reference(
        "AI_GAME_OBJECTS.md", MARKER,
        "## V2.25.1 detector boundary\n\nFor object-aware hit detection also read "
        "`AI_PHYSICAL_REGION_PROPOSAL.md`. HitRegions may partition physical search, "
        "but gameplay semantics may never create, move or prefer a physical hit."
    )
    print(f"Done. changed={changed}")


if __name__ == "__main__":
    main()
