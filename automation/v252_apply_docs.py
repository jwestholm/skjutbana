from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "<!-- V2.25.2 REGISTERED_FRESHNESS_AUTHORITY -->"
SECTIONS = {
    "ARCHITECTURE.md": """
## V2.25.2 — registered freshness authority for object-context hits

V2.25.1 physically bounded candidates per frozen HitRegion but exposed an authority
leak: legacy/bank tracks inside a region could still win, including before a registered
V2 PRE→POST frame had run. V2.25.2 keeps those candidates for recall but requires normal
object-context authority to be independently supported by CandidateGeneratorV2's
registered immediate PRE→POST maps and then by V2.22.5 second-frame persistence.

The selector contains no target/no-shoot/game weighting and never changes XY. The
explicit V2.22.5 FULL rescue remains the global physical fallback.
""",
    "HIT_DETECTION_PLAN.md": """
## V2.25.2 — close early/legacy local-authority leak

Physical V2.25.1 logs showed successful per-region balancing, but final hit XY still
collapsed into one small area. Some shots completed confirmation without any
`REGION-PROPOSAL`, proving CandidateGeneratorV2's early `waiting_post_peak` legacy path
could emit before registered evidence. Other shots selected a region even when that
region had zero registered proposals.

V2.25.2 therefore adds registered compact freshness at the same XY, immediate PRE-stack
noise as a soft penalty, early emission gating until a registered frame exists, and
registered-fresh-only local authority. Legacy coordinates may be revalidated; they are
not discarded. FULL rescue remains global.
""",
    "CURRENT_STATE.md": """
## V2.25.2 checkpoint — registered PRE→POST authority

V2.25.0 shot-id/frozen GameObject resolution is physically verified. V2.25.1 region
proposal balancing is physically verified but did not correct final XY because local
authority still leaked through early/legacy candidates. V2.25.2 closes that authority
boundary. Physical five-shot acceptance is pending.
""",
    "ROADMAP.md": """
## V2.25.2 authority correction

- [x] V2.25.0 — composable GameObject foundation + shot-id/frozen bridge.
- [x] V2.25.1 — balanced per-region physical proposal/confirmation.
- [x] V2.25.2 — registered freshness authority + early legacy emission gate; physical acceptance pending.
- [ ] V2.25.3 — continuous moving-object updates while CV resolves, retaining frozen PANG collision.
- [ ] V2.25.4 — effect/audio dispatcher foundation when required by the first migrated game.
""",
    "GAME_DEVELOPMENT.md": """
## V2.25.2 — GameObject context and physical authority

A GameObject HitRegion may supply a physical search partition, but a local hit cannot
be emitted merely because a legacy/bank candidate lies inside that region. Normal local
authority must have registered immediate PRE→POST freshness at the same physical XY.
This remains independent of role, health, penetration and reactions. Exact object
collision still begins only after HitEvent XY exists.
""",
    "AI_CONTEXT.md": """
## V2.25.2 AI guidance — registered freshness

Read `AI_REGISTERED_FRESHNESS.md`. For object-context authority distinguish proposal
recall from physical authority. Legacy/V1/bank candidates remain legal proposals but
must be revalidated against registered immediate PRE→POST evidence before normal local
emission. Do not use target/no-shoot/game semantics, do not snap XY, and preserve the
global V2.22.5 FULL rescue.
""",
}


def _append(name: str, section: str) -> bool:
    path = ROOT / name
    if not path.exists():
        if name == "ROADMAP.md":
            path.write_text("# Roadmap\n", encoding="utf-8")
        else:
            print(f"[SKIP] {name} not found")
            return False
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[OK] {name} already contains V2.25.2 section")
        return False
    with path.open("a", encoding="utf-8") as fh:
        if text and not text.endswith("\n"):
            fh.write("\n")
        fh.write("\n" + MARKER + "\n" + section.strip() + "\n")
    print(f"[PATCH] {name}")
    return True


def main() -> None:
    try:
        from automation.v251_apply_docs import main as prev
        prev()
    except Exception as exc:
        print(f"[WARN] V2.25.1 docs patch could not run: {exc}")
    changed = sum(1 for name, section in SECTIONS.items() if _append(name, section))
    for name, text in {
        "AI_GAME_OBJECTS.md": "## V2.25.2 physical authority boundary\n\nAlso read `AI_REGISTERED_FRESHNESS.md`. GameObject semantics remain downstream of physical HitEvent XY.",
        "AI_PHYSICAL_REGION_PROPOSAL.md": "## V2.25.2 authority follow-up\n\nRegion fairness alone is not authority. Legacy/bank coordinates require registered PRE→POST revalidation; see `V252_REGISTERED_FRESHNESS_AUTHORITY.md`.",
        "GAME_OBJECT_SYSTEM.md": "## V2.25.2 detector boundary\n\nThe object model is unchanged. V2.25.2 only strengthens physical hit authority before ObjectManager collision.",
    }.items():
        _append(name, text)
    print(f"Done. changed={changed}")

if __name__ == "__main__":
    main()
