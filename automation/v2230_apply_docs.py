from __future__ import annotations

from pathlib import Path

MARKERS = {
    "CURRENT_STATE.md": "### V2.23.0 — autonomous training/model pipeline",
    "HIT_DETECTION_PLAN.md": "### V2.23 — unified self-learning layer",
    "AI_CONTEXT.md": "### V2.23 model semantics",
}

SECTIONS = {
"CURRENT_STATE.md": r'''\n---\n### V2.23.0 — autonomous training/model pipeline\n\n- V2.22.1–V2.22.6 runtime is frozen as the current perception/runtime foundation while model learning becomes the primary workstream.\n- Known audio loading/mechanical false-trigger behavior and remaining runtime polish are parked TODOs unless they corrupt training data.\n- Audit: F1/F2 currently updates `SimpleAIMemory`; V2.11/V9, V2.15 Hole-AI, V2.17 NewHole-AI and V2.21.5 physical-dense are separate research/shadow models rather than one champion system.\n- V2.16 candidate packs are the main reusable bridge because they preserve actual detector candidates + GT + candidate patches/provenance. V2.19 generated worlds can compile compatible candidate packs.\n- V2.23 introduces a native append-only shot-group schema, tolerant legacy pack importer, stable physical feature contract, listwise linear/compact-MLP challengers, validation-only research champion registry, F2 capture, and offline/time-budgeted autotrain.\n- Protected holdout is never used by automatic model selection. V2.23.0 can promote only `research_shadow_champion`; `eligible_for_live_authority=false` is non-negotiable.\n''',
"HIT_DETECTION_PLAN.md": r'''\n---\n### V2.23 — unified self-learning layer\n\nThe active objective is no longer more runtime filtering. It is to learn the correct **current new hit** among the actual candidate group while preserving proposal recall as a separate metric.\n\nTraining sources: V2.23 native F2 projected-camera groups; V2.23 manual physical GT groups; V2.16 historical candidate packs; V2.19/V2.20 generated candidate packs; and later explicit evidence adapters for independent Hole-AI/NewHole-AI sources.\n\nRules: split by whole physical session whenever possible; provisional engineering splits never justify authority; never force GT into candidate pools; report proposal oracle separately from ranking; exclude GT/policy/model-output shortcuts from physical features; validation chooses challengers; protected holdout never drives autonomous selection; V2.23.0 permits research/shadow champion only.\n\nImmediate data goal: several genuinely independent physical/projected sessions plus large generated-world training. Unseen physical sessions decide eventual authority.\n''',
"AI_CONTEXT.md": r'''\n---\n### V2.23 model semantics\n\nKeep separate: (1) hole appearance — old/new real holes may both be hole-like; (2) current NEW-hole evidence — old holes are NOT-current negatives; (3) candidate ranking — choose among actual hypotheses without inventing coordinates.\n\nF2 remains compatible with legacy `SimpleAIMemory` learning but is also a V2.23 data producer. V2.23 challenger training is independent and can run after F2 or offline. V2.23 shadow models never reorder or override live candidates in this release.\n\nParked runtime TODO: mechanical/loading audio false trigger -> future audio-proposal/physical-confirmation gate; remaining CV spikes; object-hit authority; final cursor/gameplay polish.\n''',
}


def main() -> None:
    for name, marker in MARKERS.items():
        path = Path(name)
        if not path.exists():
            print(f"[SKIP] {name} not found")
            continue
        text = path.read_text(encoding="utf-8")
        if marker in text:
            print(f"[OK] {name}: V2.23 section already present")
            continue
        with path.open("a", encoding="utf-8") as handle:
            handle.write(SECTIONS[name].replace("\\n", "\n"))
        print(f"[PASS] {name}: V2.23 section appended")


if __name__ == "__main__":
    main()
