from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARK = '## V2.22.5 Fast Proposal + Local Confirmation — 2026-08-29'

SECTIONS = {
    'CURRENT_STATE.md': '''\n---\n## V2.22.5 Fast Proposal + Local Confirmation — 2026-08-29\n\n- V2.22.4-r2 proved audio dispatch and async AI shadow are no longer the dominant common-path latency. A physical test showed main acknowledgement in tens of milliseconds while the first async CV job remained around the one-second scale.\n- Stage profiling isolated V2 candidate extraction as the largest measured stage, and the live tracker still attempted another whole-ROI detector pass merely to accumulate persistence.\n- V2.22.5 changes normal live semantics to **GLOBAL PROPOSE -> LOCAL CONFIRM**. One async global proposal seeds HitScanner tracks; later real camera frames validate small PRE->POST patches around existing candidate coordinates instead of searching the full ROI again.\n- The live V2 worker uses a sparse FAST extractor. The historical high-recall extractor remains one explicit FULL rescue and remains available to offline/F2 work.\n- Re-hit/hole-in-hole remains legal when fresh temporal evidence exists. Local confirmation never drags/interpolates candidate XY.\n- Startup/calibration audio is consumed while HitScanner is not ACTIVE.\n''',
    'HIT_DETECTION_PLAN.md': '''\n---\n## V2.22.5 runtime funnel — global proposal is not persistence\n\nDo not use repeated whole-viewport/whole-ROI search merely to increment a candidate track's persistence counter.\n\nNormal gameplay funnel:\n\n1. audio timestamp and game/object snapshot,\n2. one async GLOBAL FAST candidate proposal,\n3. preserve camera XY/provenance through V1/V2 merge + cleanup,\n4. LOCAL PRE->POST confirmation around proposed coordinates on a later real camera frame,\n5. existing HitScanner track association/readiness,\n6. immediate camera->game HitEvent,\n7. AI/advisory remains off the critical path.\n\nIf local confirmation validates nothing, permit one FULL high-recall rescue. Track rescue frequency as a runtime metric. The same local physical-change primitive is intended for future object-first HitRegions.\n''',
    'AI_CONTEXT.md': '''\n---\n## V2.22.5 proposal / confirmation semantics\n\n- A candidate is a proposed physical location, not proof of a new hit.\n- Local confirmation means a later timestamped frame provides compact PRE->POST change near the same candidate XY.\n- Local confirmation never moves authoritative XY; diagnostic best-offset values are evidence only.\n- An unchanged old hole may look hole-like but should fail current-shot temporal confirmation.\n- A true re-hit may pass because fresh PRE->POST evidence exists.\n- Live FAST extraction does not replace the full research/high-recall extractor; FULL rescue and offline/F2 paths retain it.\n''',
}


def main() -> None:
    for name, section in SECTIONS.items():
        path = ROOT / name
        if not path.exists():
            print(f'[WARN] {name} not found; leaving unchanged')
            continue
        text = path.read_text(encoding='utf-8')
        marker = section.strip().splitlines()[1] if section.strip().startswith('---') else MARK
        # Use the V2.22.5 heading as idempotency marker.
        if 'V2.22.5' in text and ('GLOBAL PROPOSE -> LOCAL CONFIRM' in text or MARK in text):
            print(f'[OK] {name}: V2.22.5 section already present')
            continue
        with path.open('a', encoding='utf-8') as fh:
            if text and not text.endswith('\n'):
                fh.write('\n')
            fh.write(section)
        print(f'[UPDATED] {name}')


if __name__ == '__main__':
    main()
