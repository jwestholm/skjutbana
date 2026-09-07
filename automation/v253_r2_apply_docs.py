from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = '<!-- V2.25.3-r2 SETTINGS_PACKAGING_REPAIR -->'
SECTIONS = {
    'CURRENT_STATE.md': '''
## V2.25.3-r2 packaging repair

The first V2.25.3 cumulative archive accidentally contained a unit-test stub as
`src/engine/settings.py`. That stub was not part of the runtime design and removed
long-lived audio/camera/LED/settings APIs. V2.25.3-r2 does not distribute a replacement
settings module. Its prepare step restores the newest complete committed settings.py
when needed, then reapplies only the viewport-local content_rect fallback from V2.24.3.
The V2.25.3 detector/authority logic is otherwise unchanged.
''',
    'ROADMAP.md': '''
## V2.25.3-r2 packaging correction

- [x] Remove accidental settings.py unit-test stub from cumulative package.
- [x] Recover complete settings.py from repository history when the working copy was overwritten.
- [x] Reapply only the V2.24.3 viewport-local content_rect fallback.
- [x] Validate audio/camera/viewport/LED settings API surface before startup.
- [ ] Repeat V2.25.3 five-shot physical acceptance after startup repair.
''',
    'AI_CONTEXT.md': '''
## V2.25.3-r2 packaging lesson

Never ship a unit-test stub in a cumulative delta. Files modified only for isolated tests
must be created in a temporary test tree, not in the package tree. `settings.py` is a
shared compatibility surface; future deltas should patch it surgically or use an
idempotent installer rather than replacing it with a reconstructed subset.
''',
    'AI_CROSS_SHOT_NOVELTY.md': '''
## V2.25.3-r2 note

R2 changes packaging/installation only. Cross-thread readiness, cross-shot novelty,
rehit recovery, physical authority and FULL-rescue behavior are unchanged from V2.25.3.
''',
}


def _append(name: str, section: str) -> bool:
    path = ROOT / name
    if not path.exists():
        print(f'[SKIP] {name} not found')
        return False
    text = path.read_text(encoding='utf-8')
    if MARKER in text:
        print(f'[OK] {name} already contains V2.25.3-r2 section')
        return False
    with path.open('a', encoding='utf-8') as fh:
        if text and not text.endswith('\n'):
            fh.write('\n')
        fh.write('\n' + MARKER + '\n' + section.strip() + '\n')
    print(f'[PATCH] {name} V2.25.3-r2')
    return True


def main() -> None:
    changed = sum(1 for name, section in SECTIONS.items() if _append(name, section))
    print(f'V2.25.3-r2 docs done. changed={changed}')


if __name__ == '__main__':
    main()
