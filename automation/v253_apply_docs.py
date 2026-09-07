from __future__ import annotations
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MARKER='<!-- V2.25.3 CROSS_THREAD_NOVELTY_AUTHORITY -->'
SECTIONS={
'ARCHITECTURE.md': '''
## V2.25.3 — shared registered readiness and cross-shot physical novelty

V2.25.2 exposed an async ownership bug: CandidateGenerator marked readiness on the CV
worker scanner while authority read a different main scanner instance. V2.25.3 moves
that state to a lock-protected process-local bridge keyed by shot id and peak timestamp.
It also compares confirmed candidate locations across prior shots in canonical camera
coordinates so persistent hotspots receive a soft recurrence penalty. Re-hits remain
legal through registered signature-gain recovery. FULL rescue remains global.
''',
'HIT_DETECTION_PLAN.md': '''
## V2.25.3 — fix async readiness; suppress repeated physical hotspots softly

Physical V2.25.2 logs contained both `REGISTERED-READY` and later
`reason=no_registered_frame` for the same shot. The registered frame was real, but its
ready flag lived on the worker scanner instance. This release makes readiness shared
across worker/main and replaces local fail-open with the existing global rescue.

Because V2.25.2 marked almost every candidate fresh on the worn board, V2.25.3 adds
cross-shot camera-coordinate recurrence as a physical prior. It is soft, supports rehit
recovery, and contains no gameplay semantics.
''',
'CURRENT_STATE.md': '''
## V2.25.3 checkpoint — cross-thread novelty authority

V2.25.0 shot_id/frozen GameObject resolution is physically verified. V2.25.1 per-region
proposal works. V2.25.2 registered evidence runs physically but its readiness bridge
was instance-local and its freshness gate remained too permissive. V2.25.3 fixes both
at the authority layer; five-shot physical acceptance is pending.
''',
'ROADMAP.md': '''
## V2.25.3 authority correction

- [x] V2.25.0 — composable GameObject foundation + frozen shot bridge.
- [x] V2.25.1 — balanced physical region proposals.
- [x] V2.25.2 — registered PRE→POST freshness gate.
- [x] V2.25.3 — worker/main readiness bridge + cross-shot physical novelty; acceptance pending.
- [ ] V2.25.4 — moving-object continuity after physical XY acceptance.
- [ ] V2.25.5 — effect/audio dispatcher when required by a migrated game.
''',
'GAME_DEVELOPMENT.md': '''
## V2.25.3 — repeated hotspot handling remains physical

GameObjects still contribute only search regions. V2.25.3 can prefer a newly appearing
camera-space physical change over a hotspot repeatedly confirmed on earlier shots, but
it may not prefer a target over no-shoot or move XY into an object. Re-hit is legal and
can recover via stronger registered evidence.
''',
'AI_CONTEXT.md': '''
## V2.25.3 AI guidance

Read `AI_CROSS_SHOT_NOVELTY.md`. Worker and main scanners do not share arbitrary
instance fields. Keep authority state shot-scoped across threads, compare recurrence in
canonical camera XY, preserve legal re-hits, never use game semantics for physical
selection, and retain global FULL rescue.
''',
'GAME_OBJECT_SYSTEM.md': '''
## V2.25.3 detector boundary

The composable GameObject schema is unchanged. V2.25.3 is strictly upstream physical
authority plumbing and recurrence handling.
''',
'AI_GAME_OBJECTS.md': '''
## V2.25.3 detector note

GameObject role/type must not enter cross-shot novelty. Only physical camera evidence
and shot history are permitted before HitEvent XY is resolved.
''',
'AI_PHYSICAL_REGION_PROPOSAL.md': '''
## V2.25.3 follow-up

Per-region proposal remains the recall partition. Final local authority now also uses a
shared worker/main ready bridge and soft cross-shot camera-space novelty.
''',
'AI_REGISTERED_FRESHNESS.md': '''
## V2.25.3 follow-up

Registered freshness is still required but is not treated as sufficient on its own.
Physical testing showed it could be permissive on a worn/projected surface. V2.25.3
adds cross-shot recurrence without hard-excluding old coordinates.
''',
}

def _repair_superseded_planning_refs():
    replacements={
        'ROADMAP.md': [
            ('- [ ] V2.25.3 — continuous moving-object updates using exact shot_id snapshot collision.', '- [ ] V2.25.4 — continuous moving-object updates using exact shot_id snapshot collision.'),
            ('- [ ] V2.25.3 — continuous moving-object updates while CV resolves, using frozen shot-id snapshots for exact collision.', '- [ ] V2.25.4 — continuous moving-object updates while CV resolves, using frozen shot-id snapshots for exact collision.'),
            ('- [ ] V2.25.3 — continuous moving-object updates while CV resolves, retaining frozen PANG collision.', '- [ ] V2.25.4 — continuous moving-object updates while CV resolves, retaining frozen PANG collision.'),
        ],
        'AI_PHYSICAL_REGION_PROPOSAL.md': [
            ('V2.25.3 is reserved for continuous moving-object', 'V2.25.4 is reserved for continuous moving-object'),
        ],
        'V250_GAME_OBJECT_SYSTEM_PLAN.md': [
            ('### V2.25.3 — entity/part aggregation if a game needs it', '### Future — entity/part aggregation if a game needs it'),
            ('Moving-object continuity is now V2.25.3; the underlying GameObject design remains unchanged.', 'V2.25.3 was subsequently used for the cross-thread/novelty authority correction. Moving-object continuity is now V2.25.4; the underlying GameObject design remains unchanged.'),
        ],
        'V252_DOC_PATCH.md': [
            ('V2.25.2 moves continuous moving-object updates to the next milestone (V2.25.3) because\nthe physical XY authority leak must be closed first.', 'V2.25.2 originally moved continuous moving-object updates to V2.25.3. Physical V2.25.2 acceptance then exposed another authority issue, so V2.25.3 became the cross-thread/novelty correction and moving-object continuity is now V2.25.4.'),
        ],
    }
    for name,pairs in replacements.items():
        path=ROOT/name
        if not path.exists(): continue
        text=path.read_text(encoding='utf-8'); new=text
        for old,repl in pairs: new=new.replace(old,repl)
        if new!=text:
            path.write_text(new,encoding='utf-8'); print(f'[PATCH] {name} superseded V2.25.3 planning refs')


def _append(name,section):
    p=ROOT/name
    if not p.exists():
        if name=='ROADMAP.md': p.write_text('# Roadmap\n',encoding='utf-8')
        else:
            print(f'[SKIP] {name} not found'); return False
    t=p.read_text(encoding='utf-8')
    if MARKER in t:
        print(f'[OK] {name} already contains V2.25.3 section'); return False
    with p.open('a',encoding='utf-8') as f:
        if t and not t.endswith('\n'): f.write('\n')
        f.write('\n'+MARKER+'\n'+section.strip()+'\n')
    print(f'[PATCH] {name}'); return True

def main():
    _repair_superseded_planning_refs()
    try:
        from automation.v252_apply_docs import main as prev
        prev()
    except Exception as exc: print(f'[WARN] V2.25.2 docs patch could not run: {exc}')
    changed=sum(1 for n,s in SECTIONS.items() if _append(n,s))
    print(f'Done. changed={changed}')
if __name__=='__main__': main()
