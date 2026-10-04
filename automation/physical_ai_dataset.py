"""Fixed DEVELOPMENT witness cache and scoreboard. No capture or live authority.

Consumes the already measured chronological replay, never regenerates proposals.
Repeated session CV is research model selection, NOT untouched validation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from automation.accuracy_physical_dataset import guard_path
from automation.ai_training_results import get_git_commit

BASE = Path('/data/skjutbana/evaluation_runs/batched_shared_temporal_20260928_01')
SCHEMA = 'physical-witness-lab-1'
EVENTS = {'KOMP': range(1, 7), 'D02': range(1, 7), 'S01': range(1, 11),
          'S02': range(1, 14), 'POST_FIX': range(1, 11)}
CONTROLS = {'D02': (2, 4, 5, 6), 'S01': (4,), 'S02': (2, 8, 10)}
FAMILIES = {
    'chronological': ('support', 'signed_change', 'noise', 'choice_count'),
    'temporal_witness': ('supported', 'contradicted', 'revoked'),
    'morphology': ('area', 'radius', 'circularity'),
    'local_change': ('center_darkening', 'local_contrast_gain', 'blackhat_value',
                     'change_value', 'v251_region_evidence', 'v251_region_temporal'),
    'producer_history': ('score', 'observations'),
}
RAW_NAMES = tuple(n for names in FAMILIES.values() for n in names)
FEATURE_NAMES = RAW_NAMES + tuple(n + '__missing' for n in RAW_NAMES)


def guard(value):
    """Reject lexical names BEFORE resolution/open, including aliases/extensions."""
    text = os.fspath(value)
    if re.search(r'(^|[^a-z0-9])s03([^a-z0-9]|$)|validation_untouched', text, re.I):
        raise PermissionError('S03/untouched validation is forbidden in the physical lab')
    guard_path(text)
    resolved = str(Path(text).resolve())
    if re.search(r'(^|[^a-z0-9])s03([^a-z0-9]|$)|validation_untouched', resolved, re.I):
        raise PermissionError('Forbidden resolved source')


def install_guard():
    """Defense in depth for this process; no discovery of physical session roots."""
    def audit(event, args):
        if event in ('open', 'os.listdir', 'os.scandir') and args and isinstance(args[0], (str, bytes, Path)):
            text = os.fsdecode(args[0])
            if re.search(r'(^|[^a-z0-9])s03([^a-z0-9]|$)|validation_untouched', text, re.I):
                raise PermissionError('Forbidden validation access')
    sys.addaudithook(audit)


def read(path):
    guard(path)
    return json.loads(Path(path).read_text())


def digest(path):
    guard(path)
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


LABEL_LINEAGE_STATES = (
    'LABEL_SOURCE_LINEAGE_VERIFIED',
    'LABEL_SOURCE_LINEAGE_VERIFIED_WITH_KNOWN_TRANSFORM',
    'LABEL_SOURCE_ASSOCIATION_CONTRADICTED',
    'LABEL_SOURCE_LINEAGE_UNRESOLVED',
)


def with_label_source_lineage(data, assessment):
    """Optional hash-bound reporting sidecar; never filter/relabel model data.

    A manual label's existence/quality is not proof of its displayed source frame.
    Missing assessments remain NOT_AUDITED, not implicitly verified. This does
    not load source imagery or alter the immutable dataset/cache/feature schema.
    Verified findings must come from the cited provenance audit, not model scores.
    """
    import copy
    if assessment.get('schema') != 'physical-label-lineage-1' or assessment.get('dataset_hash') != data['dataset_hash']:
        raise ValueError('Label-lineage sidecar belongs to a different dataset/schema')
    records = assessment.get('events', [])
    if len({r['event'] for r in records}) != len(records):
        raise ValueError('Duplicate lineage assessment')
    events = copy.deepcopy(data['events']);lookup = {e['event']: e for e in events}
    for record in records:
        guard(record['event'])
        if record['event'] not in lookup or record.get('status') not in LABEL_LINEAGE_STATES:
            raise ValueError('Unknown event or lineage state')
        event = lookup[record['event']]
        for name, key in [('ground_truth.json', 'label_sha256'), ('trace.json', 'trace_sha256')]:
            path = str(Path(event['source_root']) / name);guard(path)
            expected = data['source_hashes'].get(path)
            if not expected or record.get(key) != expected:
                raise ValueError('Lineage does not match immutable label/trace identity')
        if not record.get('reason') or not record.get('report'):
            raise ValueError('Lineage needs explicit reason and provenance report')
        guard(record['report'])
        source = record.get('annotation_source_frame') or {}
        if source.get('path'):
            guard(source['path'])
        if record['status'] in LABEL_LINEAGE_STATES[:2]:
            if not source.get('path') or not re.fullmatch('[0-9a-f]{64}', str(source.get('sha256', ''))):
                raise ValueError('Verified image lineage needs immutable source identity')
        event['label_source_lineage'] = copy.deepcopy(record)
    for event in events:
        event.setdefault('label_source_lineage', dict(status='NOT_AUDITED'))
    return events


def label_lineage_partition(events):
    """Explicit diagnostic membership only; no automatic benchmark exclusion."""
    out = dict(verified=[], unresolved=[], contradicted=[], not_audited=[])
    for event in events:
        guard(event['event'])
        status = event.get('label_source_lineage', {}).get('status', 'NOT_AUDITED')
        if status not in LABEL_LINEAGE_STATES + ('NOT_AUDITED',):
            raise ValueError('Unknown lineage state')
        key = ('verified' if status in LABEL_LINEAGE_STATES[:2] else
               'unresolved' if status == LABEL_LINEAGE_STATES[3] else
               'contradicted' if status == LABEL_LINEAGE_STATES[2] else 'not_audited')
        out[key].append(event['event'])
    return out


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def output_root(path):
    guard(path)
    p = Path(path).resolve()
    if not p.is_relative_to('/data/skjutbana'):
        raise ValueError('Lab output must be under /data/skjutbana')
    return p


def vector(candidate):
    """Allowlisted measurements ONLY; no XY, GT, session or retrospective POST."""
    w = candidate['physical_witness']; m = w['measurements']
    raw = dict(candidate)
    raw.update({k: m.get(k) for k in ('support', 'signed_change', 'noise')})
    raw.update(choice_count=len(m.get('choices', [])), supported=w['state'] == 'SUPPORTED',
               contradicted=w['state'] == 'CONTRADICTED', revoked=bool(w['revocation']['revoked_now']))
    values, missing = [], []
    for name in RAW_NAMES:
        v = raw.get(name)
        absent = v is None or not np.isfinite(float(v))
        values.append(0.0 if absent else float(v)); missing.append(float(absent))
    return values + missing


def folds(sessions):
    sessions = sorted(sessions)
    if len(sessions) < 3 or len(set(sessions)) != len(sessions):
        raise ValueError('Need >=3 unique whole sessions')
    for s in sessions:
        guard(s)
    return [dict(held=[s], dev=[sessions[(i + 1) % len(sessions)]],
                 train=[x for x in sessions if x not in (s, sessions[(i + 1) % len(sessions)])])
            for i, s in enumerate(sessions)]


def build(output, base=BASE):
    output = output_root(output); guard(base)
    output.mkdir(parents=True, exist_ok=False)
    baseline = {r['event']: r for r in read(base / 'selected_results.json')['per_event']}
    events, vectors, provenance, hashes = [], [], [], {}
    def track(path):
        key = str(path)
        if key not in hashes:
            hashes[key] = digest(path)
    track(base / 'selected_results.json')
    for session, ids in EVENTS.items():
        for sid in ids:
            key = f'{session}:{sid}'
            path = base / 'after_replay' / f'{session}_{sid}_frozen.json'
            record = read(path); track(path)
            if record['event'] != key:
                raise ValueError('Event identity mismatch')
            bank = record['retained']
            if not bank or len(bank) > 24:
                raise ValueError('Unexpected bounded replay bank')
            w = bank[0]['physical_witness']
            source = Path(w['frames'][0]['frame_id']).parent.parent
            guard(source)
            gt_path = source / 'ground_truth.json'
            truth = 'NO_IMPACT' if sid in CONTROLS.get(session, ()) else 'HIT'
            gt = None
            if truth == 'HIT':
                label = read(gt_path); track(gt_path)
                gt = [float(label['camera_x']), float(label['camera_y'])]
            trace = source / 'trace.json'
            guard(trace)
            if trace.exists():
                track(trace)
            indices, candidates = [], []
            for c in bank:
                proof = c['physical_witness']
                if proof['event_id'] != key or proof['camera_xy'] != c['camera_xy']:
                    raise ValueError('Exact identity mismatch')
                indices.append(len(vectors)); vectors.append(vector(c))
                candidates.append(dict(id=c['uid'], xy=c['camera_xy']))
                provenance.append(dict(event=key, candidate_id=c['uid'], witness=proof,
                                       source_ordinal=c['source_ordinal'], reference_ordinals=c['reference_ordinals']))
            # Hash actual available input frames once; no S03 discovery and no replay.
            for f in w['frames']:
                track(Path(f['frame_id']))
            current_error = baseline[key]['current_error']
            events.append(dict(event=key, session=session, source_session=source.parent.parent.name,
                               truth=truth, gt_xy=gt, indices=indices, candidates=candidates,
                               source_root=str(source), trace_available=trace.exists(),
                               cutoff_ordinal=record['primary_cutoff_ordinal'],
                               cutoff_receipt=record['primary_cutoff_receipt'], status=record['status'],
                               frames=w['frames'],
                               baseline_xy=baseline[key]['selected_xy'], current_error=current_error,
                               protected=bool(current_error is not None and current_error <= 42),
                               caveats=['POSSIBLE_MULTIPLE_VISIBLE_CHANGES_UNCONFIRMED'] if key == 'S02:1' else [],
                               missing=['exposure_time', 'worker_delivery_parity', 'vision_oof_score',
                                        'known_hole_context', 'motion_context', 'structure_context'],
                               truth_provenance='native manual GT; controls explicit operator mapping preserved in prior reports'))
    x = np.asarray(vectors, np.float32)
    np.save(output / 'features.npy', x, allow_pickle=False)
    atomic_json(output / 'provenance.json', provenance)
    schema = dict(revision=SCHEMA, names=FEATURE_NAMES, families=FAMILIES,
                  missing='zero plus explicit missing indicator', transform='signed_log then train-only standardization',
                  correlated='family measurements are not independent physical votes', vision_enabled=False)
    manifest = dict(schema=schema, events=events, sessions=list(EVENTS), folds=folds(EVENTS),
                    evaluation_scope='REPEATED_SESSION_CV_DEVELOPMENT_SELECTION_NOT_UNTOUCHED_VALIDATION',
                    source_hashes=hashes, feature_sha256=digest(output / 'features.npy'),
                    provenance_sha256=digest(output / 'provenance.json'), code_revision=get_git_commit(),
                    builder_sha256=digest(Path(__file__)),
                    exclusions=[dict(event='D01', reason='original raw unavailable; stale source forbidden'),
                                dict(event='S01:11', reason='incomplete cutoff'),
                                dict(event='H10/H20', reason='outside fixed chronological 45-event comparison')],
                    live_authority=False, labels_used_as_features=False)
    manifest['dataset_hash'] = canonical_hash(manifest)
    atomic_json(output / 'dataset.json', manifest)
    atomic_json(output / 'baseline.json', evaluate(events, {e['event']: e['baseline_xy'] for e in events}))
    return manifest


def load(root):
    root = Path(root); guard(root)
    data = read(root / 'dataset.json')
    expected = data.pop('dataset_hash')
    if canonical_hash(data) != expected:
        raise ValueError('Dataset manifest changed')
    data['dataset_hash'] = expected
    if data['schema']['revision'] != SCHEMA or tuple(data['schema']['names']) != FEATURE_NAMES:
        raise ValueError('Unknown feature schema')
    for e in data['events']:
        guard(e['event']); guard(e['session']); guard(e['source_session']); guard(e['source_root'])
        if e['session'] not in EVENTS or e['truth'] not in ('HIT', 'NO_IMPACT', 'UNKNOWN'):
            raise ValueError('Forbidden cohort/truth')
        if e['truth']=='HIT' and (e['gt_xy'] is None or len(e['gt_xy'])!=2 or not np.isfinite(e['gt_xy']).all()):
            raise ValueError('Missing/nonfinite physical truth')
        if e['truth']!='HIT' and e['gt_xy'] is not None:
            raise ValueError('Non-impact/unknown truth cannot carry a positive label')
        for frame in e['frames']: guard(frame['frame_id'])
    for source in data['source_hashes']: guard(source)
    if len({e['event'] for e in data['events']}) != len(data['events']):
        raise ValueError('Duplicate event lineage')
    lineage = {}
    for e in data['events']:
        if lineage.setdefault(e['source_session'], e['session']) != e['session']:
            raise ValueError('Source session aliases would cross folds')
    if data['folds'] != folds(data['sessions']):
        raise ValueError('Invalid session folds')
    for name, key in [('features.npy', 'feature_sha256'), ('provenance.json', 'provenance_sha256')]:
        if digest(root / name) != data[key]:
            raise ValueError('Dataset cache changed')
    x = np.load(root / 'features.npy', allow_pickle=False)
    indices = [i for e in data['events'] for i in e['indices']]
    if indices != list(range(len(x))) or x.shape != (len(indices), len(FEATURE_NAMES)) or not np.isfinite(x).all():
        raise ValueError('Invalid feature rows')
    return data, x


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def evaluate(events, predictions):
    """None means UNKNOWN. Never counts abstention as correct control rejection."""
    if set(predictions) != {e['event'] for e in events}:
        raise ValueError('Incomplete/foreign prediction events')
    rows = []
    for e in events:
        xy = predictions[e['event']]
        if xy is not None and (len(xy) != 2 or not np.isfinite(xy).all()):
            raise ValueError('Invalid selected XY')
        err = distance(xy, e['gt_xy']) if xy is not None and e['truth'] == 'HIT' else None
        nearest = min((distance(c['xy'], e['gt_xy']) for c in e['candidates']), default=None) if e['truth'] == 'HIT' else None
        rows.append(dict(event=e['event'], session=e['session'], truth=e['truth'], selected_xy=xy,
                         error=err, oracle_error=nearest, state='UNKNOWN' if xy is None else 'PHYSICAL_HIT',
                         protected=e.get('protected', False), caveats=e.get('caveats', [])))
    def stats(rr):
        hits = [r for r in rr if r['truth'] == 'HIT']; controls = [r for r in rr if r['truth'] == 'NO_IMPACT']
        errors = [r['error'] for r in hits if r['error'] is not None]
        counts = {str(t): sum(r['error'] is not None and r['error'] <= t for r in hits) for t in (5, 10, 20, 42)}
        return dict(events=len(rr), impacts=len(hits), controls=len(controls), selected=counts,
                    selected5_rate=counts['5']/len(hits) if hits else None,
                    oracle={str(t): sum(r['oracle_error'] is not None and r['oracle_error'] <= t for r in hits) for t in (5,10,20,42)},
                    median_error=float(np.median(errors)) if errors else None,
                    p95_error=float(np.percentile(errors,95)) if errors else None,
                    error_denominator='emitted physical events only; see UNKNOWN count',
                    catastrophic_gt100=sum(x > 100 for x in errors),
                    control_false_emissions=sum(r['state'] == 'PHYSICAL_HIT' for r in controls),
                    control_unknown=sum(r['state'] == 'UNKNOWN' for r in controls), validated_control_rejections=0,
                    unknown=sum(r['state'] == 'UNKNOWN' for r in rr), unknown_rate=sum(r['state']=='UNKNOWN' for r in rr)/max(1,len(rr)),
                    protected_failures=[r['event'] for r in rr if r['protected'] and (r['error'] is None or r['error'] > 42)])
    sessions = {s: stats([r for r in rows if r['session']==s]) for s in sorted({r['session'] for r in rows})}
    result = stats(rows)
    result.update(per_session=sessions, worst_session_selected5=min((s['selected5_rate'] for s in sessions.values() if s['impacts']), default=None), per_event=rows)
    return result


def validate_upstream(prediction, event, fold):
    """Contract only: nested upstream exclusion, including the outer held session.

    Passing metadata alone is not model verification. Fusion remains disabled.
    """
    for s in prediction.get('training_sessions', []): guard(s)
    used = set(prediction.get('training_sessions', []))
    if not used or not prediction.get('model_sha256') or not prediction.get('dataset_hash'):
        raise ValueError('Missing upstream lineage')
    if prediction.get('event') != event['event'] or prediction.get('session') != event['session']:
        raise ValueError('Upstream event mismatch')
    if used & (set(fold['held']) | set(fold['dev']) | {event['session']}):
        raise ValueError('Upstream session leakage')
    if not used <= set(fold['train']):
        raise ValueError('Upstream trained outside permitted outer training fold')
