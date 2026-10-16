"""Source-bound crop adapter for physical_trace_label, never a detector/label source.

Only original saved gray tiles or native BGR review frames enter the UI. Capture files are read
only. Exact labels extend the existing camera_x/y, space, quality convention in a
separate versioned document; UNKNOWN never means no physical shot.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from automation.accuracy_physical_dataset import guard_path
from automation.object_physical_replay import safe_file
from automation.physical_trace_label import AnnotationDataError, display_to_camera, fit_display

SCHEMA = 'paired_exact_camera_ground_truth_v1'
MANIFEST_SCHEMA = 'paired_original_crop_manifest_v1'
REVIEW_LABEL_SCHEMA = 'physical_gt_review_exact_camera_v1'


def checked(path):
    if 's03' in str(path).lower():
        raise PermissionError('S03 validation source forbidden')
    guard_path(path)
    p = Path(path).resolve()
    guard_path(p)
    if 's03' in str(p).lower():
        raise PermissionError('S03 validation source forbidden')
    return p


def digest(path):
    return hashlib.sha256(checked(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_capture(capture):
    """Whitelist image metadata only; never discover/load candidate sidecars.

    Trace includes detector fields but NONE are returned to the label UI. Event
    IDs reuse existing log identity parsing, never planned shot ordinal/aim.
    """
    from automation.object_candidate_reconciliation import paired_event_ids
    root = checked(capture)
    hashes = {}
    def read(path):
        path = checked(path)
        hashes[str(path)] = digest(path)
        return json.loads(path.read_text())
    play = read(safe_file(root, 'play.json'))
    if play.get('mode') != 'OBJECT_HIT_GAMEPLAY_ACCEPTANCE':
        raise AnnotationDataError('paired play-can capture required')
    log = safe_file(root, 'runtime.log'); hashes[str(log)] = digest(log)
    ids = paired_event_ids(root, log.read_text().splitlines())
    shots = []
    for sid in ids:
        path = safe_file(root, f'object_crops/shots/shot_{sid:08d}/trace.json')
        shot = dict(shot_id=sid, event_id=sid, availability='UNAVAILABLE', frames=[], tiles=[],
                    reason='IMAGE_DATA_UNAVAILABLE')
        if not path.exists():
            shots.append(shot); continue
        t = read(path)
        if t.get('shot_id') != sid or t.get('coordinate_space') != 'camera':
            raise AnnotationDataError('trace identity/coordinate space mismatch')
        shot.update(trace_path=str(path), trace_sha256=hashes[str(path)], peak_ts=t['peak_ts'])
        crops = t.get('object_evidence_crops', [])
        if not crops:
            shots.append(shot); continue
        size = t.get('context', {}).get('runtime_geometry', {}).get('calibration', {}).get('camera_frame_size')
        if not size or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise AnnotationDataError('missing/invalid canonical camera size')
        shot['full_camera_size'] = size
        frames = {}; entries = []; seen = set()
        regions = t.get('context', {}).get('shot_game_context', {}).get('camera_hit_regions', [])
        # Context inventory only, not displayed; no scores/roles/support values.
        shot['frozen_camera_regions'] = [{k:r.get(k) for k in ('object_id','x','y','width','height')} for r in regions]
        for c in crops:
            checked(path.parent / c['path'])
            file = checked(safe_file(path.parent, c['path']))
            a = np.load(file, allow_pickle=False)
            box = c.get('camera_bbox'); stamp = c.get('timestamp'); kind = c.get('kind')
            if (not box or len(box) != 4 or any(not isinstance(v,(int,float)) or not math.isfinite(v) or int(v)!=v for v in box)
                    or not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1])):
                raise AnnotationDataError('missing/invalid native crop origin')
            box = list(map(int, box))
            if (a.ndim != 2 or a.dtype != np.uint8 or list(a.shape) != c.get('shape')
                    or a.shape != (box[3]-box[1], box[2]-box[0]) or str(a.dtype) != c.get('dtype')
                    or hashlib.sha256(a.tobytes()).hexdigest() != c.get('sha256')):
                raise AnnotationDataError('source crop hash/shape/dtype mismatch')
            if kind not in ('pre', 'post') or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                raise AnnotationDataError('missing crop time/kind')
            if (kind == 'pre' and stamp >= t['peak_ts']) or (kind == 'post' and stamp < t['peak_ts']):
                raise AnnotationDataError('PRE/POST violates recorded audio ordering')
            acq = c.get('acquisition', {})
            # Same timestamp is same saved camera observation, not independent tile frames.
            frame_id = f'{sid}:{stamp:.9f}'
            if acq.get('frame_shape') and acq['frame_shape'][:2] != [size[1],size[0]]:
                raise AnnotationDataError('acquisition/canonical frame dimensions disagree')
            identity = {k:acq.get(k) for k in ('stream_id','sequence','receive_wall_ts','rotation',
                                               'mirror_horizontal','mirror_vertical','frame_shape')}
            if frame_id in frames and (frames[frame_id]['acquisition'] != identity or frames[frame_id]['kind'] != kind):
                raise AnnotationDataError('conflicting acquisition at same camera timestamp')
            frames.setdefault(frame_id, dict(frame_id=frame_id, timestamp=stamp, kind=kind,
                              acquisition=identity, source_full_frame_hash=None, crops=[]))
            if (frame_id, tuple(box)) in seen:
                raise AnnotationDataError('duplicate crop/frame identity')
            seen.add((frame_id, tuple(box)))
            hashes[str(file)] = digest(file)
            entry = dict(path=str(file.relative_to(root)), source_image_sha256=hashes[str(file)],
                pixel_sha256=c['sha256'], timestamp=stamp, frame_id=frame_id, kind=kind,
                crop_origin=box[:2], crop_size=[a.shape[1], a.shape[0]], camera_bbox=box,
                full_camera_size=size, shape=list(a.shape), dtype=str(a.dtype), acquisition=identity)
            entries.append(entry); frames[frame_id]['crops'].append(entry)
        shot['frames'] = sorted(frames.values(), key=lambda f:f['timestamp'])
        shot['tiles'] = [dict(camera_bbox=list(box), entries=sorted(
            [e for e in entries if e['camera_bbox'] == list(box)], key=lambda e:e['timestamp']))
            for box in sorted({tuple(e['camera_bbox']) for e in entries})]
        shot['availability'] = 'AVAILABLE' if any(f['kind']=='post' for f in shot['frames']) else 'UNAVAILABLE'
        shot['reason'] = None if shot['availability']=='AVAILABLE' else 'NO_SAVED_POST'
        shots.append(shot)
    return dict(schema=MANIFEST_SCHEMA, capture_path=str(root), capture_play_sha256=hashes[str(root/'play.json')],
                input_hashes=hashes, shots=shots, candidate_overlay=False,
                scope='native saved grayscale crops; no full-board reconstruction or physical labels')


def source_image(manifest, entry):
    checked(Path(manifest['capture_path']) / entry['path'])
    p = checked(safe_file(Path(manifest['capture_path']), entry['path']))
    if digest(p) != entry['source_image_sha256']:
        raise AnnotationDataError('source image file hash mismatch')
    a = np.load(p, allow_pickle=False)
    if (list(a.shape) != entry['shape'] or str(a.dtype) != entry['dtype']
            or hashlib.sha256(a.tobytes()).hexdigest() != entry['pixel_sha256']):
        raise AnnotationDataError('source pixels mismatch')
    return a


def load_gt_review(capture):
    """Adapt separate native BGR frame records to the SAME original-image UI.

    Only whitelisted review image metadata is read. No trace candidates, game
    geometry, calibration or detector outputs are loaded. Roles stay explicit;
    a GT_SETTLED frame is never represented as causal live POST evidence.
    """
    from automation.physical_gt_capture import SCHEMA as capture_schema, FRAME_SCHEMA, STATES, PRE_RULE, pre_provenance, TIMING_SCHEMA, frame_timing
    from types import SimpleNamespace
    root = checked(capture); review = root/'gt_review'; hashes = {}
    def read(p):
        p=checked(p); hashes[str(p)]=digest(p)
        return json.loads(p.read_text())
    config=read(review/'config.json'); manifest=read(review/'manifest.json')
    play=safe_file(root,'play.json'); hashes[str(play)]=digest(play)
    if (config.get('schema') != capture_schema or config.get('live_authority') is not False
            or config.get('capture_path') != str(root) or config.get('capture_play_sha256') != hashes[str(play)]
            or any(manifest.get(k) != config.get(k) for k in config)):
        raise AnnotationDataError('review capture identity/role contract mismatch')
    shots=[]; seen=set()
    for row in manifest.get('shots',[]):
        sid=row['shot_id']
        if type(sid) is not int or sid<=0 or sid in seen or row.get('event_id')!=sid:
            raise AnnotationDataError('invalid/duplicate review shot identity')
        seen.add(sid); entries=[]; pre_checks=[]
        for item in row['frames']:
            # An interrupted queued write is recoverable only if BOTH immutable
            # pixels and a complete, hash-valid metadata sidecar actually exist.
            p=checked(safe_file(review,item['path'])); meta_path=p.with_suffix('.json')
            if not p.exists() or not meta_path.exists(): continue
            meta=read(meta_path)
            if item.get('metadata_sha256') and hashes[str(meta_path)]!=item['metadata_sha256']:
                raise AnnotationDataError('review frame metadata hash mismatch')
            role=meta.get('role'); state=meta.get('projector_state'); acq=meta.get('acquisition',{})
            if (meta.get('schema') != FRAME_SCHEMA or meta.get('live_authority') is not False
                    or meta.get('candidate_overlay') is not False or meta.get('shot_id') != sid
                    or meta.get('event_id')!=sid or meta.get('capture_path')!=str(root)
                    or meta.get('source_filename')!=item['path'] or role!=item['role']
                    or role not in ('PRE','LIVE_POST','GT_SETTLED') or meta.get('status')!='SAVED'
                    or state!=item['projector_state']):
                raise AnnotationDataError('review frame source/role mismatch')
            stamp=meta.get('frame_timestamp')
            if not isinstance(stamp,(int,float)) or not math.isfinite(stamp):
                raise AnnotationDataError('review frame time missing')
            if (role=='PRE' and stamp>=row['peak_ts']) or (role=='LIVE_POST' and not row['peak_ts']<=stamp<=row['live_closed_ts']):
                raise AnnotationDataError('review PRE/LIVE_POST ordering mismatch')
            if config.get('timing_schema')==TIMING_SCHEMA:
                expected=frame_timing(SimpleNamespace(timestamp=stamp,acquisition=acq),row['peak_ts'],row.get('audio_acquisition'),role)
                if meta.get('timing')!=expected or ('timing' in item and item['timing']!=expected):
                    raise AnnotationDataError('review timing provenance mismatch')
            if role=='PRE':
                proof=pre_provenance(SimpleNamespace(timestamp=stamp,acquisition=acq),row['peak_ts'],row.get('audio_acquisition'))
                if config.get('pre_selection_rule')==PRE_RULE:
                    if not proof['pre_cutoff_verified'] or any(meta.get(k)!=v for k,v in proof.items()):
                        raise AnnotationDataError('review PRE cutoff/provenance mismatch')
                else:
                    # Legacy "before audio" alone never certified pre-damage.
                    proof.update(pre_cutoff_verified=False,reason='LEGACY_PRE_WITHOUT_SAFETY_GUARD')
                pre_checks.append(proof)
            if role=='GT_SETTLED':
                if (state not in STATES or meta.get('projector_rgb')!=list(STATES[state])
                        or meta.get('presentation_monotonic',0)<row['live_closed_monotonic']
                        or acq.get('read_start_monotonic',0)<meta['presentation_monotonic']+meta['settle_delay_s']):
                    raise AnnotationDataError('GT frame predates terminal live/presentation barrier')
            elif state!='GAME_UNMODIFIED':
                raise AnnotationDataError('neutral frame cannot masquerade as live evidence')
            a=np.load(p,allow_pickle=False); hashes[str(p)]=digest(p)
            if (a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or list(a.shape)!=meta.get('shape')
                    or meta.get('full_camera_size')!=[a.shape[1],a.shape[0]] or meta.get('color_space')!='BGR'
                    or (acq.get('frame_shape') is not None and acq['frame_shape']!=list(a.shape))
                    or hashes[str(p)]!=meta.get('image_sha256')
                    or hashlib.sha256(a.tobytes()).hexdigest()!=meta.get('pixel_sha256')):
                raise AnnotationDataError('review full-frame hash/shape/color mismatch')
            size=meta['full_camera_size']
            entries.append(dict(path=str(p.relative_to(root)),source_image_sha256=hashes[str(p)],
                pixel_sha256=meta['pixel_sha256'],timestamp=stamp,frame_id=f'{sid}:{acq.get("stream_id")}:{acq.get("sequence")}:{stamp:.9f}',
                kind='pre' if role=='PRE' else 'post',role=role,projector_state=state,live_authority=False,
                crop_origin=[0,0],crop_size=size,full_camera_size=size,camera_bbox=[0,0,*size],
                shape=meta['shape'],dtype='uint8',color_space='BGR',acquisition=acq,
                orientation=meta.get('orientation'),metadata_path=str(meta_path.relative_to(root)),
                metadata_sha256=hashes[str(meta_path)]))
        sizes={tuple(e['full_camera_size']) for e in entries}
        if len(sizes)>1: raise AnnotationDataError('canonical camera size changed during event')
        orientation={tuple(e['acquisition'].get(k) for k in ('rotation','mirror_horizontal','mirror_vertical')) for e in entries}
        if len(orientation)>1: raise AnnotationDataError('canonical camera orientation changed during event')
        available=any(e['role']=='GT_SETTLED' for e in entries)
        shots.append(dict(shot_id=sid,event_id=sid,peak_ts=row['peak_ts'],
            availability='AVAILABLE' if available else 'UNAVAILABLE',
            reason=None if available else 'NO_SAVED_GT_SETTLED',
            trace_path=str(review/'manifest.json'),trace_sha256=hashes[str(review/'manifest.json')],
            full_camera_size=list(next(iter(sizes))) if sizes else None,
            frames=[dict(frame_id=e['frame_id'],timestamp=e['timestamp'],kind=e['kind'],role=e['role'],crops=[e]) for e in entries],
            tiles=[dict(camera_bbox=entries[0]['camera_bbox'],entries=entries)] if entries else [],
            pre_causality=pre_checks,
            capture_status=row['status'],missing=row['missing']))
    return dict(schema='physical_gt_original_frame_manifest_v1',exact_gt_schema=REVIEW_LABEL_SCHEMA,
        capture_path=str(root),capture_play_sha256=hashes[str(play)],input_hashes=hashes,shots=shots,
        candidate_overlay=False,scope='Original native BGR full frames; GT_SETTLED for HUMAN REVIEW ONLY, never live evidence')


def assert_inputs(manifest):
    for p, expected in manifest['input_hashes'].items():
        if digest(p) != expected:
            raise AnnotationDataError(f'capture changed since preparation: {p}')


def crop_click(click, origin, scale, entry):
    """Actual display inverse + native origin; no calibration, XY prior or snap."""
    if any(k not in entry for k in ('crop_origin','crop_size','full_camera_size')):
        raise AnnotationDataError('crop metadata required')
    width, height = entry['crop_size']
    local = display_to_camera(click, origin, scale, (height, width))
    if local is None or not all(math.isfinite(v) for v in local):
        raise AnnotationDataError('click outside original source crop')
    xy = [entry['crop_origin'][i]+local[i] for i in (0,1)]
    if any(not (0 <= xy[i] <= entry['full_camera_size'][i]-1) for i in (0,1)):
        raise AnnotationDataError('camera XY outside canonical frame')
    return dict(crop_local_xy=list(local), camera_xy=xy, display_xy=list(click),
                display_origin=list(origin), display_scale=list(scale) if isinstance(scale,tuple) else [scale,scale])


def view_transform(entry, viewport, zoom=1., pan=(0.,0.)):
    """Reuse fit_display with actual rounded per-axis scale, including zoom/pan."""
    w,h = entry['crop_size']
    _, size, _ = fit_display((h,w), viewport)
    size = tuple(max(1, round(v*zoom)) for v in size)
    origin = tuple(round((viewport[i]-size[i])/2 + pan[i]) for i in (0,1))
    return (size[0]/w, size[1]/h), size, origin


def unknown_row(sid, reason='UNREVIEWED'):
    return dict(shot_id=sid, event_id=sid, physical_truth_status='UNKNOWN', physical_event='UNKNOWN',
                object_result='UNKNOWN', object_id=None, camera_xy=None, reason=reason)


def new_document(manifest):
    return dict(schema=manifest.get('exact_gt_schema',SCHEMA), capture_path=manifest['capture_path'],
        capture_play_sha256=manifest['capture_play_sha256'], revision=0, history=[],
        shots=[unknown_row(s['shot_id'], s['reason'] or 'UNREVIEWED') for s in manifest['shots']])


def make_label(manifest, shot, entry, click, inspected_pre, *, note='', independent_observation=None):
    if entry.get('kind') != 'post':
        raise AnnotationDataError('final label click must be on original POST')
    review = manifest.get('exact_gt_schema') == REVIEW_LABEL_SCHEMA
    if independent_observation is not None and (not review or not str(independent_observation).strip()):
        raise AnnotationDataError('independent physical observation requires GT review and a non-empty operator note')
    if not inspected_pre and independent_observation is None:
        raise AnnotationDataError('inspect at least one original PRE before certifying a new hole')
    if review and entry.get('role')!='GT_SETTLED':
        raise AnnotationDataError('final review label requires original GT_SETTLED frame')
    row = dict(shot_id=shot['shot_id'], event_id=shot['shot_id'], physical_truth_status='EXACT_CAMERA_XY',
        physical_event='SINGLE_IMPACT', object_result='UNKNOWN', object_id=None,
        space='camera', camera_x=click['camera_xy'][0], camera_y=click['camera_xy'][1], quality='precise',
        operator_method='MANUAL_VISUAL', candidate_overlay=False, clicked_view='ORIGINAL_GT_SETTLED' if review else 'ORIGINAL_POST',
        source=deepcopy(entry), trace_sha256=shot['trace_sha256'], pre_inspected=deepcopy(inspected_pre),
        selected_post_timestamp=entry['timestamp'], created_at=utc(), note=note, **click)
    if review:
        row['review_truth_basis']=dict(schema='physical_gt_manual_basis_v1',
            method='INDEPENDENT_PHYSICAL_OBSERVATION' if independent_observation is not None else 'MANUAL_PRE_POST_INSPECTION',
            observation_note=independent_observation,causal_pre_certified=False,
            causal_pre_reason='MANUAL_XY_DOES_NOT_CERTIFY_PRE_CAUSALITY',
            pre_causality=deepcopy(shot.get('pre_causality',[])))
    return row


def validate_document(doc, manifest):
    review = manifest.get('exact_gt_schema') == REVIEW_LABEL_SCHEMA
    if (doc.get('schema') != manifest.get('exact_gt_schema',SCHEMA) or doc.get('capture_path') != manifest['capture_path']
            or doc.get('capture_play_sha256') != manifest['capture_play_sha256']):
        raise AnnotationDataError('exact GT schema/capture identity mismatch')
    if type(doc.get('revision')) is not int or doc['revision'] < 0 or not isinstance(doc.get('history'),list):
        raise AnnotationDataError('versioned revision/history required')
    shots = {s['shot_id']:s for s in manifest['shots']}; seen=set()
    for row in doc.get('shots', []):
        sid = row.get('shot_id')
        if type(sid) is not int or sid not in shots or sid in seen or row.get('event_id') != sid:
            raise AnnotationDataError('unknown/duplicate label event identity')
        seen.add(sid); shot=shots[sid]
        if row.get('physical_truth_status') == 'UNKNOWN':
            if (row.get('camera_xy') is not None or row.get('camera_x') is not None or row.get('camera_y') is not None
                    or row.get('physical_event') != 'UNKNOWN'):
                raise AnnotationDataError('UNKNOWN may not contain physical XY/event claims')
            continue
        if (row.get('physical_truth_status') != 'EXACT_CAMERA_XY' or row.get('physical_event') != 'SINGLE_IMPACT'
                or row.get('operator_method') != 'MANUAL_VISUAL' or row.get('candidate_overlay') is not False
                or row.get('clicked_view') != ('ORIGINAL_GT_SETTLED' if review else 'ORIGINAL_POST') or row.get('quality') != 'precise'
                or row.get('space') != 'camera' or not row.get('created_at')):
            raise AnnotationDataError('independent original POST manual label contract required')
        if row.get('object_result') != 'UNKNOWN' or row.get('object_id') is not None:
            raise AnnotationDataError('object labels belong to separate operator truth, not exact XY inference')
        entries = [c for f in shot['frames'] for c in f['crops']]
        entry = row.get('source')
        if (entry not in entries or entry.get('kind') != 'post'
                or (review and entry.get('role')!='GT_SETTLED')
                or row.get('trace_sha256') != shot.get('trace_sha256')
                or row.get('selected_post_timestamp') != entry['timestamp']):
            raise AnnotationDataError('label source is not bound to this event original POST')
        pre = row.get('pre_inspected')
        basis=row.get('review_truth_basis')
        independent=False
        if basis is not None:
            if (not review or basis.get('schema')!='physical_gt_manual_basis_v1'
                    or basis.get('causal_pre_certified') is not False
                    or basis.get('causal_pre_reason')!='MANUAL_XY_DOES_NOT_CERTIFY_PRE_CAUSALITY'
                    or basis.get('pre_causality')!=shot.get('pre_causality',[])):
                raise AnnotationDataError('manual label cannot certify PRE causality')
            independent=basis.get('method')=='INDEPENDENT_PHYSICAL_OBSERVATION'
            if ((independent and not isinstance(basis.get('observation_note'),str))
                    or (independent and not basis['observation_note'].strip())
                    or (not independent and (basis.get('method')!='MANUAL_PRE_POST_INSPECTION' or basis.get('observation_note') is not None))):
                raise AnnotationDataError('explicit manual observation basis required')
        if (not isinstance(pre,list) or (not pre and not independent)
                or any(e not in entries or e['kind'] != 'pre' or e['camera_bbox'] != entry['camera_bbox'] for e in pre)):
            raise AnnotationDataError('original matching PRE inspection identity required')
        for k in ('display_xy', 'display_origin', 'display_scale', 'crop_local_xy', 'camera_xy'):
            if k not in row or len(row[k]) != 2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in row[k]):
                raise AnnotationDataError('finite recorded click transform required')
        expected = crop_click(row['display_xy'], tuple(row['display_origin']), tuple(row['display_scale']), entry)
        if (any(not np.allclose(row[k], expected[k], rtol=0, atol=1e-9) for k in ('crop_local_xy','camera_xy'))
                or [row.get('camera_x'),row.get('camera_y')] != row['camera_xy']):
            raise AnnotationDataError('saved XY differs from exact display/crop transform')
    if seen != set(shots):
        raise AnnotationDataError('every event must have explicit exact label or UNKNOWN')
    return doc


def atomic_document(path, doc, manifest):
    path = checked(path)
    if path == Path(manifest['capture_path']) or path.is_relative_to(Path(manifest['capture_path'])):
        raise AnnotationDataError('never write labels into original capture')
    if not path.is_relative_to(Path('/data/skjutbana')):
        raise AnnotationDataError('label output must be under /data/skjutbana')
    assert_inputs(manifest); validate_document(doc, manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.exact_gt_', dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(doc,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
        validate_document(json.loads(Path(tmp).read_text()),manifest)
        os.replace(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def save_row(path, manifest, row, expected_revision=None):
    path=checked(path)
    doc=json.loads(path.read_text()) if path.exists() else new_document(manifest)
    validate_document(doc,manifest)
    if expected_revision is not None and doc['revision'] != expected_revision:
        raise AnnotationDataError('labels changed in another viewer; reopen before saving')
    before=next(r for r in doc['shots'] if r['shot_id']==row['shot_id'])
    doc['history'].append(dict(revision=doc['revision'],replaced_at=utc(),previous=deepcopy(before)))
    doc['shots']=[row if r['shot_id']==row['shot_id'] else r for r in doc['shots']]
    doc['revision']+=1
    atomic_document(path,doc,manifest)
    return doc


def label_shots(manifest):
    """Adapt to existing _run_target_ui's shot structure, without causal_overlay."""
    return [dict(trace={'shot_id':s['shot_id']}, paired=s) for s in manifest['shots']]


def ui_frames(shot, tile_index, frame_index, pre_index):
    s=shot['paired']
    if s['availability']!='AVAILABLE':
        return None,None,0,0
    tile=s['tiles'][tile_index % len(s['tiles'])]
    posts=[e for e in tile['entries'] if e['kind']=='post']
    pres=[e for e in tile['entries'] if e['kind']=='pre']
    post=posts[frame_index % len(posts)] if posts else None
    pre=pres[pre_index % len(pres)] if pres else None
    return post,pre,len(posts),len(pres)


def prepare_package(manifest, output):
    """Immutable, candidate-free review aids, not new labels or a camera mosaic."""
    import cv2
    out=checked(output); root=Path(manifest['capture_path'])
    if not out.is_relative_to(Path('/data/skjutbana')) or out==root or out.is_relative_to(root):
        raise AnnotationDataError('new external /data/skjutbana package required')
    out.mkdir(parents=True,exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'truth_status.json').write_text(json.dumps(new_document(manifest),indent=2)+'\n')
    for shot in manifest['shots']:
        directory=out/f"shot_{shot['shot_id']:03d}";directory.mkdir()
        (directory/'frames.json').write_text(json.dumps(shot,indent=2)+'\n')
        if shot['availability']!='AVAILABLE':
            (directory/'UNAVAILABLE.txt').write_text('IMAGE DATA UNAVAILABLE. No inferred frames or label.\n');continue
        for i,tile in enumerate(shot['tiles']):
            entries=tile['entries'];canvas=np.full((len(entries)*160,660,3),32,np.uint8)
            for j,e in enumerate(entries):
                a=source_image(manifest,e);scale=min(620/a.shape[1],120/a.shape[0],1.)
                if a.ndim==2:a=cv2.cvtColor(a,cv2.COLOR_GRAY2BGR)
                thumb=cv2.resize(a,(max(1,round(a.shape[1]*scale)),max(1,round(a.shape[0]*scale))),interpolation=cv2.INTER_NEAREST)
                canvas[j*160+30:j*160+30+thumb.shape[0],10:10+thumb.shape[1]]=thumb
                cv2.putText(canvas,f"{e.get('role',e['kind'].upper())} {e.get('projector_state','')} {e['timestamp']:.6f}",(10,j*160+20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            if not cv2.imwrite(str(directory/f'tile_{i+1:02d}.png'),canvas):raise OSError('contact sheet write failed')
    (out/'README.md').write_text('''# Independent physical label preparation
Original saved grayscale tiles only. No candidate coordinates, support, ranking,
heatmaps or detector boxes are rendered. Contact sheets are aids, never GT.
Some views cover only part of the board; missing full-frame pixels are not reconstructed.
Frames are grouped by recorded timestamp/acquisition identity, not counted per tile.
PRE means recorded before audio, not certified before physical damage. Verify a NEW
hole against PRE; otherwise UNKNOWN. No physical labels were generated by preparation.

Start the existing label UI with --paired-object-crops --capture CAPTURE --output
THIS_DIRECTORY/exact_camera_ground_truth.json. Original POST clicks only; Enter
confirms a reviewed exact coordinate. U/S marks UNKNOWN; N means no visible hole
and remains UNKNOWN (never NO_PHYSICAL_SHOT). Arrow left/right selects event;
up/down selects POST; brackets select tile; comma/period selects PRE; 3=POST,
4=PRE, B=blink, +/-=zoom, mouse right-drag=pan, 0=fit, R=cancel pending click.
Inspect PRE first, return to original POST and click independently. Enter saves,
Q exits; previous saves persist with revision history. Reopen to revise a label.
No automatic object/discharge labels and no calibration/game coordinate conversion.
Event with missing images cannot receive exact XY. Source hashes are checked on save.
''')
    assert_inputs(manifest)
    if manifest.get('exact_gt_schema')==REVIEW_LABEL_SCHEMA:
        path=out/'README.md'
        text=path.read_text().replace('Original saved grayscale tiles only.','Original saved full native BGR camera frames only.')
        text=text.replace('--paired-object-crops','--gt-review').replace('Original POST clicks only','Original GT_SETTLED clicks only')
        text+='\nGT_SETTLED WHITE/BLACK/MID_GRAY are post-decision human review, NEVER live evidence. LIVE_POST is inspection only. Full-frame crop origin is (0,0).\n'
        path.write_text(text)
    return out
