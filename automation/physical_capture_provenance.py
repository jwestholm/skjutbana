"""Small D02 provenance sidecar/preflight; never starts capture or invents notes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from automation.physical_collection import load_plan, preflight
from automation.physical_label_reset import resolve_session

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO/'research/physical_capture_plans/D02.json'
GEOMETRY_KEYS = ('camera_calibration','camera','viewport','scanport','scanport_rect','content_rect','range_projection')
WORKING_KEYS = ('analysis_roi_crop_v2221_enabled','analysis_playfield_edge_guard_screen_px',
                'analysis_crop_padding_camera_px','analysis_fallback_edge_guard_camera_px')


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_settings(path):
    settings=json.loads(Path(path).read_text())
    # Do not copy unrelated settings (e.g. device credentials) into research.
    return {k:settings.get(k) for k in GEOMETRY_KEYS}


def saved_configuration(settings_path=REPO/'content/settings.json', ai_settings=REPO/'content/ai/settings.json'):
    """Read-only setup aid; saved configuration is not observed physical geometry."""
    import config
    ai=json.loads(Path(ai_settings).read_text());geometry=geometry_settings(settings_path)
    return dict(status='SAVED_CONFIGURATION_ONLY_NOT_RUNTIME_VERIFICATION',geometry=geometry,
        calibration_sha256=hashlib.sha256(json.dumps(geometry['camera_calibration'],sort_keys=True).encode()).hexdigest(),
        configured_framebuffer_size_px=[config.SCREEN_WIDTH,config.SCREEN_HEIGHT],
        working_geometry_settings={k:ai.get(k,'RUNTIME_DEFAULT') for k in WORKING_KEYS},
        runtime_geometry='Per-frame AnalysisGeometry; export recorded window_debug with geometry-postflight after capture',
        semantics='Use as a setup aid; independently observe board bounds, seam and actual camera/display configuration')


def template(output):
    output.mkdir(parents=True,exist_ok=False)
    write(output/'plan.json',load_plan(PLAN))
    write(output/'geometry_notes.json',dict(schema='physical-capture-provenance-1',session='D02',
        observer=None,observed_at=None,board_revision=None,camera_frame_size_px=None,
        board_bounds_camera_px=None,board_bounds_basis=None,
        seam=dict(observable=None,points_camera_px=None,reason=None),
        working_geometry=dict(status='PENDING_RUNTIME',reason='Per-frame analysis crop; geometry-postflight exports recorded dimensions after capture'),
        display=dict(framebuffer_size_px=None,viewport_xywh=None,content_rect_viewport_xywh=None,
            monitor_description=None,projection_description=None,window_position_is_not_physical_truth=True),
        calibration_verified_unchanged_since_observation=False,
        notes='Fill actual observed geometry after setup. Bounds are physical board, not game target bounds.'))
    write(output/'event_notes.json',dict(schema='physical-independent-event-notes-1',session='D02',
        truth_source='human observation and actual hole labeling, never planned kind or game result',
        events=[dict(protocol_id=e['protocol_id'],actual_event_ids=[],observed_at=None,
            human_observation=None,discharge_observed=None,new_damage_observed=None,
            after_shot_delay_ms=None,status='NOT_OBSERVED') for e in load_plan(PLAN)['protocol_events']]))


def numbers(value,count,name):
    if not isinstance(value,list) or len(value)!=count or any(type(x) not in (int,float) or not math.isfinite(x) for x in value):
        raise ValueError(name+' needs '+str(count)+' finite numbers')


def validate_notes(notes):
    if notes.get('session')!='D02':raise ValueError('Only the D02 preparation is supported')
    for key in ('observer','observed_at','board_revision','board_bounds_basis'):
        if not isinstance(notes.get(key),str) or not notes[key].strip():raise ValueError('Missing human field: '+key)
    size=notes.get('camera_frame_size_px');numbers(size,2,'camera_frame_size_px')
    if min(size)<=0:raise ValueError('Camera dimensions must be positive')
    bounds=notes.get('board_bounds_camera_px')
    if not isinstance(bounds,list) or len(bounds)<4:raise ValueError('At least four physical board boundary points required')
    for point in bounds:
        numbers(point,2,'board point')
        if not (0<=point[0]<=size[0] and 0<=point[1]<=size[1]):raise ValueError('Board point outside camera frame')
    area=abs(sum(p[0]*q[1]-q[0]*p[1] for p,q in zip(bounds,bounds[1:]+bounds[:1])))/2
    if area<=0:raise ValueError('Degenerate physical board polygon')
    seam=notes.get('seam',{})
    if seam.get('observable') is True:
        points=seam.get('points_camera_px')
        if not isinstance(points,list) or len(points)<2:raise ValueError('Observable seam requires at least two camera points')
        for point in points:
            numbers(point,2,'seam point')
            if not (0<=point[0]<=size[0] and 0<=point[1]<=size[1]):raise ValueError('Seam point outside camera frame')
    elif seam.get('observable') is not False or not seam.get('reason'):
        raise ValueError('Record observed seam or explicitly explain why unavailable')
    display=notes.get('display',{})
    for key,n in [('framebuffer_size_px',2),('viewport_xywh',4),('content_rect_viewport_xywh',4)]:
        numbers(display.get(key),n,key)
        if min(display[key][-2:])<=0:raise ValueError('Display dimensions must be positive')
    for key in ('monitor_description','projection_description'):
        if not isinstance(display.get(key),str) or not display[key].strip():raise ValueError('Missing display '+key)
    if notes.get('calibration_verified_unchanged_since_observation') is not True:
        raise ValueError('Human geometry/calibration verification still pending')


def resolve_pre_capture(binding):
    meta=json.loads(Path(binding).read_text())
    if meta.get('planned_session_id')!='D02' or meta.get('session_class')!='DEVELOPMENT':
        raise ValueError('Only an explicit D02 development binding is supported')
    root=Path(meta['trace_root'])
    if not root.is_absolute():root=REPO/root
    if any(re.search(r'(^|_)S03(_|$)',part,re.I) for part in (*root.parts,*root.resolve().parts)):
        raise ValueError('Forbidden validation path; no session opened')
    return resolve_session(binding=binding,session='D02',require_shots=False)


def plan_matches_binding(plan_path,meta):
    plan=load_plan(plan_path)
    if plan.get('collection_plan_id')!=meta.get('collection_plan_id') or [r for r in plan['rows'] if r['session']=='D02']!=meta.get('rows'):
        raise ValueError('Plan differs from the bound D02 rows/identity')


def snapshot(binding, notes_path, output, settings_path=REPO/'content/settings.json', ai_settings=REPO/'content/ai/settings.json', plan_path=PLAN):
    notes=json.loads(notes_path.read_text());validate_notes(notes)
    root,meta=resolve_pre_capture(binding)
    plan_matches_binding(plan_path,meta)
    if (root/'shots').exists() and any((root/'shots').iterdir()):
        raise ValueError('D02 already has events; bind a fresh session, never reuse an old root')
    geometry=geometry_settings(settings_path)
    if not geometry['camera_calibration'] and not (geometry['scanport'] or geometry['scanport_rect']):
        raise ValueError('No saved calibration or scanport identity')
    write(output,dict(schema='physical-capture-provenance-1',session='D02',status='PRE_CAPTURE_PROVENANCE_ONLY',
        captured_at=datetime.now(timezone.utc).isoformat(),trace_root=str(root),binding_sha256=sha(binding),
        plan_sha256=sha(plan_path),saved_configuration=saved_configuration(settings_path,ai_settings),
        notes=notes,notes_sha256=sha(notes_path),geometry=geometry,
        geometry_sha256=hashlib.sha256(json.dumps(geometry,sort_keys=True).encode()).hexdigest(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_hashes={str(p.relative_to(REPO)):sha(p) for p in [REPO/'config.py',REPO/'src/engine/physical_trace.py',
            REPO/'src/engine/camera/demo_object_zone_extension.py']},
        settings_sha256=sha(settings_path),ai_settings_sha256=sha(ai_settings),
        semantics='Observed physical geometry plus saved configuration; not physical hit truth or proof of runtime display equivalence'))


def check(plan,binding,provenance,settings_path=REPO/'content/settings.json',ai_settings=REPO/'content/ai/settings.json'):
    root,meta=resolve_pre_capture(binding)
    ok,reasons=preflight(plan,'D02',root,binding=binding,settings_path=ai_settings)
    saved=json.loads(provenance.read_text())
    plan_matches_binding(plan,meta)
    validate_notes(saved['notes'])
    if saved.get('session')!='D02' or saved['binding_sha256']!=sha(binding) or Path(saved['trace_root']).resolve()!=root.resolve():
        reasons.append('Provenance belongs to another binding/session')
    if saved.get('plan_sha256')!=sha(plan):reasons.append('Capture plan changed since provenance snapshot')
    if saved['geometry']!=geometry_settings(settings_path):reasons.append('Saved geometry changed; observe and snapshot a fresh revision')
    if saved['ai_settings_sha256']!=sha(ai_settings):reasons.append('Detector/capture settings changed since provenance snapshot')
    if (root/'shots').exists() and any((root/'shots').iterdir()):reasons.append('Capture root is not empty; do not restart into it')
    for relative,expected in saved['source_hashes'].items():
        if sha(REPO/relative)!=expected:reasons.append('Source changed since provenance snapshot: '+relative)
    return not reasons,reasons


def geometry_postflight(binding,output):
    """Export actual bound D02 metadata without modifying capture or labels."""
    root,meta=resolve_pre_capture(binding)
    rows=[]
    shots=root/'shots'
    if shots.is_symlink():raise ValueError('Refusing a symlinked shots directory')
    for directory in sorted(shots.iterdir()) if shots.exists() else []:
        if not directory.name.startswith('shot_'):continue
        if directory.is_symlink():raise ValueError('Refusing a symlinked shot directory')
        path=directory/'trace.json'
        if path.is_symlink():raise ValueError('Refusing a symlinked trace')
        if not path.is_file():continue
        trace=json.loads(path.read_text());frames=trace.get('frames',[]);stages=[]
        for stage in trace.get('stages',[]):
            debug=stage.get('window_debug',{})
            geometry={k:v for k,v in debug.items() if k.startswith('v2221_')}
            if geometry:stages.append(dict(frame_ts=debug.get('frame_ts'),geometry=geometry))
        rows.append(dict(shot_id=trace['shot_id'],trace_sha256=sha(path),
            frame_shapes=sorted({tuple(f['shape']) for f in frames if isinstance(f.get('shape'),list)}),
            context=trace.get('context',{}),analysis_stages=stages,
            geometry_status='RECORDED' if stages else 'UNAVAILABLE_NO_RECORDED_ANALYSIS_GEOMETRY'))
    if not rows:raise ValueError('No captured D02 events; no postflight result created')
    write(output,dict(session='D02',trace_root=str(root),binding_sha256=sha(binding),events=rows,
        semantics='Recorded camera/crop metadata, not independent verification of physical board or display geometry'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('template');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('snapshot');p.add_argument('--binding',type=Path,required=True);p.add_argument('--notes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path,default=PLAN)
    p=sub.add_parser('check');p.add_argument('--plan',type=Path,required=True);p.add_argument('--binding',type=Path,required=True);p.add_argument('--provenance',type=Path,required=True)
    p=sub.add_parser('inspect');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('geometry-postflight');p.add_argument('--binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.command=='template':template(args.output);print('PLANNED ONLY:',args.output)
        elif args.command=='snapshot':snapshot(args.binding,args.notes,args.output,plan_path=args.plan);print('PRE-CAPTURE PROVENANCE:',args.output)
        elif args.command=='inspect':write(args.output,saved_configuration());print('SAVED CONFIGURATION ONLY:',args.output)
        elif args.command=='geometry-postflight':geometry_postflight(args.binding,args.output);print('RECORDED GEOMETRY:',args.output)
        else:
            ok,reasons=check(args.plan,args.binding,args.provenance)
            print('D02 PREFLIGHT PASS' if ok else 'D02 NOT READY');print('\n'.join(reasons));raise SystemExit(0 if ok else 2)
    except (ValueError,OSError,KeyError,TypeError) as exc:parser.exit(2,f'PROVENANCE NOT READY: {exc}\n')
