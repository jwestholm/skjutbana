"""Deterministic gameplay crop replay/analyzer; no labels or current hardware.

Extends object_physical_replay's live-query adapter for play-can captures. CPU
cutoffs are neutralized OFFLINE only: recorded deadlines remain recorded facts.
Missing pixels/mapping fail closed. Compact-only runs are analyzed, not replayed.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from copy import deepcopy
import cProfile
from dataclasses import fields
import csv
import hashlib
import json
import math
from pathlib import Path
import pstats
import subprocess
import time
from types import SimpleNamespace as S
from unittest.mock import patch

import numpy as np

from automation.physical_collection import _development_path
from automation.object_physical_replay import safe_file, patch_evidence
from src.engine.camera import object_physical_query as live
from src.engine.camera.analysis_geometry_v2221 import AnalysisGeometryV2221
from src.engine.input.hit_input import HitInput
from src.engine.input.object_hit_v2223 import (ObjectShotSnapshotV2223,
    GameHitRegionV240, CameraHitRegionV240, transform_camera_point_to_game)

CLOCK=time.perf_counter


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


class SavedCrops:
    """Full-camera shape metadata, with access ONLY to recorded exact slices."""
    def __init__(self,shape,crops):self.shape=tuple(shape);self.crops=crops

    def crop(self,geometry):
        bbox=(geometry.crop_x0,geometry.crop_y0,geometry.crop_x1,geometry.crop_y1)
        if bbox not in self.crops:raise ValueError(f'missing saved crop {bbox}')
        return self.crops[bbox]


def crop_array(geometry,array):
    if not isinstance(array,SavedCrops):raise TypeError('crop replay accepts saved inputs only')
    return array.crop(geometry)


def mapping(trace):
    ctx=trace['context'];shot=ctx['shot_game_context'];geo=ctx['runtime_geometry']
    if not shot['transform_method'].startswith('homography'):
        raise ValueError('saved homography required; no current-calibration fallback')
    h=np.asarray(geo['runtime_homography'],np.float64)
    if h.shape!=(3,3) or not np.isfinite(h).all():raise ValueError('invalid saved mapping')
    viewport=shot['viewport_xywh']
    convert=lambda x,y:transform_camera_point_to_game(x,y,viewport,
        lambda a,b:HitInput._transform(None,h,a,b))
    errors=[]
    for frame in trace['object_query_diagnostic']['candidate_ledger']:
        for row in frame['objects']:
            for c in row['candidates']:
                if c.get('game_xy') is not None:
                    errors.append(math.dist(convert(*c['camera_xy']),c['game_xy']))
    if errors and max(errors)>1e-6:raise ValueError('saved mapping disagrees with live coordinates')
    return convert,dict(source='saved_runtime_homography_and_shot_viewport',
        checked_pairs=len(errors),max_game_error=max(errors,default=None))


def load(capture):
    root=_development_path(capture)
    manifest=json.loads((root/'play.json').read_text())
    if manifest.get('mode')!='OBJECT_HIT_GAMEPLAY_ACCEPTANCE':raise ValueError('play-can manifest required')
    logs={}
    for line in (root/'runtime.log').read_text().splitlines():
        if '[OBJECT-HIT] ' in line:
            r=json.loads(line.split('[OBJECT-HIT] ',1)[1]);logs[int(r['shot_id'])]=r
    hashes={str(root/n):digest(root/n) for n in ('play.json','runtime.log','event_notes.json','runtime_exit.json') if (root/n).exists()}
    shots=[]
    # No discovery outside this explicit gameplay capture. IDs come from its log.
    for sid,log in sorted(logs.items()):
        directory=safe_file(root,f'object_crops/shots/shot_{sid:08d}')
        trace_path=safe_file(root,f'object_crops/shots/shot_{sid:08d}/trace.json')
        if not trace_path.exists():
            shots.append(dict(sid=sid,log=log,trace=None,boundary='COMPACT_LOG_ONLY'));continue
        hashes[str(trace_path)]=digest(trace_path)
        t=json.loads(trace_path.read_text())
        if t['shot_id']!=sid:raise ValueError('trace/log shot identity mismatch')
        if not t['completeness']['trace_complete']:raise ValueError(f'shot {sid}: incomplete crop persistence')
        convert,map_meta=mapping(t)
        w,h=t['context']['runtime_geometry']['calibration']['camera_frame_size']
        if w*h!=t['object_query_diagnostic']['full_camera_pixels']:raise ValueError('saved frame dimensions disagree')
        by_frame={};kind_by_ts={}
        for c in sorted(t['object_evidence_crops'],key=lambda c:c['seq']):
            file=safe_file(directory,c['path']);_development_path(file)
            pixels=np.load(file,allow_pickle=False)
            if (list(pixels.shape)!=c['shape'] or str(pixels.dtype)!=c['dtype'] or
                    hashlib.sha256(pixels.tobytes()).hexdigest()!=c['sha256']):
                raise ValueError('saved crop hash/shape/type mismatch')
            bbox=tuple(map(int,c['camera_bbox']));ts=c['timestamp']
            if list(bbox)!=c['camera_bbox'] or pixels.shape!=(bbox[3]-bbox[1],bbox[2]-bbox[0]):
                raise ValueError('invalid crop origin/dimensions')
            if bbox in by_frame.setdefault(ts,{}):raise ValueError('duplicate crop identity')
            if ts in kind_by_ts and kind_by_ts[ts]!=c['kind']:raise ValueError('conflicting frame kind')
            kind_by_ts[ts]=c['kind'];by_frame[ts][bbox]=pixels;pixels.flags.writeable=False
            hashes[str(file)]=digest(file)
        frames={ts:S(timestamp=ts,frame_bgr=SavedCrops((h,w),cs)) for ts,cs in by_frame.items()}
        q=t['object_query_diagnostic'];pre=[frames[ts] for ts in q['pre_frame_ts']]
        if any(kind_by_ts[f.timestamp]!='pre' or f.timestamp>=t['peak_ts'] for f in pre):
            raise ValueError('invalid causal PRE identity')
        posts=[frames[ts] for ts in sorted(frames) if kind_by_ts[ts]=='post']
        if any(f.timestamp<t['peak_ts'] or f.timestamp>q['decision_ts'] for f in posts):
            raise ValueError('saved POST outside recorded event cutoff')
        ctx=t['context']['shot_game_context']
        def region(cls,r):return cls(**{f.name:r[f.name] for f in fields(cls) if f.name in r})
        snap=ObjectShotSnapshotV2223(sid,t['peak_ts'],t['peak_ts'],(),
            game_regions=tuple(region(GameHitRegionV240,r) for r in ctx['game_hit_regions']),
            camera_regions=tuple(region(CameraHitRegionV240,r) for r in ctx['camera_hit_regions']),
            transform_available=ctx['transform_available'],transform_method=ctx['transform_method'],
            viewport_xywh=ctx['viewport_xywh'])
        shots.append(dict(sid=sid,log=log,trace=t,snapshot=snap,convert=convert,
            mapping=map_meta,pre=pre,posts=posts,frames=frames,boundary='CLOCK_NEUTRAL_SAVED_CROP_EVIDENCE'))
    return root,shots,hashes


def parity_view(value):
    """Exclude clocks/costs and unavailable known-hole context, never evidence."""
    omit={'delivered_ts','nearest_known_hole_px','timing','decision_ts'}
    if isinstance(value,dict):return {k:parity_view(v) for k,v in value.items() if k not in omit}
    if isinstance(value,list):return [parity_view(v) for v in value]
    return value


def differences(a,b,path=''):
    if type(a)!=type(b):return [path+':type']
    if isinstance(a,dict):
        if a.keys()!=b.keys():return [path+':keys']
        return [d for k in a for d in differences(a[k],b[k],path+'/'+k)]
    if isinstance(a,list):
        if len(a)!=len(b):return [path+':length']
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,path+'/'+str(i))]
    return [] if a==b else [path]


def replay(shot, *, profile_path=None, scalar=False):
    original=shot['trace']['object_query_diagnostic'];observations=[];prepared=[]
    profiler=cProfile.Profile() if profile_path else None
    with ExitStack() as stack:
        stack.enter_context(patch.object(AnalysisGeometryV2221,'crop_array',crop_array))
        stack.enter_context(patch.object(live.time,'perf_counter',new=lambda:0.))
        if scalar:
            stack.enter_context(patch.object(live.ObjectPhysicalQuery,'_evidence_many',staticmethod(
                lambda a,n,d,points:[live.ObjectPhysicalQuery._evidence(a,n,d,x,y) for x,y in points])))
        if profiler:profiler.enable()
        start=CLOCK()
        q=live.ObjectPhysicalQuery(shot['snapshot'],shot['pre'],camera_to_game=shot['convert'],
            top_k=original['top_k'],margin_px=original['margin_camera_px'])
        prepare_ms=(CLOCK()-start)*1000
        # Retain only already prepared crop-local envelope maps for forensic reads.
        prepared=[dict(bbox=(t['geometry'].crop_x0,t['geometry'].crop_y0,t['geometry'].crop_x1,t['geometry'].crop_y1),
            pre=t['pre'],envelope=t['pre_envelope'],noise=t['noise']) for t in q.tiles]
        select=q._select_hypotheses
        def collect(candidates,production):
            observations.append(dict(frame_ts=q.last_frame_ts,candidates=deepcopy(candidates)))
            return select(candidates,production)
        q._select_hypotheses=collect
        for f in shot['posts']:
            if q.final:break
            q.observe(f,delivered_ts=f.timestamp)
        if not q.final:q.finish('UNCERTAIN','saved_post_sequence_exhausted',shot['posts'][-1].timestamp)
        cost=(CLOCK()-start)*1000
        if profiler:profiler.disable();profiler.dump_stats(str(profile_path))
    recorded={f['frame_ts']:f for f in original['candidate_ledger']}
    comps=[]
    for f in q.result['candidate_ledger']:
        if f['frame_ts'] in recorded:
            diff=differences(parity_view(recorded[f['frame_ts']]),parity_view(f))
            comps.append(dict(frame_ts=f['frame_ts'],equal=not diff,differences=diff[:30]))
    # Profiling is optional; cost includes observer copies, excludes input loading.
    result=dict(shot_id=shot['sid'],mode=shot['boundary'],mapping=shot['mapping'],
        original_status=original['object_decision'],original_reason=original['reason'],
        replay=q.result,all_candidates=observations,recorded_ledger_comparison=comps,
        prepare_cpu_ms=prepare_ms,total_cpu_ms=cost,
        unavailable=['original thread scheduling/CPU cutoffs','original known-hole list (context only)',
                     'pixels outside saved crops','exposure timing/longer history/physical cause'],
        recorded_timeout=original['reason']=='processing_deadline_exceeded')
    return result,prepared


def object_rows(shot,result=None,prepared=()):
    log=shot['log'];sid=shot['sid'];rows=[]
    final=(result['replay']['object_decision'] if result else dict(state=log['status'],reason=log['reason'],
        supported_object_ids=log['supported_object_ids']))
    per_object=(result['replay']['per_object'] if result else log['per_object'])
    for ob in per_object:
        cs=ob.get('candidates') or ([ob['best_candidate']] if ob.get('best_candidate') else [])
        c=cs[0] if cs else {};fresh=c.get('event_freshness') or {};m=c.get('evidence') or {}
        oid=ob['object_id'];association=next((a for a in c.get('regions',[]) if a['object_id']==oid),{})
        row=dict(shot_id=sid,object_id=oid,camera_xy=c.get('camera_xy'),inside_exact_shape=association.get('inside_exact_shape'),
            support_state='SUPPORTED' if oid in final['supported_object_ids'] else 'UNKNOWN',
            qualification_reason=c.get('evidence_decision',ob.get('decision','unavailable')),
            event_freshness=fresh,clean_pre_certified=fresh.get('clean_pre_certified'),
            physical_cause_certified=fresh.get('physical_cause_certified'),same_polarity=fresh.get('same_polarity'),
            current_evidence=m,current_core_signed=m.get('center_abs',0)-2*m.get('center_dark',0) if m else None,
            previous_same_xy_evidence=c.get('previous_same_xy_evidence'),registration=c.get('registration'),
            object_candidates_examined=ob.get('observed_count'),retained_candidates=len(cs),
            per_object_time_ms=None,time_scope='not individually instrumented; query cost reported separately',
            final_decision=final,recorded_decision=log['status'],recorded_reason=log['reason'],
            recorded_supported=oid in log['supported_object_ids'],
            pre_observations=len(shot['pre']) if result else None,physical_cause='UNRESOLVED_NO_INDEPENDENT_GT')
        if result and c:
            x,y=c['camera_xy'];bbox=tuple(c['tile_camera_bbox']);x0,y0,x1,y1=map(int,bbox)
            px,py=int(x-x0),int(y-y0);tile=next(t for t in prepared if t['bbox']==bbox)
            yy,xx=np.ogrid[:y1-y0,:x1-x0];d2=(xx-px)**2+(yy-py)**2
            core=d2<=4;ring=(d2>=16)&(d2<=64);timeline=[]
            for ts,f in sorted(shot['frames'].items()):
                pixels=f.frame_bgr.crops[bbox]
                timeline.append(dict(timestamp=ts,kind='PRE' if ts<shot['trace']['peak_ts'] else 'POST',
                    core_mean=float(np.mean(pixels[core])),ring_mean=float(np.mean(pixels[ring]))))
            row['raw_intensity_timeline']=timeline
            if tile['envelope'] is not None:
                lo,hi=tile['envelope'];row['pre_envelope']=dict(core_low=float(np.mean(lo[core])),
                    core_high=float(np.mean(hi[core])),core_mean_width=float(np.mean((hi-lo)[core])),
                    pre_frame_ts=[f.timestamp for f in shot['pre']],plane='latest_PRE')
            last=shot['posts'][-1].frame_bgr.crops[bbox]
            try:
                row['nuisance_probe']={name:patch_evidence(tile['pre'],last,(px,py),local_registration=reg)
                    for name,reg in [('unregistered',False),('surround_registered',True)]}
            except ValueError:row['nuisance_probe']={'unavailable':'insufficient_crop_context'}
        rows.append(row)
    return rows


def run(capture,output, *, resume=False,repeats=1,profile=False,compare_scalar=False):
    root,shots,hashes=load(capture);out=_development_path(output)
    if not out.resolve().is_relative_to(Path('/data/skjutbana')):raise ValueError('output must be under /data/skjutbana')
    if out.resolve()==root.resolve() or out.resolve().is_relative_to(root.resolve()):raise ValueError('output must not modify capture')
    provenance=dict(input_hashes=hashes,source_hashes={str(p):digest(p) for p in [Path(__file__),Path(live.__file__),
        Path('src/engine/shot_region_freshness_v252.py'),Path('src/engine/camera/candidate_generator_v2.py'),
        Path('automation/object_physical_replay.py'),Path('automation/physical_collection.py'),
        Path('src/engine/input/object_hit_v2223.py'),Path('src/engine/input/hit_input.py'),
        Path('src/engine/game_objects/geometry.py'),Path('src/engine/camera/analysis_geometry_v2221.py')]},
        repeats=repeats,profile=profile,compare_scalar=compare_scalar)
    if resume:
        if json.loads((out/'provenance.json').read_text())!=provenance:raise ValueError('resume input/code/options mismatch')
    else:
        out.mkdir(parents=True,exist_ok=False);save(out/'provenance.json',provenance)
    rows=[];summary=[];candidates=[]
    for shot in shots:
        sid=shot['sid'];path=out/f'shot_{sid}.json'
        if resume and path.exists():record=json.loads(path.read_text())
        else:
            if shot['trace']:
                res,prepared=replay(shot);samples=[res['total_cpu_ms']]
                for _ in range(repeats-1):
                    other,_=replay(shot)
                    if parity_view(other['replay'])!=parity_view(res['replay']):raise ValueError('nondeterministic replay')
                    samples.append(other['total_cpu_ms'])
                if profile:
                    replay(shot,profile_path=out/f'shot_{sid}.prof')
                    with (out/f'shot_{sid}_profile.txt').open('w') as stream:
                        pstats.Stats(str(out/f'shot_{sid}.prof'),stream=stream).sort_stats('cumulative').print_stats(35)
                record=dict(result=res,rows=object_rows(shot,res,prepared),cpu_samples_ms=samples)
                if compare_scalar:
                    scalar_result,_=replay(shot,scalar=True)
                    diff=differences(parity_view(res['all_candidates']),parity_view(scalar_result['all_candidates']))
                    diff+=differences(parity_view(res['replay']),parity_view(scalar_result['replay']))
                    record['scalar_comparison']=dict(exact_equal=not diff,differences=diff[:30],
                        scalar_cpu_ms=scalar_result['total_cpu_ms'])
                    if diff:raise ValueError('scalar/batched semantic difference: '+str(diff[:3]))
            else:record=dict(result=None,rows=object_rows(shot),cpu_samples_ms=[])
            save(path,record)  # One immutable completed event; interruption resumes here.
        rows.extend(record['rows']);r=record['result'];original=shot['log']
        if r:
            candidates.extend(dict(shot_id=sid,camera_xy=c['camera_xy'],confirmed=c['confirmed'],
                inside_objects=[a['object_id'] for a in c['regions'] if a['inside_exact_shape'] is True])
                for c in r['all_candidates'][-1]['candidates'])
        summary.append(dict(shot_id=sid,mode=shot['boundary'],recorded_supported=original['supported_object_ids'],
            recorded_reason=original['reason'],recorded_audio_to_decision_ms=original['timing']['audio_to_decision_ms'],
            replay_decision=r['replay']['object_decision'] if r else None,cpu_ms=record['cpu_samples_ms'],
            recorded_ledger_comparison=r['recorded_ledger_comparison'] if r else None,
            scalar_comparison=record.get('scalar_comparison')))
    for row in rows:
        old=[r for r in rows if r['shot_id']<row['shot_id'] and r['camera_xy'] and row['camera_xy']]
        nearest=min(old,key=lambda r:math.dist(row['camera_xy'],r['camera_xy']),default=None)
        row['nearest_previous_candidate']=None if nearest is None else dict(shot_id=nearest['shot_id'],
            object_id=nearest['object_id'],camera_xy=nearest['camera_xy'],distance_px=math.dist(row['camera_xy'],nearest['camera_xy']))
        row['recurrence_semantics']='best logged/replayed candidates only; context, never a veto'
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('inputs changed during replay')
    for c in candidates:
        old=[p for p in candidates if p['shot_id']<c['shot_id']]
        previous=min(old,key=lambda p:math.dist(c['camera_xy'],p['camera_xy']),default=None)
        c['nearest_prior']=None if previous is None else dict(shot_id=previous['shot_id'],
            camera_xy=previous['camera_xy'],distance_px=math.dist(c['camera_xy'],previous['camera_xy']))
    save(out/'candidate_recurrence.json',candidates)
    save(out/'summary.json',summary);save(out/'objects.json',rows)
    with (out/'objects.csv').open('w') as stream:
        keys=list(dict.fromkeys(k for r in rows for k in r));writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    report=['# OBJECT_HIT saved-input replay','',
        'Unlabelled evidence analysis, not object accuracy. CPU-neutral replay does not reproduce live scheduling/deadlines.',
        'Certification flags are metadata limitations, not required Boolean gate inputs. Two PRE samples only describe an observed range.',
        'Physical cause remains unresolved; registration/illumination probes are diagnostic counterfactuals, never automatic vetoes.','',
        '| Shot | Recorded supports | Recorded reason | Replay supports | CPU median ms |','| --- | --- | --- | --- | --- |']
    for s in summary:
        count=len(s['replay_decision']['supported_object_ids']) if s['replay_decision'] else 'unavailable'
        report.append(f"| {s['shot_id']} | {len(s['recorded_supported'])} | {s['recorded_reason']} | {count} | {np.median(s['cpu_ms']) if s['cpu_ms'] else 'unavailable'} |")
    report+=['','See objects.csv/json for per-object evidence, PRE range, raw timelines, nuisance probes and recurrence.',
        'shot_N.json retains complete reconstructed candidate lists and comparison to every available recorded ledger.',
        'No GT is read into replay. Intended aim is not used. No pixels are supplied outside the captured crops.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--compare-scalar',action='store_true',help='Assert complete scalar/batched evidence and decision parity per shot')
    p.add_argument('--repeats',type=int,default=1)
    a=p.parse_args()
    if not 1<=a.repeats<=1000:p.error('repeats must be 1..1000')
    run(a.capture,a.output,resume=a.resume,repeats=a.repeats,profile=a.profile,compare_scalar=a.compare_scalar)


if __name__=='__main__':main()
