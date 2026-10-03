"""One frozen D02 pre-retention causal-witness audit. No new proposals or fitting."""
from __future__ import annotations
import argparse, json, math, hashlib
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from automation.causal_pre_onset_research import sample, read_guard, digest
from src.engine.offline.clean_physical_change import local_contrast_batch, causal_local_onset_batch, persistent_local_support

ROOT = Path('/data/skjutbana/physical_traces/session_20260926_162528_D02_4e1d6abd')

def candidate_coordinates(trace, cutoff):
    """Coordinate domain is frozen before any ground truth is read."""
    coordinates = {}
    def add(c, stage, origin=(0., 0.)):
        xy=(float(c['camera_x'])+origin[0], float(c['camera_y'])+origin[1])
        if not all(math.isfinite(v) for v in xy):raise ValueError('nonfinite camera XY')
        coordinates.setdefault(xy, []).append(dict(stage=stage, score=c.get('score'),
            novelty=c.get('v2222_novelty_factor'), pre_shot_change=c.get('pre_shot_change')))
    seen=set()
    for s in trace['stages']:
        p=s.get('pipeline')
        if not isinstance(p,dict) or s['timestamp']>cutoff or s.get('candidate_pool_shot_id')!=trace['shot_id']:continue
        if p['source_frame_ts'] in seen:continue
        seen.add(p['source_frame_ts']);up=p['upstream'];origin=up['crop_origin_camera'];u=up['pipeline']
        for c in u.get('pre_limit_candidates',[]):add(c,'pre_limit',origin)
        for r in u.get('hybrid_merge',{}).get('records',[]):
            add(r['candidate'],'hybrid_input',origin)
            if r.get('retained') and r.get('output_candidate'):add(r['output_candidate'],'hybrid_output',origin)
        for stage,cs in p['cleanup_stages'].items():
            for c in cs:add(c,stage)
        for c in p['retained_candidates']:add(c,'retained')
    audit=trace['decision_input']['complete_track_audit']
    for batch in audit['associations']:
        for r in batch['records']:add(r['candidate'],'association')
    for track in audit['tracks']:
        add(track,'final_track')
    return coordinates

def run(output):
    output.mkdir(parents=True,exist_ok=False);read_guard(output,[ROOT]);cv2.setNumThreads(1)
    hashes={};events=[]
    for sid in range(1,7):
        path=ROOT/'shots'/f'shot_{sid:08d}'/'trace.json';hashes[str(path)]=digest(path);t=json.loads(path.read_text())
        a=t['decision_input']['complete_track_audit'];cut=min(t['decision_input']['timestamp'],a['decision_observed_at'],t.get('next_audio_peak_ts',math.inf))
        pool=candidate_coordinates(t,cut);points=list(pool);series={};snapshot=None
        for f in sorted(t['frames'],key=lambda f:f['timestamp']):
            if f['timestamp']>cut:continue
            path=ROOT/'shots'/f'shot_{sid:08d}'/f['path'];hashes[str(path)]=digest(path)
            im=np.load(path,mmap_mode='r',allow_pickle=False)
            # One frame's local samples at a time; no T x candidates pixel bank.
            signal=local_contrast_batch(sample(im,points)[None])[0]
            if f['kind']=='pre_snapshot':snapshot=signal
            else:
                if f['timestamp'] in series:assert np.array_equal(series[f['timestamp']],signal)
                series[f['timestamp']]=signal
        times=np.array(sorted(series));signals=np.stack([series[ts] for ts in times]);onsets=causal_local_onset_batch(signals,times,cutoff=cut)
        rows=[]
        for j,(xy,r) in enumerate(zip(points,onsets)):
            track=next((c for c in a['tracks'] if (c['camera_x'],c['camera_y'])==xy),None)
            r.update(camera_xy=xy,stages=pool[xy],final_track=None if track is None else dict(track_id=track['track_id'],rank=track['final_rank'],eligible=track['eligible'],readiness=track['readiness_inputs'],local_confirmed=track['local_confirmed']))
            if r['state']=='ONSET_FOUND':
                r['causal_support']=persistent_local_support(signals[r['pre_index'],j],signals[-3:,j],r['noise'])
                r['old_support']=persistent_local_support(snapshot[j],signals[-3:,j],r['noise'])
            rows.append(r)
        # Labels are read only after inference over the complete unlabeled domain.
        gtpath=ROOT/'shots'/f'shot_{sid:08d}'/'ground_truth.json';gt=None
        if sid in (1,3):
            hashes[str(gtpath)]=digest(gtpath);gt=json.loads(gtpath.read_text())
            for row in rows:row['distance_px']=math.dist(row['camera_xy'],[gt['camera_x'],gt['camera_y']])
        found=[r for r in rows if r['state']=='ONSET_FOUND']
        event=dict(event=sid,cutoff=cut,coordinates=len(points),states=dict(Counter(r['state'] for r in rows)),supported=found,nearest_distance_px=min(r['distance_px'] for r in rows) if gt else None)
        (output/f'event_{sid}.json').write_text(json.dumps(dict(event=event,times=times.tolist(),candidates=rows),indent=2)+'\n')
        events.append(event);print(json.dumps(event),flush=True)
    assignments=ROOT/'physical_assignments.json';hashes[str(assignments)]=digest(assignments)
    labels=json.loads(assignments.read_text());assert all(labels[str(i)]['state']=='NO_PHYSICAL_SHOT' for i in (2,4,5,6))
    stop=any(events[i-1]['supported'] for i in (2,4,5,6))
    (output/'summary.json').write_text(json.dumps(dict(events=events,false_supported_witness_stop=stop,selector_run=False,protocol='Frozen local method, recorded coordinates only; GT joined after inference; any supported negative halts this line'),indent=2)+'\n')
    (output/'input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
