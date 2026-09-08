"""Read-only multi-event physical audit. Explicit label mappings never alter captures."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from src.engine.offline.causal_candidates import analyze_trace


def distance(c, gt):
    return math.hypot(c['camera_x'] - gt['camera_x'], c['camera_y'] - gt['camera_y'])


def pool_summary(pool, gt):
    if pool is None:
        return {'availability': 'UNAVAILABLE'}
    out = {'availability': 'OBSERVED', 'count': len(pool)}
    if gt and pool:
        i = min(range(len(pool)), key=lambda i: distance(pool[i], gt))
        out.update(nearest_distance_px=distance(pool[i], gt), nearest_list_position=i+1,
                   nearest_candidate=pool[i], first_positive_positions={str(r): next((j+1 for j,c in enumerate(pool) if distance(c,gt)<=r), None) for r in (5,10,20,42)})
    return out


def audit(root, mapping=None):
    rows=[]
    paths = sorted((root/'shots').glob('*/trace.json'))
    traces = [json.loads(path.read_text()) for path in paths]
    for index, (path, t) in enumerate(zip(paths, traces)):
        sid=str(t['shot_id']); assignment=(mapping or {}).get(sid, {})
        label_id=assignment.get('label_shot_id',sid)
        gp=root/'shots'/f'shot_{int(label_id):08d}'/'ground_truth.json' if label_id is not None else None
        gt=json.loads(gp.read_text()) if gp and gp.exists() else None
        state=assignment.get('state', gt.get('quality','approximate') if gt else 'unresolved')
        if state=='NO_PHYSICAL_SHOT': gt=None
        stages=t['stages']; outcome=t.get('outcome',{})
        observations=[]
        for i,s in enumerate(stages):
            pool=s.get('candidates'); tracks=s.get('tracks')
            confirmed=[c for c in tracks if c.get('state')=='confirmed'] if isinstance(tracks,list) and all(isinstance(c,dict) for c in tracks) else None
            observations.append({'index':i,'timestamp':s['timestamp'],'event_state':(s.get('event') if isinstance(s.get('event'),dict) else {}).get('state'),
                                 'retained':pool_summary(pool,gt),'emitted_track_states':pool_summary(confirmed,gt),
                                 'local_confirmation_evidence':pool_summary([c for c in tracks if isinstance(c,dict) and c.get('last_candidate',{}).get('v2225_local_confirm')],gt) if isinstance(tracks,list) and all(isinstance(c,dict) for c in tracks) else {'availability':'UNAVAILABLE'}})
        pipeline=next((s['pipeline'] for s in reversed(stages) if isinstance(s.get('pipeline'),dict)),{})
        nonempty=[s for s in stages if s.get('candidates')]
        retained=nonempty[-1]['candidates'] if nonempty else None
        final=outcome.get('final_camera_xy') if outcome.get('emitted') else None
        # Selection is not inferred from emission: track snapshots provide separate evidence.
        selected=next((c for s in reversed(stages) for c in s.get('tracks',[]) if isinstance(c,dict) and c.get('track_id')==outcome.get('matched_track_id')),None)
        debug=[s.get('window_debug',{}) for s in nonempty]
        rows.append({'shot_id':sid,'peak_ts':t['peak_ts'],'physical_state':state,'assignment':assignment,
          'ground_truth':gt,'ground_truth_source':str(gp) if gt else None,'trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'raw':pool_summary(pipeline.get('raw_candidates') if isinstance(pipeline.get('raw_candidates'),list) else None,gt),
          'filtered':pool_summary(pipeline.get('filtered_candidates') if isinstance(pipeline.get('filtered_candidates'),list) else None,gt),
          'last_observed_retained':pool_summary(retained,gt),'observations':observations,
          'selected_track':selected,'selected_distance_px':distance(selected,gt) if selected and gt else None,
          'emitted':final,'emitted_distance_px':distance(final,gt) if final and gt else None,
          'outcome':outcome,'complete':t.get('completeness'),
          'registration_observations':[{k:v for k,v in d.items() if k.startswith('v2_registration')} for d in debug]})
        causal = analyze_trace(t, traces[index+1]['peak_ts'] if index+1 < len(traces) else None, gt)
        rows[-1]['causal_candidate_audit_v1'] = {k:v for k,v in causal.items() if k not in ('records','observations')}
    return {'schema':'physical-session-audit-1','coordinate_space':'original_camera','session':str(root),
      'limitations':['Saved-list position is not authority rank. Observations may follow the decision; do not treat union recall as decision-time recall.',
                     'RAW and FILTERED not captured by these producers. Legacy tracks are top-eight debug views: presence of local-confirm flag proves confirmation; absence does not prove loss. Track state confirmed means emitted, not local-confirmation eligibility.',
                     'Physical assignments require temporal evidence; missing label never implies nonphysical event.'], 'shots':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--mapping',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=json.loads(a.mapping.read_text()) if a.mapping else None
    result=audit(a.root,m);a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'scorecard.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['|Event|GT state|GT camera XY|Nearest retained px / position|Emitted error px|Runtime|Latency ms|','|---|---|---|---|---|---|---|']
    for s in result['shots']:
        g=s['ground_truth'];r=s['last_observed_retained'];o=s['outcome']
        lines.append(f"|{s['shot_id']}|{s['physical_state']}|{None if g is None else (round(g['camera_x'],2),round(g['camera_y'],2))}|{r.get('nearest_distance_px')} / {r.get('nearest_list_position')}|{s['emitted_distance_px']}|{o.get('status')}|{o.get('detector_e2e_latency_ms')}|")
    (a.output/'scorecard.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__': main()
