"""Reproducible read-only causal physical audit; output must be a new directory."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
from src.engine.offline.causal_candidates import analyze_trace
from automation.physical_score_audit import source


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted((root/'shots').glob('*/trace.json'))
    traces = [json.loads(p.read_text()) for p in paths]
    rows, artifacts = [], []
    for index, (path, t) in enumerate(zip(paths, traces)):
        gp = path.parent/'ground_truth.json'
        gt = json.loads(gp.read_text()) if gp.exists() else None
        next_peak = traces[index+1]['peak_ts'] if index+1<len(traces) else None
        a = analyze_trace(t, next_peak, gt)
        (output/f"event_{t['shot_id']:02d}.json").write_text(json.dumps(a, indent=2)+'\n')
        selected = t.get('decision_input', {}).get('deterministic_selection', {})
        c = selected.get('last_candidate', {})
        nearest = lambda pool: min((math.hypot(x['camera_x']-gt['camera_x'],x['camera_y']-gt['camera_y']) for x in pool),default=None) if gt else None
        last = next((s['candidates'] for s in reversed(t['stages']) if s.get('candidates')), [])
        decision_pool = t.get('decision_input', {}).get('retained_candidates', [])
        row = {k:v for k,v in a.items() if k not in ('records','observations')}
        row.update(peak_ts=t['peak_ts'], physical_shot=None if gt is None else sum((p.parent/'ground_truth.json').exists() for p in paths[:index+1]),
                   ground_truth=gt, trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   stages=len(t['stages']), frames=len(t['frames']), completeness=t.get('completeness'),
                   outcome=t.get('outcome'), audio_trigger=t.get('audio_trigger'),
                   last_retained_nearest_px=nearest(last), decision_retained_nearest_px=nearest(decision_pool),
                   current_error_px=nearest([selected]), winner=selected, winner_source=source(c),
                   winner_source_percentile=sum(x.get('score',0)<=c.get('score',0) for x in decision_pool if source(x)==source(c))/max(1,sum(source(x)==source(c) for x in decision_pool)),
                   track_age_s=selected.get('last_seen_ts',0)-selected.get('first_seen_ts',0))
        rows.append(row)
        # Load every saved array, verify every element is finite and inventory
        # shape/dtype/range. This also includes maps omitted from summaries.
        for f in sorted(path.parent.rglob('*.npy')):
            arr = np.load(f, mmap_mode='r', allow_pickle=False)
            artifacts.append({'event':t['shot_id'],'path':str(f.relative_to(root)), 'shape':list(arr.shape),
                              'dtype':str(arr.dtype),'min':float(np.min(arr)), 'max':float(np.max(arr)),
                              'finite':bool(np.all(np.isfinite(arr)))})
        print('audited event',t['shot_id'],flush=True)
    physical = [r for r in rows if r['ground_truth']]
    metrics = {name:{str(tol):sum(r[name]['hits'][str(tol)] for r in physical) for tol in (5,10,20,42)}
               for name in ('causal_oracle','post_decision_only_oracle','cross_event_only_oracle','posthoc_all_observed_oracle')}
    metrics['last_observed_retained_oracle']={str(tol):sum(r['last_retained_nearest_px'] is not None and r['last_retained_nearest_px']<=tol for r in physical) for tol in (5,10,20,42)}
    metrics['decision_retained_oracle']={str(tol):sum(r['decision_retained_nearest_px'] is not None and r['decision_retained_nearest_px']<=tol for r in physical) for tol in (5,10,20,42)}
    wrong=[r for r in physical if r['current_error_px']>42]
    clusters=Counter()
    for r in wrong:
        c=r['winner']['last_candidate']
        clusters[r['winner_source']]+=1
        for label, condition in [('psc_ge_100',c.get('pre_shot_change',0)>=100),('local_confirm',c.get('v2225_local_confirm',0)>0),
                                  ('zero_confirm_darkening',c.get('v2225_confirm_darkening')==0),('temporal_rescue',c.get('v2_rescue_temporal',0)>0),
                                  ('score_ge_35',c.get('score',0)>=35),('unique_frame_hits_2',r['winner'].get('v2226_unique_frame_hits')==2)]:
            if condition: clusters[label]+=1
    result={'status':'DEVELOPMENT_CAUSAL_AUDIT','physical_events':len(physical),'all_events':len(rows), 'metrics_counts':metrics,
            'false_winner_overlapping_clusters':dict(clusters),'rows':rows,'artifact_count':len(artifacts)}
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    lines=['|Event|Physical|Decision +s|Causal nearest px|Last retained px|Posthoc all px|CURRENT px|Source|','|---|---|---|---|---|---|---|---|']
    fmt=lambda v:'—' if v is None else f'{v:.3f}'
    for r in rows:
        lines.append('|'+ '|'.join(map(str,[r['shot_id'],r['physical_shot'] or 'FALSE',fmt(r['decision_cutoff']-r['peak_ts']),fmt((r['causal_oracle'] or {}).get('nearest_px')),fmt(r['last_retained_nearest_px']),fmt((r['posthoc_all_observed_oracle'] or {}).get('nearest_px')),fmt(r['current_error_px']),r['winner_source']]))+'|')
    (output/'timeline.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'metrics':metrics,'clusters':dict(clusters),'artifacts':len(artifacts)},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)

if __name__=='__main__': main()
