"""Offline temporal PRE/POST forensics for one physical trace session."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from src.engine.offline.registered_impact import prepare_pair, extract, sample

def _examples(features):
    by={}
    for e in features['examples']:
        if e['role'] in ('nearest_eligible','current_winner'):
            by.setdefault(int(e['shot']),{})[e['role']]=e
    return by

def run(root: Path, features_path: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    examples=_examples(json.loads(features_path.read_text()))
    rows=[]
    for tp in sorted((root/'shots').glob('*/trace.json')):
        trace=json.loads(tp.read_text()); sid=int(trace['shot_id'])
        if sid not in examples: continue
        frames=sorted(trace['frames'],key=lambda f:f['timestamp'])
        preframe=next(f for f in frames if f['kind']=='pre_snapshot')
        pre=np.load(tp.parent/preframe['path'],mmap_mode='r')
        targets={role:e for role,e in examples[sid].items()}
        for role,example in targets.items():
            tr=example['track']
            coords=[('final_track',(float(tr['camera_x']),float(tr['camera_y'])))]
            for i,a in enumerate(example.get('association_history',[])):
                c=a.get('candidate',{}); x=c.get('camera_x'); y=c.get('camera_y')
                if x is not None and y is not None: coords.append((f'observation_{i}',(float(x),float(y))))
            lc=tr.get('last_candidate',{})
            if lc.get('camera_x') is not None: coords.append(('confirmation_candidate',(float(lc['camera_x']),float(lc['camera_y']))))
            # De-duplicate coordinates while preserving provenance.
            uniq=[]; seen=set()
            for label,c in coords:
                key=(round(c[0],3),round(c[1],3))
                if key not in seen: seen.add(key); uniq.append((label,c))
            coord_rows=[]
            for coord_label,xy in uniq:
                samples=[]
                for f in frames:
                    image=np.load(tp.parent/f['path'],mmap_mode='r')
                    pp=sample(pre,xy,(0,0),radius=16); qq=sample(image,xy,(0,0),radius=16)
                    pair=prepare_pair(pp,qq,translation=(0,0)); vals=extract(pair,(16,16))
                    samples.append(dict(kind=f['kind'],timestamp=f['timestamp'],dt=f['timestamp']-trace['peak_ts'],**{k:vals[k] for k in ('ring_affine_center_dark','ring_affine_center_bright','ring_affine_compact','ring_affine_signed_contrast','ring_affine_concentration','ring_affine_entropy','ring_affine_component_area','ring_affine_peak')}))
                before=[s for s in samples if s['dt']<-.05]; after=[s for s in samples if s['dt']>=.05]
                bd=[s['ring_affine_center_dark'] for s in before]; ad=[s['ring_affine_center_dark'] for s in after]
                base=float(np.median(bd)) if bd else 0.; peak=max(ad,default=0.)
                coord_rows.append(dict(coord_label=coord_label,xy=xy,before_stability_dark=float(np.std(bd)),impact_onset_dark=float(peak-base),after_persistence_dark=float(sum(v>base+2 for v in ad)),after_dark_peak=peak,after_dark_median=float(np.median(ad)) if ad else 0.))
            best=max(coord_rows,key=lambda r:(r['after_persistence_dark'],r['impact_onset_dark']))
            row=dict(shot=sid,role=role,gt_distance=example.get('gt_distance'),xy=list(coords[0][1]),coordinate_rows=coord_rows,best_temporal_coordinate=best['coord_label'],**{k:best[k] for k in ('before_stability_dark','impact_onset_dark','after_persistence_dark','after_dark_peak','after_dark_median')})
            rows.append(row)
            continue
            # Legacy single-coordinate path retained below for readable history.
            xy=(float(tr['camera_x']),float(tr['camera_y']))
            samples=[]
            for f in frames:
                image=np.load(tp.parent/f['path'],mmap_mode='r')
                # Registration is deliberately held fixed for this first
                # temporal study: use the same camera-space patch and the
                # event's saved local geometry. Full-frame phase correlation
                # per frame is prohibitively expensive and would confound
                # temporal change with a new registration estimate.
                pp=sample(pre,xy,(0,0),radius=16)
                qq=sample(image,xy,(0,0),radius=16)
                pair=prepare_pair(pp,qq,translation=(0,0))
                vals=extract(pair,(16,16))
                samples.append(dict(kind=f['kind'],timestamp=f['timestamp'],dt=f['timestamp']-trace['peak_ts'],**{
                    k:vals[k] for k in ('ring_affine_center_dark','ring_affine_center_bright','ring_affine_compact','ring_affine_signed_contrast','ring_affine_concentration','ring_affine_entropy','ring_affine_component_area','ring_affine_peak')}))
            before=[s for s in samples if s['dt']<-.05]
            after=[s for s in samples if s['dt']>=.05]
            def series(key,items): return [float(s[key]) for s in items]
            dark=series('ring_affine_center_dark',samples); compact=series('ring_affine_compact',samples)
            before_dark=series('ring_affine_center_dark',before); after_dark=series('ring_affine_center_dark',after)
            before_comp=series('ring_affine_compact',before); after_comp=series('ring_affine_compact',after)
            base=float(np.median(before_dark)) if before_dark else 0.
            peak=max(after_dark,default=0.)
            row=dict(shot=sid,role=role,gt_distance=examples[sid][role].get('gt_distance'),xy=xy,samples=samples,
                before_stability_dark=float(np.std(before_dark)),before_stability_compact=float(np.std(before_comp)),
                impact_onset_dark=float(peak-base),after_persistence_dark=float(sum(v>base+2 for v in after_dark)),
                after_frames=len(after),after_dark_peak=peak,after_dark_median=float(np.median(after_dark)) if after_dark else 0.,
                after_compact_peak=max(after_comp,default=0.),first_after_dark=float(after_dark[0]) if after_dark else 0.,
                temporal_dark_range=float(max(dark)-min(dark)) if dark else 0.)
            rows.append(row)
    out=dict(status='RESEARCH_ONLY',input_session=str(root),features=str(features_path),examples=len(rows),rows=rows)
    (output/'temporal_features.json').write_text(json.dumps(out,indent=2)+'\n')
    # Compact per-role summaries; no fitted selector.
    summary={}
    for role in ('nearest_eligible','current_winner'):
        rr=[r for r in rows if r['role']==role]
        summary[role]={k:{'median':float(np.median([x[k] for x in rr])),'q25':float(np.percentile([x[k] for x in rr],25)),'q75':float(np.percentile([x[k] for x in rr],75))} for k in ('before_stability_dark','impact_onset_dark','after_persistence_dark','after_dark_peak','after_dark_median')}
    (output/'temporal_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(examples=len(rows),summary=summary)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,a.features,a.output)
