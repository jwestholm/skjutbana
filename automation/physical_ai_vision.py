"""Physical-cache adapter for V2.23.5 registered maps / V2.23.6 heatmap NN.

Own full-map peaks, never dense/target snapping. Vision-to-fusion remains disabled:
nested OOF model generation and verification are separate future work.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2
from automation import physical_ai_dataset as ds
from src.engine.ai.training_v223.evidence_patch_v2235 import build_registered_evidence_channels
from src.engine.ai.training_v223.heatmap_v2236 import downsample_block_mean, coarse_to_camera
from src.engine.ai.training_v223.heatmap_model_v2236 import (
    init_heatmap_model, extract_grid_patches, peak_camera_xy, train_stage, HeatmapTrainingSampleV2236,
)


def prepare(dataset, output, event_ids=None):
    data,_=ds.load(dataset); output=ds.output_root(output); output.mkdir(parents=True,exist_ok=False)
    # Reuse recorded reconstruction geometry; no GT-derived crop or PRE choice.
    from automation.shared_witness_research import events as descriptors
    from automation.chronological_pre_research import CROP
    geometry={e['id']:e for e in descriptors()}
    requested=set(event_ids or [e['event'] for e in data['events']])
    for key in requested: ds.guard(key)
    if not requested <= {e['event'] for e in data['events']}: raise ValueError('Unknown Vision events')
    rows=[]; cv2.setNumThreads(1)
    for e in data['events']:
        if e['event'] not in requested: continue
        x,y,w,h=geometry[e['event']].get('crop',CROP)
        refs=e['frames'][:3]; posts=e['frames'][-3:]
        if len(e['frames'])<6: raise ValueError('Insufficient chronological history')
        hashes={}
        def image(frame):
            path=Path(frame['frame_id']); ds.guard(path)
            hashes[str(path)]=ds.digest(path)
            if hashes[str(path)] != data['source_hashes'][str(path)]: raise ValueError('Source frame changed')
            return np.array(np.load(path,mmap_mode='r',allow_pickle=False)[y:y+h,x:x+w],dtype=np.uint8)
        pre=np.median(np.stack([image(f) for f in refs]),axis=0).astype(np.uint8)
        channels,meta=build_registered_evidence_channels(pre,[image(f) for f in posts])
        maps=downsample_block_mean(channels)
        name=e['event'].replace(':','_')+'.npy'; np.save(output/name,maps,allow_pickle=False)
        rows.append(dict(event=e['event'],session=e['session'],path=name,sha256=ds.digest(output/name),
                         origin=[x,y],shape=[h,w],source_hashes=hashes,reference_frames=refs,post_frames=posts,
                         status=e['status'],channel_metadata=meta,
                         policy='first three/last three saved PRE-history; frozen cutoff, not selected PRE; clean pre-damage is not guaranteed'))
    manifest=dict(schema='physical-vision-v2236-adapter-1',dataset=str(Path(dataset).resolve()),
                  dataset_hash=data['dataset_hash'],events=rows,source_sha256=ds.digest(Path(__file__)),
                  gt_chooses_images=False,vision_fusion_enabled=False,live_authority=False)
    ds.atomic_json(output/'vision.json',manifest)
    return manifest


def train(cache, output, train_sessions, held_sessions, epochs=2, seed=2360):
    for s in train_sessions+held_sessions: ds.guard(s)
    if not train_sessions or not held_sessions or set(train_sessions)&set(held_sessions):
        raise ValueError('Vision requires disjoint nonempty whole-session splits')
    if epochs<1 or epochs>100: raise ValueError('Bounded Vision adapter epochs: 1..100')
    cache=Path(cache); manifest=ds.read(cache/'vision.json'); data,_=ds.load(manifest['dataset'])
    if data['dataset_hash']!=manifest['dataset_hash']: raise ValueError('Vision data identity changed')
    available={e['session'] for e in manifest['events']}
    if not set(train_sessions+held_sessions)<=available: raise ValueError('Missing requested sessions')
    lookup={e['event']:e for e in data['events']}; samples=[]; maps_by_event={}; rng=np.random.default_rng(seed)
    for row in manifest['events']:
        if ds.digest(cache/row['path'])!=row['sha256']: raise ValueError('Vision cache changed')
        maps=np.load(cache/row['path'],allow_pickle=False); maps_by_event[row['event']]=maps
        e=lookup[row['event']]
        if row['session'] not in train_sessions or e['truth']!='HIT': continue
        yy,xx=np.indices(maps.shape[1:]); grid=np.column_stack([xx.ravel(),yy.ravel()])
        camera=grid*4+1.5+np.asarray(row['origin'])
        distances=np.linalg.norm(camera-np.asarray(e['gt_xy']),axis=1)
        pos=np.flatnonzero(distances<=6.5); neg=np.flatnonzero(distances>42)
        # Dense-grid training targets only; inference peaks never use GT.
        if not len(pos) or not len(neg): continue
        strength=maps.max(axis=0).ravel()
        hard=neg[np.argsort(-strength[neg].astype(float),kind='stable')[:128]]
        random=rng.choice(neg,min(128,len(neg)),replace=False)
        indices=np.unique(np.concatenate([pos,hard,random]))
        samples.append(HeatmapTrainingSampleV2236(row['session'],e['event'],extract_grid_patches(maps,grid[indices]),distances[indices]))
    if len(samples)<2: raise ValueError('Need >=2 actual physical image training events')
    output=ds.output_root(output); output.mkdir(parents=True,exist_ok=False)
    model=init_heatmap_model('spatial_conv',hidden=4,seed=seed)
    history=train_stage(model,samples,epochs=epochs,learning_rate=.003,l2=.001,seed=seed,stage_name='physical_map_adapter')
    model.metadata.update(train_sessions=train_sessions,held_sessions=held_sessions,dataset_hash=data['dataset_hash'],
                          role='VISION_OWN_HEATMAP',live_authority=False,seed=seed,
                          control_training='not supported by existing positive-pair loss; controls evaluated only',
                          code_revision=ds.get_git_commit(),adapter_sha256=ds.digest(Path(__file__)))
    model.save(output)
    rows=[]
    for row in manifest['events']:
        if row['session'] not in held_sessions: continue
        e=lookup[row['event']]
        score=model.score_map(maps_by_event[e['event']])
        peaks=peak_camera_xy(score,top_k=5)+np.asarray(row['origin'])
        distances=[ds.distance(p,e['gt_xy']) for p in peaks] if e['truth']=='HIT' else []
        rows.append(dict(event=e['event'],session=e['session'],truth=e['truth'],peaks=peaks.tolist(),
                         top1_error=distances[0] if distances else None,
                         oracle_error=min(distances) if distances else None,
                         false_proposals=sum(d>42 for d in distances) if e['truth']=='HIT' else len(peaks) if e['truth']=='NO_IMPACT' else None))
    def metrics(rr):
        positive=[r for r in rr if r['truth']=='HIT']; errors=[r['top1_error'] for r in positive if r['top1_error'] is not None]
        known=[r['false_proposals'] for r in rr if r['false_proposals'] is not None]
        return dict(impacts=len(positive),localization={str(t):sum(r['top1_error'] is not None and r['top1_error']<=t for r in positive) for t in (5,10)},
                    proposal_oracle={str(t):sum(r['oracle_error'] is not None and r['oracle_error']<=t for r in positive) for t in (5,10)},
                    median_error=float(np.median(errors)) if errors else None,
                    false_proposals_per_event=float(np.mean(known)) if known else None)
    result=dict(training_loss=history,train_sessions=train_sessions,held_sessions=held_sessions,
                per_event=rows,metrics=metrics(rows),per_session={s:metrics([r for r in rows if r['session']==s]) for s in held_sessions},
                model_sha256=ds.digest(output/'model.npz'),dataset_hash=data['dataset_hash'],
                status='OFFLINE_VISUAL_ADAPTER_NOT_LIVE_VALIDATION',fusion_features_enabled=False,
                limitations=['fixed first/last history representation not qualified clean PRE',
                             'positive-pair training does not train control abstention',
                             'no nested OOF score generation for downstream fusion',
                             'no learned probability calibration or performance claim'])
    ds.atomic_json(output/'result.json',result)
    return result


def main():
    ds.install_guard(); p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--dataset',type=Path,required=True); a.add_argument('--output',type=Path,required=True); a.add_argument('--events',nargs='+')
    a=sub.add_parser('train'); a.add_argument('--cache',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a.add_argument('--train-sessions',nargs='+',required=True); a.add_argument('--held-sessions',nargs='+',required=True); a.add_argument('--epochs',type=int,default=2)
    args=p.parse_args()
    if args.cmd=='prepare': result=prepare(args.dataset,args.output,args.events)
    else: result=train(args.cache,args.output,args.train_sessions,args.held_sessions,args.epochs)
    print(result.get('metrics',{'events':len(result['events'])} if 'events' in result else {}))


if __name__=='__main__': main()
