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


def prepare(dataset, output, event_ids=None, *, native=False):
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
        maps=channels if native else downsample_block_mean(channels)
        name=e['event'].replace(':','_')+'.npy'; np.save(output/name,maps,allow_pickle=False)
        rows.append(dict(event=e['event'],session=e['session'],path=name,sha256=ds.digest(output/name),
                         origin=[x,y],shape=[h,w],source_hashes=hashes,reference_frames=refs,post_frames=posts,
                         status=e['status'],channel_metadata=meta,
                         policy='first three/last three saved PRE-history; frozen cutoff, not selected PRE; clean pre-damage is not guaranteed'))
    repo=Path(__file__).resolve().parents[1]
    map_code=('automation/physical_ai_vision.py','src/engine/ai/training_v223/evidence_patch_v2235.py',
              'src/engine/ai/training_v223/heatmap_v2236.py')
    manifest=dict(schema='physical-vision-proposals-1' if native else 'physical-vision-v2236-adapter-1',dataset=str(Path(dataset).resolve()),
                  dataset_hash=data['dataset_hash'],events=rows,source_sha256=ds.digest(Path(__file__)),
                  map_builder_hashes={f:ds.digest(repo/f) for f in map_code},code_revision=ds.get_git_commit(),
                  native_stride=1 if native else 4,
                  gt_chooses_images=False,vision_fusion_enabled=False,live_authority=False)
    manifest['cache_hash']=ds.canonical_hash(manifest)
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


# Bounded proposal-search extension. The lab lifecycle remains in
# physical_ai_lab.run; this module supplies the visual adapter and scoreboard.
import json
import time
from src.engine.ai.training_v223.heatmap_model_v2236 import (
    HeatmapModelV2236, _local_maxima, train_physical_proposals,
)

BUDGETS=(1,3,8)
CHANNEL_SETS={'all':tuple(range(8)), 'persistent':(2,4,5,7), 'morphology':(0,1,3,6,7)}


def code_identity():
    from automation import physical_ai_lab as lab
    identity=lab.code_identity();repo=Path(__file__).resolve().parents[1]
    for f in ('automation/physical_ai_vision.py','src/engine/ai/training_v223/heatmap_model_v2236.py',
              'src/engine/ai/training_v223/heatmap_v2236.py','src/engine/ai/training_v223/evidence_patch_v2235.py'):
        identity['source_hashes'][f]=ds.digest(repo/f)
    identity['role']='VISION_PROPOSAL';return identity


def load(cache):
    cache=Path(cache);manifest=ds.read(cache/'vision.json')
    expected=manifest.pop('cache_hash')
    if ds.canonical_hash(manifest)!=expected:raise ValueError('Vision manifest changed')
    manifest['cache_hash']=expected
    if manifest.get('native_stride')!=1 or manifest['schema']!='physical-vision-proposals-1':
        raise ValueError('Proposal lab needs native maps; legacy stride4 smoke is not this benchmark')
    data,_=ds.load(manifest['dataset'])
    if data['dataset_hash']!=manifest['dataset_hash']:raise ValueError('Vision/classic benchmark mismatch')
    if len(manifest['events'])!=len(data['events']) or {e['event'] for e in manifest['events']}!={e['event'] for e in data['events']}:
        raise ValueError('Vision benchmark must cover every fixed event exactly once')
    events={e['event']:e for e in data['events']}
    for row in manifest['events']:
        ds.guard(row['event']);ds.guard(row['session']);ds.guard(row['path'])
        for f in row['reference_frames']+row['post_frames']:ds.guard(f['frame_id'])
        for f,h in row['source_hashes'].items():
            ds.guard(f)
            if data['source_hashes'].get(f)!=h:raise ValueError('Vision source frame identity mismatch')
        path=(cache/row['path']).resolve()
        if not path.is_relative_to(cache.resolve()):raise ValueError('Vision path escaped cache')
        if row['session']!=events[row['event']]['session']:raise ValueError('Vision source session mismatch')
        if ds.digest(path)!=row['sha256']:raise ValueError('Vision map hash mismatch')
        maps=np.load(path,mmap_mode='r',allow_pickle=False)
        if maps.ndim!=3 or maps.shape[0]!=8 or list(maps.shape[1:])!=row['shape'] or maps.dtype!=np.uint8:
            raise ValueError('Invalid native map geometry')
    data=dict(data,classic_dataset_hash=data['dataset_hash'],dataset_hash=expected,
              vision_rows={r['event']:r for r in manifest['events']})
    return data,cache


def config_for(seed, trial):
    rng=np.random.default_rng(np.random.SeedSequence([seed,trial]))
    return dict(seed=int(rng.integers(1,2**30)),kind='spatial_conv',hidden=int(rng.choice([2,4,8])),
                epochs=int(rng.choice([3,6,10])),learning_rate=float(10**rng.uniform(-3.5,-2)),
                l2=float(10**rng.uniform(-5,-2.5)),stride=int(rng.choice([1,2,4])),
                channels=str(rng.choice(list(CHANNEL_SETS))),positive_radius=float(rng.choice([3.5,4.5,6.5])),
                loss_kind=str(rng.choice(['bce','pair'])),hard_count=int(rng.choice([32,64,128])),
                nms_radius_px=float(rng.choice([6,12,24])),budgets=list(BUDGETS),
                temporal='first3_last3',normalization='train_channel_standardization')


def representation_maps(maps,representation='native'):
    """GT-free, shape-preserving representation; native cache stays immutable.

    context9 retains all four native persistent channels. Unused slots contain
    their 9x9 context, giving the existing 5x5 model a 13px context field at stride1.
    This does not recover temporal information discarded by the original cache.
    """
    if representation=='native':return maps
    if representation!='context9':raise ValueError('Unknown physical representation')
    out=np.array(maps,copy=True)
    for target,source in zip((0,1,3,6),(2,4,5,7)):
        out[target]=cv2.GaussianBlur(maps[source],(9,9),2,borderType=cv2.BORDER_REFLECT_101)
    return out


def event_maps(cache,row,stride,representation='native'):
    ds.guard(Path(cache)/row['path'])
    maps=np.load(Path(cache)/row['path'],mmap_mode='r',allow_pickle=False)
    maps=representation_maps(maps,representation)
    return maps if stride==1 else downsample_block_mean(maps,stride)


def training_samples(data,cache,fold,cfg):
    """Held imagery and truth are never loaded by this training sampler."""
    if set(fold['train'])&set(fold['held']+fold['dev']):raise ValueError('Vision fold overlap')
    samples=[];events=[];rng=np.random.default_rng(cfg['seed']);stride=cfg['stride']
    for e in data['events']:
        if e['session'] not in fold['train'] or e['truth']=='UNKNOWN':continue
        row=data['vision_rows'][e['event']];maps=event_maps(cache,row,stride,cfg.get('representation','native'))
        yy,xx=np.indices(maps.shape[1:]);grid=np.column_stack([xx.ravel(),yy.ravel()])
        if e['truth']=='HIT':
            camera=grid*stride+(stride-1)*.5+np.asarray(row['origin'])
            distances=np.linalg.norm(camera-np.asarray(e['gt_xy']),axis=1)
        elif e['truth']=='NO_IMPACT':distances=np.full(len(grid),np.inf)
        else:raise ValueError('Unknown truth must not train as negative')
        pos=np.flatnonzero(distances<=cfg['positive_radius']);neg=np.flatnonzero(distances>42)
        strength=np.max(maps[list(CHANNEL_SETS[cfg['channels']])],axis=0).ravel()
        hard=neg[np.argsort(-strength[neg].astype(float),kind='stable')[:128]]
        rand=rng.choice(neg,min(128,len(neg)),replace=False) if len(neg) else neg
        idx=np.unique(np.concatenate([pos,hard,rand]))
        if not len(idx):continue
        patches=extract_grid_patches(maps,grid[idx])
        samples.append(HeatmapTrainingSampleV2236(e['session'],e['event'],patches,distances[idx]))
        events.append(e['event'])
    if len(samples)<2:raise ValueError('Insufficient train-session physical samples')
    # Event-balanced channel statistics from TRAIN patches only. Per-image V2235
    # robust map scaling is deterministic GT-free preprocessing, not a fit over
    # held-session populations.
    means=np.stack([(s.patches.astype(float)/255).mean(axis=(0,2,3)) for s in samples])
    seconds=np.stack([((s.patches.astype(float)/255)**2).mean(axis=(0,2,3)) for s in samples])
    mean=means.mean(axis=0);std=np.sqrt(np.maximum(0,seconds.mean(axis=0)-mean**2));std=np.maximum(std,.02)
    norm=dict(mean=mean.tolist(),scale=std.tolist(),training_events=events,training_sessions=fold['train'],channels=list(CHANNEL_SETS[cfg['channels']]))
    for s in samples:s.patches=normalize(s.patches,norm,patches=True)
    return samples,norm


def normalize(value,norm,*,patches=False):
    shape=(1,8,1,1) if patches else (8,1,1)
    x=np.asarray(value,np.float32)/255
    x=(x-np.asarray(norm['mean'],np.float32).reshape(shape))/np.asarray(norm['scale'],np.float32).reshape(shape)
    active=np.zeros(8,np.float32);active[norm['channels']]=1
    return x*active.reshape(shape)*255


def proposals(score,*,stride,origin,nms_radius_px,limit=8):
    """Direct learned maxima in camera space; no label/classic input exists."""
    if limit not in BUDGETS:raise ValueError('Only fixed 1/3/8 proposal budgets')
    if stride not in (1,2,4):raise ValueError('Unsupported native map stride')
    if not np.isfinite(score).all():raise ValueError('Nonfinite heatmap')
    pts=_local_maxima(score,top_k=limit,nms_radius=max(1,int(np.ceil(nms_radius_px/stride))))
    return [dict(camera_xy=(p*stride+(stride-1)*.5+np.asarray(origin)).tolist(),
                 confidence=float(score[int(p[1]),int(p[0])]),rank=i+1,source='VISION',
                 confidence_semantics='uncalibrated learned logit') for i,p in enumerate(pts)]


def union(classic,vision):
    """Remove only exact duplicate XY; near points cannot replace a witness."""
    xy=[list(c['xy']) for c in classic];seen={tuple(p) for p in xy};added=0;duplicates=0;near=0
    for p in vision:
        point=p['camera_xy']
        if tuple(point) in seen:duplicates+=1;continue
        near+=int(any(ds.distance(point,old)<=3.5 for old in xy))
        xy.append(list(point));seen.add(tuple(point));added+=1
    return xy,dict(added=added,exact_duplicates=duplicates,near_duplicates_r3_5=near)


def validate_lineage(proposal,event,fold,cache_hash):
    ds.guard(proposal['held_session']);ds.guard(proposal['event'])
    for s in proposal['training_sessions']:ds.guard(s)
    if proposal['source']!='VISION' or proposal['held_session']!=event['session'] or proposal['event']!=event['event']:
        raise ValueError('Foreign Vision proposal')
    if proposal['training_sessions']!=fold['train'] or event['session'] in proposal['training_sessions']:
        raise ValueError('Vision upstream session leakage')
    if proposal['data_hash']!=cache_hash or not proposal['model_sha256'] or not proposal['code'] or not proposal['model_id']:
        raise ValueError('Incomplete Vision lineage')


def evaluate_union(events,outputs):
    if set(outputs)!={e['event'] for e in events}:raise ValueError('Incomplete OOF coverage')
    results={}
    for k in BUDGETS:
        rows=[]
        for e in events:
            pp=outputs[e['event']]['proposals'][:k]
            if len(outputs[e['event']]['proposals'])>8:raise ValueError('Unbounded proposal list')
            combined,counts=union(e['candidates'],pp)
            truth=e['truth'];gt=e['gt_xy']
            err=lambda points:min((ds.distance(p,gt) for p in points),default=None) if truth=='HIT' else None
            ve=err([p['camera_xy'] for p in pp]);ce=err([c['xy'] for c in e['candidates']]);ue=err(combined)
            rows.append(dict(event=e['event'],session=e['session'],truth=truth,vision_error=ve,classic_error=ce,combined_error=ue,
                             proposals=len(pp),**counts,
                             not_near_gt5=sum(ds.distance(p['camera_xy'],gt)>5 for p in pp) if truth=='HIT' else None,
                             nuisance_gt42=sum(ds.distance(p['camera_xy'],gt)>42 for p in pp) if truth=='HIT' else None,
                             confidence=[p['confidence'] for p in pp]))
        def stats(rr):
            hits=[r for r in rr if r['truth']=='HIT'];errors=[r['vision_error'] for r in hits if r['vision_error'] is not None]
            oracle=lambda field:{str(t):sum(r[field] is not None and r[field]<=t for r in hits) for t in (5,10,20,42)}
            unique=[r['event'] for r in hits if r['combined_error'] is not None and r['combined_error']<=5 and (r['classic_error'] is None or r['classic_error']>5)]
            count=sum(r['proposals'] for r in rr)
            return dict(impacts=len(hits),classic_oracle=oracle('classic_error'),vision_oracle=oracle('vision_error'),combined_oracle=oracle('combined_error'),
                        unique_saves=unique,still_unavailable5=[r['event'] for r in hits if r['combined_error'] is None or r['combined_error']>5],
                        median_vision_nearest=float(np.median(errors)) if errors else None,
                        proposal_count=count,proposals_per_event=count/max(1,len(rr)),added_per_event=sum(r['added'] for r in rr)/max(1,len(rr)),
                        exact_duplicate_rate=sum(r['exact_duplicates'] for r in rr)/max(1,count),near_duplicate_rate=sum(r['near_duplicates_r3_5'] for r in rr)/max(1,count),
                        distractors_gt5=sum(r['not_near_gt5'] or 0 for r in rr),nuisance_gt42=sum(r['nuisance_gt42'] or 0 for r in rr),
                        controls=[r for r in rr if r['truth']=='NO_IMPACT'],unknown_truth=[r for r in rr if r['truth']=='UNKNOWN'])
        m=stats(rows);m['per_session']={s:stats([r for r in rows if r['session']==s]) for s in sorted({r['session'] for r in rows})}
        m['worst_session_combined5']=min(v['combined_oracle']['5']/v['impacts'] for v in m['per_session'].values() if v['impacts'])
        m['per_event']=rows;results[str(k)]=m
    return results


def mine_full_map_samples(model,data,cache,fold,cfg,norm,samples,stop=None):
    """Adapter of V2236's full-map negative miner to explicit physical events.

    Only TRAIN events are loaded. GT defines supervised negative exclusion, never
    inference peaks. Native NO_IMPACT may supply negatives; UNKNOWN is skipped.
    Reuses the original bounded maxima extractor and patch representation.
    """
    from src.engine.ai.training_v223.heatmap_model_v2236 import _local_maxima
    if set(fold['train'])&set(fold['held']+fold['dev']):raise ValueError('Mining fold overlap')
    for session in fold['train']+fold['held']+fold['dev']:ds.guard(session)
    if norm['training_sessions']!=fold['train']:raise ValueError('Mining scaler lineage')
    count=cfg.get('full_map_negatives',0)
    if not 0<=count<=64:raise ValueError('Full-map mining is bounded at 64/TRAIN event')
    if count==0:return list(samples),[]
    by_id={s.shot_id:s for s in samples};result=[];journal=[];stride=cfg['stride']
    for e in data['events']:
        if e['session'] not in fold['train'] or e['truth']=='UNKNOWN':continue
        if stop and stop():raise InterruptedError('TRAIN heatmap mining interrupted')
        s=by_id[e['event']];row=data['vision_rows'][e['event']]
        maps=event_maps(cache,row,stride,cfg.get('representation','native'))
        score=model.score_map(normalize(maps,norm));yy,xx=np.indices(score.shape)
        if e['truth']=='HIT':
            distances=np.hypot(xx*stride+(stride-1)*.5+row['origin'][0]-e['gt_xy'][0],
                               yy*stride+(stride-1)*.5+row['origin'][1]-e['gt_xy'][1])
        elif e['truth']=='NO_IMPACT':distances=np.full(score.shape,np.inf)
        else:raise ValueError('Unqualified negative truth')
        points=_local_maxima(score,top_k=count,nms_radius=max(1,int(np.ceil(6/stride))),allowed=distances>42).astype(int)
        patches=normalize(extract_grid_patches(maps,points),norm,patches=True)
        dd=distances[points[:,1],points[:,0]]
        result.append(HeatmapTrainingSampleV2236(s.session_id,s.shot_id,
            np.concatenate([s.patches,patches]),np.concatenate([s.distances,dd])))
        journal.append(dict(event=e['event'],session=e['session'],count=len(points),
            camera_xy=(points*stride+(stride-1)*.5+np.asarray(row['origin'])).tolist(),
            scores=score[points[:,1],points[:,0]].tolist(),training_sessions=fold['train'],
            held_sessions=fold['held'],truth=e['truth'],scope='TRAIN_ONLY_FULL_HEATMAP'))
    if {s.shot_id for s in result}!=set(by_id):raise ValueError('Mining sample coverage changed')
    return result,journal


def fit_proposal_model(model,data,cache,fold,cfg,norm,samples,path,*,seed,identity,stop=None):
    """Same optimizer, with optional bounded full-map mining/refinement stage.

    Epoch checkpoints preserve warmup and refinement separately. Mining is
    deterministic from the checkpointed warmup, so interrupted mining is rerun
    without using partially refined weights or changing its sample bank.
    """
    args={k:cfg[k] for k in ('epochs','learning_rate','l2','positive_radius','loss_kind','hard_count')}
    history=train_physical_proposals(model,samples,seed=seed,checkpoint=path/'training_checkpoint.npz',
                                    identity=identity,stop=stop,**args)
    refine=cfg.get('refine_epochs',0)
    if not refine:return history
    if not 1<=refine<=8:raise ValueError('Refinement bounded at 8 epochs')
    mined=[]
    if cfg.get('full_map_negatives',0):
        samples,mined=mine_full_map_samples(model,data,cache,fold,cfg,norm,samples,stop)
    ds.atomic_json(path/'full_map_mining.json',dict(identity=identity,events=mined,normalization=norm,
        model_before_mining=ds.canonical_hash({k:v.tolist() for k,v in model.arrays.items()})))
    args['epochs']=refine
    refinement=train_physical_proposals(model,samples,seed=seed+7001,checkpoint=path/'refinement_checkpoint.npz',
        identity=dict(parent=identity,mining_hash=ds.canonical_hash(mined)),stop=stop,**args)
    return dict(warmup=history,refinement=refinement,full_map_mining=mined)


def train_trial(directory,data,cache,cfg,stop):
    if cfg['budgets']!=list(BUDGETS):raise ValueError('Proposal budgets are frozen')
    outputs={};reports=[];identity=code_identity();cv2.setNumThreads(1)
    for i,fold in enumerate(data['folds']):
        path=directory/f'fold_{i:02d}';path.mkdir(exist_ok=True);done=path/'complete.json'
        if done.exists():
            record=ds.read(done)
            if record['config']!=cfg or record['cache_hash']!=data['dataset_hash'] or record['fold']!=fold:raise ValueError('Fold provenance changed')
            for name,h in record['model_hashes'].items():
                if ds.digest(path/name)!=h:raise ValueError('Fold model changed')
        else:
            if stop():raise InterruptedError('Vision deadline')
            samples,norm=training_samples(data,cache,fold,cfg)
            model=init_heatmap_model(cfg['kind'],hidden=cfg['hidden'],seed=cfg['seed']+i)
            training=fit_proposal_model(model,data,cache,fold,cfg,norm,samples,path,seed=cfg['seed']+i,
                         identity=dict(cache=data['dataset_hash'],fold=fold,config=cfg,normalization=norm),stop=stop)
            del samples
            model.metadata.update(config=cfg,normalization=norm,fold=fold,cache_hash=data['dataset_hash'],
                                  code=identity,live_authority=False,role='VISION_PROPOSAL')
            model.save(path);model_hash=ds.digest(path/'model.npz');pred={}
            for e in data['events']:
                if e['session'] not in fold['held']:continue
                if stop():raise InterruptedError('Model checkpoint retained before held scoring')
                row=data['vision_rows'][e['event']];maps=event_maps(cache,row,cfg['stride'],cfg.get('representation','native'))
                score=model.score_map(normalize(maps,norm))
                pp=proposals(score,stride=cfg['stride'],origin=row['origin'],nms_radius_px=cfg['nms_radius_px'])
                for p in pp:
                    p.update(event=e['event'],model_id=f'{directory.name}/fold_{i:02d}',training_sessions=fold['train'],
                             held_session=e['session'],model_sha256=model_hash,data_hash=data['dataset_hash'],
                             source_session=e['source_session'],classic_dataset_hash=data['classic_dataset_hash'],code=identity,
                             input_frames=dict(reference=row['reference_frames'],post=row['post_frames']),
                             causal_status=e['status'],stride=cfg['stride'])
                    validate_lineage(p,e,fold,data['dataset_hash'])
                # GT enters here ONLY for post-prediction diagnosis. Never
                # inserted into sampling/inference or retained proposal ranking.
                diagnostic=dict(reason='UNKNOWN_NO_PHYSICAL_LOCALIZATION_LABEL')
                if e['truth']=='HIT':
                    yy,xx=np.indices(score.shape)
                    gx=(e['gt_xy'][0]-row['origin'][0]-(cfg['stride']-1)*.5)/cfg['stride']
                    gy=(e['gt_xy'][1]-row['origin'][1]-(cfg['stride']-1)*.5)/cfg['stride']
                    near=((xx-gx)**2+(yy-gy)**2)*cfg['stride']**2<=25
                    maxima=score>=cv2.dilate(score,np.ones((3,3),np.uint8))-1e-7
                    exists=bool(np.any(near&maxima))
                    selected_near=any(ds.distance(p['camera_xy'],e['gt_xy'])<=5 for p in pp)
                    diagnostic=dict(reason='RECOVERED_TOP8' if selected_near else 'LOCAL_MAXIMUM_EXISTS_BELOW_TOP8_OR_NMS' if exists else 'NO_LEARNED_LOCAL_MAXIMUM_WITHIN5',
                                    best_within5_logit=float(score[near].max()) if near.any() else None,
                                    gt_region_in_crop=bool(near.any()),acquisition_time_known=False,
                                    temporal_qualification='fixed early reference; not independently qualified clean PRE')
                pred[e['event']]=dict(proposals=pp,diagnostic=diagnostic)
            record=dict(config=cfg,cache_hash=data['dataset_hash'],fold=fold,outputs=pred,training=training,
                        model_hashes={n:ds.digest(path/n) for n in ('model.npz','model.json')})
            ds.atomic_json(done,record)
        expected={e['event']:e for e in data['events'] if e['session'] in fold['held']}
        if set(expected)!=set(record['outputs']):raise ValueError('Incomplete held fold')
        for event,pred in record['outputs'].items():
            for p in pred['proposals']:validate_lineage(p,expected[event],fold,data['dataset_hash'])
        if set(outputs)&set(record['outputs']):raise ValueError('Duplicate OOF event')
        outputs.update(record['outputs']);reports.append(dict(fold=fold,model_hashes=record['model_hashes'],training=record['training']))
    return dict(metrics=evaluate_union(data['events'],outputs),outputs=outputs,folds=reports,
                evaluation='OUT_OF_SESSION_DEVELOPMENT_CV_NOT_UNTOUCHED_VALIDATION')


def policy(m):
    controls=m['controls']+m['unknown_truth']
    return (m['combined_oracle']['5'],len(m['unique_saves']),m['worst_session_combined5'],m['combined_oracle']['10'],
            -(m['median_vision_nearest'] if m['median_vision_nearest'] is not None else float('inf')),
            -m['distractors_gt5'],-m['proposal_count'],-sum(c['proposals'] for c in controls))


def compare(metrics,previous):
    if previous is None:return dict(accepted=True,reason='first bounded OOF model; not a gain claim')
    names=('combined5','unique_saves5','worst_session_combined5','combined10',
           'median_vision_nearest','distractors_gt5','proposal_count','control_unknown_proposals')
    a,b=policy(metrics),policy(previous)
    for name,new,old in zip(names,a,b):
        if new!=old:return dict(accepted=new>old,reason='first differing fixed-budget criterion: '+name,
                                criterion_value=new,incumbent_value=old,values_are_policy_oriented=True)
    return dict(accepted=False,reason='identical fixed-budget policy values')


def publish(root,result,data):
    state=ds.read(root/'champion.json');decisions={}
    if result['status']=='COMPLETE':
        for k in map(str,BUDGETS):
            old=state['tracks'].get(k);metrics=result['metrics'][k]
            decisions[k]=compare(metrics,old['metrics'] if old else None);accept=decisions[k]['accepted']
            if accept:
                entry=dict(trial=result['trial'],budget=int(k),metrics=metrics,cache_hash=data['dataset_hash'],
                           model_directory=str(root/'trials'/result['trial']),live_authority=False)
                target=root/'champions'/f'top{k}'/(result['trial']+'.json')
                if target.exists() and ds.read(target)!=entry:raise ValueError('Vision champion history overwrite')
                if not target.exists():ds.atomic_json(target,entry)
                state['tracks'][k]=entry;state['history'][k].append(result['trial'])
        state['champion']=state['tracks']['8']['trial'];ds.atomic_json(root/'champion.json',state)
    with (root/'progress.jsonl').open('a') as f:
        f.write(json.dumps(dict(trial=result['trial'],status=result['status'],decisions=decisions))+'\n')
        f.flush()
    return state


def refresh(root,data):
    from automation import physical_ai_lab as lab
    records=lab.completed(root)
    state=dict(champion=None,tracks={},history={str(k):[] for k in BUDGETS},live_authority=False,role='VISION_PROPOSAL')
    leaderboard=[]
    for r in records:
        if r['status']!='COMPLETE':continue
        for k in map(str,BUDGETS):
            old=state['tracks'].get(k);m=r['metrics'][k];decision=compare(m,old['metrics'] if old else None);accepted=decision['accepted']
            leaderboard.append(dict(trial=r['trial'],budget=int(k),combined5=m['combined_oracle']['5'],unique_saves=m['unique_saves'],**decision))
            if accepted:
                entry=dict(trial=r['trial'],budget=int(k),metrics=m,cache_hash=data['dataset_hash'],
                           model_directory=str(root/'trials'/r['trial']),live_authority=False)
                target=root/'champions'/f'top{k}'/(r['trial']+'.json')
                if target.exists() and ds.read(target)!=entry:raise ValueError('Historical Vision champion changed')
                if not target.exists():ds.atomic_json(target,entry)
                state['tracks'][k]=entry;state['history'][k].append(r['trial'])
    if state['tracks']:state['champion']=state['tracks']['8']['trial']
    ds.atomic_json(root/'champion.json',state);ds.atomic_json(root/'leaderboard.json',leaderboard)
    top=sorted((r for r in records if r['status']=='COMPLETE'),key=lambda r:policy(r['metrics']['8']),reverse=True)[:5]
    unresolved=sorted(set.intersection(*(set(r['metrics']['8']['still_unavailable5']) for r in top))) if top else []
    diagnostics={e:[] for e in unresolved}
    for r in top:
        full=ds.read(root/'trials'/r['trial']/'result.json')
        for e in unresolved:diagnostics[e].append(dict(trial=r['trial'],**full['outputs'][e]['diagnostic']))
    ds.atomic_json(root/'failure_wall.json',dict(completed_trials=sum(r['status']=='COMPLETE' for r in records),
                  common_unavailable5=unresolved,top_models=[r['trial'] for r in top],diagnostics=diagnostics,
                  interpretation='OOF proposal coverage, not selected accuracy; no low-contrast or motion reason invented from distance',
                  best_after={str(n):max(r['metrics']['8']['combined_oracle']['5'] for r in records[:n] if r['status']=='COMPLETE') for n in (100,1000,len(records)) if n and len(records)>=n and any(r['status']=='COMPLETE' for r in records[:n])}))
    ds.atomic_json(root/'journal.json',[{k:r[k] for k in ('trial','status','finished_at','error') if k in r} for r in records])
    return state


def lab_main(argv):
    from automation import physical_ai_lab as lab
    import sys
    ds.install_guard();p=argparse.ArgumentParser(description='Bounded OOF Vision proposals; no live authority')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('vision-prepare');q.add_argument('--dataset',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    for cmd in ('vision-run','vision-resume','vision-status'):
        q=sub.add_parser(cmd);q.add_argument('--experiment',required=True);q.add_argument('--root',type=Path,default=lab.ROOT)
        if cmd!='vision-status':q.add_argument('--hours',type=float);q.add_argument('--max-trials',type=int)
        if cmd=='vision-run':q.add_argument('--dataset',type=Path,required=True);q.add_argument('--seed',type=int,default=2361)
    a=p.parse_args(argv)
    if a.command=='vision-prepare':
        print(prepare(a.dataset,a.output,native=True)['cache_hash']);return
    root=lab.location(a.root,a.experiment)
    if a.command=='vision-status':
        c=ds.read(root/'champion.json')
        print(json.dumps(dict(champions={k:dict(trial=v['trial'],combined=v['metrics']['combined_oracle'],unique_saves=v['metrics']['unique_saves']) for k,v in c['tracks'].items()},history=c['history'],failure_wall=ds.read(root/'failure_wall.json')),indent=2));return
    if a.command=='vision-run' and a.hours is None:p.error('--hours required')
    lab.run(root,a.dataset if a.command=='vision-run' else None,hours=a.hours,seed=getattr(a,'seed',2361),
            max_trials=a.max_trials,resume=a.command=='vision-resume',adapter=sys.modules[__name__])


if __name__=='__main__': main()
