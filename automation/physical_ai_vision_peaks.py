"""Bounded peak diagnostics/reranking adapter for the existing offline Vision lab.

This module never emits runtime hits or changes camera coordinates.
"""
from pathlib import Path
import numpy as np
import cv2
from automation import physical_ai_dataset as ds, physical_ai_vision as vision


def raw_peaks(score):
    """Same stable 3x3 maxima/order as V2236; no NMS or truth input."""
    if score.ndim!=2 or not np.isfinite(score).all():raise ValueError('Finite 2D heatmap required')
    y,x=np.nonzero(score>=cv2.dilate(score,np.ones((3,3),np.uint8))-1e-7)
    order=np.argsort(-score[y,x],kind='stable')
    return np.column_stack([x[order],y[order]]),score[y[order],x[order]]


def nms_trace(points,radius,limit=None):
    """Greedy NMS in supplied order; grid only accelerates identical comparisons.

    Returns accepted source indices, and first suppressor source index for each
    rejected point. Diagnostics may inspect all peaks; inference uses <=32 only.
    """
    if radius<=0:raise ValueError('Positive NMS radius required')
    cells={};accepted=[];suppressed=np.full(len(points),-2,int);r2=radius*radius
    for i,p in enumerate(points):
        cx,cy=np.floor(p/radius).astype(int);near=[]
        for dx in (-1,0,1):
            for dy in (-1,0,1):near.extend(cells.get((cx+dx,cy+dy),()))
        hits=[j for j in near if np.sum((p-points[j])**2)<r2]
        if hits:suppressed[i]=min(hits)
        else:
            accepted.append(i);suppressed[i]=-1;cells.setdefault((cx,cy),[]).append(i)
            if limit is not None and len(accepted)>=limit:break
    return np.asarray(accepted,int),suppressed


def audit_peak(score,*,stride,origin,nms_radius_px,gt_xy):
    """Post-prediction diagnosis only. GT never changes raw ordering or NMS."""
    points,values=raw_peaks(score);radius=max(1,int(np.ceil(nms_radius_px/stride)))
    accepted,suppressors=nms_trace(points,radius)
    camera=points*stride+(stride-1)*.5+np.asarray(origin)
    distance=np.linalg.norm(camera-np.asarray(gt_xy),axis=1)
    ranks={int(v):i+1 for i,v in enumerate(accepted)}
    def detail(i):
        j=int(suppressors[i]);after=ranks.get(i)
        return dict(camera_xy=camera[i].tolist(),error=float(distance[i]),logit=float(values[i]),raw_rank=i+1,
                    higher_logit_count=int(np.sum(values>values[i])),earlier_ordered_count=i,
                    survived_nms=j<0,post_nms_rank=after,
                    suppressor=None if j<0 else dict(camera_xy=camera[j].tolist(),logit=float(values[j]),raw_rank=j+1,
                                                   distance=float(np.linalg.norm(camera[i]-camera[j])),post_nms_rank=ranks[j]),
                    failure_class='NMS_SUPPRESSED' if j>=0 else 'BELOW_TOPK' if after>8 else 'IN_TOP8',
                    raw_below_top8=bool(i>=8))
    good=np.flatnonzero(distance<=5)
    return dict(raw_count=len(points),nms_count=len(accepted),nearest=detail(int(np.argmin(distance))),
                best_good=detail(int(good[0])) if len(good) else None,
                internal_coverage={str(n):bool(np.any(distance[:n]<=5)) for n in (16,24,32)},
                top8_xy=camera[accepted[:8]].tolist())


# This is a peak-stage adapter, not Evidence Fusion: same V223 MLP implementation,
# independent model role, and unchanged final Vision budgets.
import argparse
import json
import sys
from automation import physical_ai_lab as lab
from src.engine.ai.training_v223.heatmap_model_v2236 import HeatmapModelV2236, init_heatmap_model, train_physical_proposals
from src.engine.ai.training_v223.model import RankModelV223, train_fusion_model
from src.engine.ai.canonical_challenger import transform_matrix
from src.engine.ai.training_v223.evidence_patch_v2235 import EVIDENCE_CHANNEL_NAMES

POOL=32
ORDERS=('rerank_before_nms','nms_before_rerank')
PROTECTED=('KOMP:1','KOMP:5','S02:5','S02:11','S02:12','S02:13')
FEATURES=('logit','heat_contrast','heat_sharpness','raw_rank','stride','stronger_distance','neighbors_r12','border_distance')+tuple('mean_'+s for s in EVIDENCE_CHANNEL_NAMES)+tuple('contrast_'+s for s in EVIDENCE_CHANNEL_NAMES)
SCHEMA='vision-peak-local-1'


def code_identity():
    value=vision.code_identity();value['role']='VISION_PEAK_RERANK'
    value['source_hashes']['automation/physical_ai_vision_peaks.py']=ds.digest(Path(__file__))
    return value


def extract_pool(score,maps,*,stride,origin,nms_radius_px,order):
    """Only images, model output and recorded geometry; no event truth input.

    Raw order retains 32 before suppression; NMS-first retains 32 survivors.
    These are deliberately different bounded pools, not a claim of identical
    candidate sets across order variants. Neither pool is emitted downstream.
    """
    if order not in ORDERS:raise ValueError('Unknown peak ordering')
    xy,logit=raw_peaks(score);radius=max(1,int(np.ceil(nms_radius_px/stride)))
    ids=np.arange(min(POOL,len(xy))) if order==ORDERS[0] else nms_trace(xy,radius,POOL)[0]
    pts=xy[ids];camera=pts*stride+(stride-1)*.5+np.asarray(origin)
    distances=np.linalg.norm(camera[:,None]-camera[None,:],axis=2)
    peaks=[];h,w=score.shape
    for local,(idx,(x,y)) in enumerate(zip(ids,pts)):
        # Same registered V2235 maps; measurements remain correlated channels.
        # Clipped local windows; distance-to-border records their context.
        xs=np.clip(np.arange(x-3,x+4),0,w-1);ys=np.clip(np.arange(y-3,y+4),0,h-1)
        patch=maps[:,ys[:,None],xs[None,:]].astype(float)/255
        heat=score[ys[:,None],xs[None,:]]
        ring=np.ones((7,7),bool);ring[2:5,2:5]=False
        means=patch[:,2:5,2:5].mean((1,2));contrasts=means-patch[:,ring].mean(1)
        stronger=distances[local,:local]
        raw=[float(logit[idx]),float(logit[idx]-heat[ring].mean()),float(logit[idx]-np.mean([heat[2,3],heat[4,3],heat[3,2],heat[3,4]])),
             int(idx)+1,stride,float(stronger.min()) if len(stronger) else float(np.hypot(h,w)*stride),
             int(np.sum((distances[local]>0)&(distances[local]<=12))),float(min(x,y,w-1-x,h-1-y)*stride)]
        raw+=means.tolist()+contrasts.tolist()
        peaks.append(dict(camera_xy=camera[local].tolist(),confidence=float(logit[idx]),source='VISION',
                          raw_rank=int(idx)+1,rank=local+1,pool_rank=local+1,
                          nms_rank=local+1 if order==ORDERS[1] else None,
                          stride=stride,feature_schema=SCHEMA,
                          peak_features=dict(zip(FEATURES,raw))))
    return peaks


def feature_matrix(peaks,subset='all'):
    names=FEATURES if subset=='all' else FEATURES[:8] if subset=='vision' else FEATURES[8:] if subset=='change' else None
    if names is None:raise ValueError('Unknown feature subset')
    x=np.asarray([[p['peak_features'][n] for n in names] for p in peaks],np.float32).reshape(len(peaks),len(names))
    if not np.isfinite(x).all():raise ValueError('Nonfinite peak feature')
    return names,transform_matrix(x,'signed_log')


def select(peaks,scores,*,order,nms_radius_px):
    if len(peaks)>POOL or len(scores)!=len(peaks) or not np.isfinite(scores).all():raise ValueError('Invalid bounded peak scores')
    ranking=np.argsort(-np.asarray(scores),kind='stable');suppression={};survivors=[]
    if order==ORDERS[0]:
        xy=np.asarray([peaks[i]['camera_xy'] for i in ranking])
        stride=peaks[0]['stride'] if peaks else 1
        if any(p['stride']!=stride for p in peaks):raise ValueError('Mixed peak strides')
        radius=max(1,int(np.ceil(nms_radius_px/stride)))*stride
        keep,sup=nms_trace(xy,radius) if len(xy) else ([],[])
        survivors=[int(ranking[i]) for i in keep]
        suppression={int(ranking[i]):int(ranking[j]) for i,j in enumerate(sup) if j>=0}
    elif order==ORDERS[1]:survivors=ranking.tolist()
    else:raise ValueError('Unknown peak ordering')
    detail=[dict(raw_rank=p['raw_rank'],reranked_rank=int(np.flatnonzero(ranking==i)[0])+1,
                 survived_nms=i not in suppression,suppressor_pool_index=suppression.get(i),
                 output_rank=survivors.index(i)+1 if i in survivors else None) for i,p in enumerate(peaks)]
    output=[dict(peaks[i],rank=rank+1,reranked_rank=detail[i]['reranked_rank'],reranker_score=float(scores[i])) for rank,i in enumerate(survivors[:8])]
    return output,detail


def validate_upstream(peak,event,outer,*,training):
    """Actual cache training is nested; reject metadata inconsistent with it."""
    ds.guard(event['session']);ds.guard(peak['held_session'])
    for s in peak['training_sessions']:ds.guard(s)
    sessions=set(peak['training_sessions']);allowed=set(outer['train'])
    if peak['held_session']!=event['session'] or peak['event']!=event['event']:raise ValueError('Foreign peak event')
    if event['session'] in sessions or not sessions<=allowed:raise ValueError('Nested upstream session leakage')
    expected=allowed-{event['session']} if training else allowed
    if sessions!=expected:raise ValueError('Wrong nested training lineage')
    if not peak['model_sha256'] or not peak['base_data_hash']:raise ValueError('Missing base lineage')


def _extract_events(data,cache,events,model,model_dir,cfg,model_id,train_sessions):
    result={};model_hash=ds.digest(model_dir/'model.npz')
    for e in events:
        row=data['vision_rows'][e['event']];maps=vision.event_maps(cache,row,cfg['stride'],cfg.get('representation','native'))
        score=model.score_map(vision.normalize(maps,model.metadata['normalization']))
        # Reranker measurements retain the original eight-channel semantics,
        # even when the upstream Vision model uses a context representation.
        evidence_maps=vision.event_maps(cache,row,cfg['stride'])
        modes={}
        for order in ORDERS:
            peaks=extract_pool(score,evidence_maps,stride=cfg['stride'],origin=row['origin'],nms_radius_px=cfg['nms_radius_px'],order=order)
            for p in peaks:p.update(event=e['event'],model_id=model_id,model_sha256=model_hash,
                training_sessions=list(train_sessions),held_session=e['session'],source_session=e['source_session'],
                base_data_hash=data['dataset_hash'],input_frames=dict(reference=row['reference_frames'],post=row['post_frames']),
                causal_status=e['status'],base_code=model.metadata['code'],stride=cfg['stride'],
                representation_id=cfg.get('representation','native'),normalization_training_sessions=model.metadata['normalization']['training_sessions'],
                mining_training_sessions=list(train_sessions))
            modes[order]=peaks
        result[e['event']]=modes
    return result


def prepare(experiment,trial,output,*,resume=False,stop=None):
    """Cache 15 genuinely cross-fitted TRAIN models, never reuse global OOF rows.

    Outer held/development sessions are absent from every inner fit. Source outer
    models are used only for their own held predictions, preserving the baseline.
    """
    ds.guard(experiment);ds.guard(trial);output=ds.output_root(output)
    if Path(trial).name!=trial:raise ValueError('Use explicit trial id')
    run=ds.read(Path(experiment)/'manifest.json');data,cache=vision.load(run['dataset'])
    source=Path(experiment)/'trials'/trial;result=ds.read(source/'result.json');cfg=result['config']
    if result['dataset_hash']!=data['dataset_hash']:raise ValueError('Source trial/cache mismatch')
    intent=dict(schema=SCHEMA,pool=POOL,orders=list(ORDERS),experiment=str(Path(experiment).resolve()),trial=trial,
                base_result_sha256=ds.digest(source/'result.json'),vision_cache=str(cache),vision_cache_hash=data['dataset_hash'],
                config=cfg,code=code_identity())
    if resume:
        if ds.read(output/'intent.json')!=intent:raise ValueError('Peak preparation changed')
    else:output.mkdir(parents=True,exist_ok=False);ds.atomic_json(output/'intent.json',intent)
    cv2.setNumThreads(1);shards=[]
    for i,fold in enumerate(data['folds']):
        if stop and stop():raise InterruptedError('Nested peak preparation interrupted')
        path=output/f'fold_{i:02d}';path.mkdir(exist_ok=True);done=path/'peaks.json'
        if done.exists():
            saved=ds.read(done)
            if saved['intent_hash']!=ds.canonical_hash(intent):raise ValueError('Cached fold changed')
        else:
            outer_dir=source/f'fold_{i:02d}';outer=HeatmapModelV2236.load(outer_dir)
            source_fold=ds.read(outer_dir/'complete.json')
            for n,h in source_fold['model_hashes'].items():
                if ds.digest(outer_dir/n)!=h:raise ValueError('Source outer model hash changed')
            if outer.metadata['fold']!=fold or outer.metadata['normalization']['training_sessions']!=fold['train']:raise ValueError('Source outer split changed')
            held=[e for e in data['events'] if e['session'] in fold['held']]
            hp=_extract_events(data,cache,held,outer,outer_dir,cfg,f'{trial}/fold_{i:02d}',fold['train']);training={};inner_models=[]
            for eid,modes in hp.items():
                if [p['camera_xy'] for p in modes[ORDERS[1]][:8]]!=[p['camera_xy'] for p in result['outputs'][eid]['proposals']]:
                    raise ValueError('Source held heatmap no longer reproduces original top8')
            for j,session in enumerate(fold['train']):
                inner=dict(train=[s for s in fold['train'] if s!=session],held=[session],dev=fold['held']+fold['dev'])
                inner_dir=path/f'inner_{j:02d}';inner_dir.mkdir(exist_ok=True)
                complete=inner_dir/'complete.json'
                identity=dict(intent=intent,outer=fold,inner=inner,seed=cfg['seed']+100*i+j)
                if complete.exists():
                    record=ds.read(complete)
                    if record['identity']!=identity:raise ValueError('Inner fold identity changed')
                    for n,h in record['model_hashes'].items():
                        if ds.digest(inner_dir/n)!=h:raise ValueError('Inner model changed')
                    model=HeatmapModelV2236.load(inner_dir)
                else:
                    samples,norm=vision.training_samples(data,cache,inner,cfg)
                    assert all(s.session_id in inner['train'] for s in samples)
                    model=init_heatmap_model(cfg['kind'],hidden=cfg['hidden'],seed=identity['seed'])
                    history=vision.fit_proposal_model(model,data,cache,inner,cfg,norm,samples,inner_dir,
                        seed=identity['seed'],identity=identity,stop=stop)
                    model.metadata.update(normalization=norm,fold=inner,outer_fold=fold,code=code_identity(),role='VISION_NESTED_TRAIN_PEAKS',live_authority=False)
                    model.save(inner_dir)
                    record=dict(identity=identity,history=history,model_hashes={n:ds.digest(inner_dir/n) for n in ('model.json','model.npz')})
                    ds.atomic_json(complete,record)
                inner_events=[e for e in data['events'] if e['session']==session]
                training.update(_extract_events(data,cache,inner_events,model,inner_dir,cfg,str(inner_dir),inner['train']))
                inner_models.append(dict(directory=str(inner_dir),**record))
            saved=dict(intent_hash=ds.canonical_hash(intent),fold=fold,train=training,held=hp,inner_models=inner_models,
                       outer_model=dict(directory=str(outer_dir),model_hashes=source_fold['model_hashes']))
            ds.atomic_json(done,saved)
        shards.append(dict(path=str(done.relative_to(output)),sha256=ds.digest(done)))
        print('peak cache',i,fold['held'],'complete',flush=True)
    manifest=dict(intent=intent,folds=shards,baseline_metrics=result['metrics'],baseline_outputs=result['outputs'],live_authority=False)
    manifest['dataset_hash']=ds.canonical_hash(manifest);ds.atomic_json(output/'peaks.json',manifest)
    return manifest


def load(cache):
    cache=Path(cache);manifest=ds.read(cache/'peaks.json');expected=manifest.pop('dataset_hash')
    if ds.canonical_hash(manifest)!=expected:raise ValueError('Peak cache manifest changed')
    if manifest['intent']['schema']!=SCHEMA or manifest['intent']['pool']!=POOL:raise ValueError('Peak schema changed')
    data,_=vision.load(manifest['intent']['vision_cache']);events={e['event']:e for e in data['events']};shards=[]
    if data['dataset_hash']!=manifest['intent']['vision_cache_hash']:raise ValueError('Upstream visual cache changed')
    if len(manifest['folds'])!=len(data['folds']):raise ValueError('Incomplete outer folds')
    for i,item in enumerate(manifest['folds']):
        ds.guard(item['path']);p=(cache/item['path']).resolve()
        if not p.is_relative_to(cache.resolve()) or ds.digest(p)!=item['sha256']:raise ValueError('Peak shard identity changed')
        shard=ds.read(p);fold=data['folds'][i]
        if shard['fold']!=fold:raise ValueError('Outer fold changed')
        producers={}
        for model in shard['inner_models']+[shard['outer_model']]:
            for n,h in model['model_hashes'].items():
                if ds.digest(Path(model['directory'])/n)!=h:raise ValueError('Upstream model changed')
            meta=ds.read(Path(model['directory'])/'model.json')['metadata']
            mf=meta['fold']
            if meta['normalization']['training_sessions']!=mf['train']:raise ValueError('Upstream scaler lineage mismatch')
            producers[model['model_hashes']['model.npz']]=mf
        for group in ('train','held'):
            expected_events={e['event'] for e in data['events'] if e['session'] in fold[group]}
            if set(shard[group])!=expected_events:raise ValueError('Peak event coverage changed')
            for eid,modes in shard[group].items():
                if set(modes)!=set(ORDERS):raise ValueError('Missing NMS variant')
                for peaks in modes.values():
                    if len(peaks)>POOL:raise ValueError('Internal peak cap exceeded')
                    feature_matrix(peaks)
                    for peak in peaks:
                        validate_upstream(peak,events[eid],fold,training=group=='train')
                        mf=producers.get(peak['model_sha256'])
                        if mf is None or mf['train']!=peak['training_sessions'] or events[eid]['session'] not in mf['held']:
                            raise ValueError('Peak/model lineage mismatch')
                        if peak['base_data_hash']!=manifest['intent']['vision_cache_hash']:raise ValueError('Wrong peak input hash')
        shards.append(shard)
    if len(shards)!=len(data['folds']):raise ValueError('Incomplete outer folds')
    return dict(data,dataset_hash=expected,peak_manifest=manifest),shards


def config_for(seed,trial):
    # First pair is a predeclared NMS-order ablation; all NN settings identical.
    if trial<2:return dict(seed=seed,hidden=16,epochs=24,learning_rate=.003,l2=.001,hard_negative_count=4,
                           subset='all',order=ORDERS[trial],pool=POOL)
    rng=np.random.default_rng(np.random.SeedSequence([seed,trial]))
    return dict(seed=int(rng.integers(1,2**30)),hidden=int(rng.choice([8,16,32])),epochs=int(rng.choice([12,24,48])),
                learning_rate=float(10**rng.uniform(-3.5,-2)),l2=float(10**rng.uniform(-4,-2)),
                hard_negative_count=int(rng.choice([0,4,8])),subset=str(rng.choice(['all','vision','change'])),
                order=str(rng.choice(ORDERS)),pool=POOL)


def samples(data,shard,cfg):
    events={e['event']:e for e in data['events']};result=[];ids=[]
    for eid,modes in shard['train'].items():
        e=events[eid]
        if e['session'] not in shard['fold']['train']:raise ValueError('Held event in reranker training')
        if e['truth']=='UNKNOWN':continue
        peaks=modes[cfg['order']]
        for peak in peaks:validate_upstream(peak,e,shard['fold'],training=True)
        names,x=feature_matrix(peaks,cfg['subset'])
        distance=np.array([ds.distance(p['camera_xy'],e['gt_xy']) for p in peaks]) if e['truth']=='HIT' else np.full(len(peaks),np.inf)
        result.append((x,distance));ids.append(eid)
    return names,result,ids


def train_trial(directory,data,shards,cfg,stop):
    if cfg['pool']!=POOL:raise ValueError('Internal pool is frozen')
    outputs={};reports=[];events={e['event']:e for e in data['events']}
    for i,shard in enumerate(shards):
        fold=shard['fold'];path=directory/f'fold_{i:02d}';path.mkdir(exist_ok=True);done=path/'complete.json'
        identity=dict(cache=data['dataset_hash'],fold=fold,config=cfg)
        if done.exists():
            saved=ds.read(done)
            if saved['identity']!=identity:raise ValueError('Reranker fold changed')
            for n,h in saved['model_hashes'].items():
                if ds.digest(path/n)!=h:raise ValueError('Reranker model changed')
        else:
            if stop():raise InterruptedError('Peak reranker budget')
            names,training,ids=samples(data,shard,cfg)
            model,history=train_fusion_model(training,feature_names=names,seed=cfg['seed']+i,checkpoint=path/'checkpoint.npz',
                checkpoint_identity=identity,stop=stop,**{k:cfg[k] for k in ('hidden','epochs','learning_rate','l2','hard_negative_count')})
            model.metadata.update(role='VISION_PEAK_RERANK',outer_fold=fold,training_events=ids,feature_schema=SCHEMA,
                                  nested_upstream=True,code=code_identity(),cache_hash=data['dataset_hash'])
            model.save(path);mh=ds.digest(path/'model.npz');pred={}
            for eid,modes in shard['held'].items():
                peaks=modes[cfg['order']];_,x=feature_matrix(peaks,cfg['subset']);scores=model.score_matrix(x)
                selected,detail=select(peaks,scores,order=cfg['order'],nms_radius_px=data['peak_manifest']['intent']['config']['nms_radius_px'])
                for p in selected:p.update(vision_training_sessions=p['training_sessions'],base_vision_model_id=p['model_id'],
                    reranker_model_id=f'{directory.name}/fold_{i:02d}',reranker_model_sha256=mh,reranker_training_sessions=fold['train'],
                    data_hash=data['dataset_hash'],code=code_identity(),confidence_semantics='raw Vision logit; reranker_score is also uncalibrated')
                e=events[eid];diagnostic=dict(reason='NO_PHYSICAL_LOCALIZATION_LABEL')
                if e['truth']=='HIT':
                    distances=[ds.distance(p['camera_xy'],e['gt_xy']) for p in peaks];j=int(np.argmin(distances)) if distances else None
                    good=[k for k,d in enumerate(distances) if d<=5]
                    in3=any(ds.distance(p['camera_xy'],e['gt_xy'])<=5 for p in selected[:3])
                    best=min(good,key=lambda k:detail[k]['reranked_rank']) if good else None
                    reason='IN_FINAL_TOP3' if in3 else 'NO_GOOD_PEAK_IN_INTERNAL_POOL' if not good else 'GOOD_PEAK_NMS_SUPPRESSED' if all(not detail[k]['survived_nms'] for k in good) else 'GOOD_PEAK_EXISTS_BUT_RANKED_LOW'
                    diagnostic=dict(reason=reason,nearest_error=distances[j] if j is not None else None,
                        nearest=None if j is None else dict(peaks[j],**detail[j]),best_good=None if best is None else dict(peaks[best],**detail[best]),
                        temporal_qualification='TEMPORAL_INPUT_UNQUALIFIED; first3/last3 registered maps, no onset/revocation query',
                        good_peak_count=len(good))
                pred[eid]=dict(proposals=selected,diagnostic=diagnostic,pool_details=detail)
            saved=dict(identity=identity,outputs=pred,training=history,training_events=ids,model_hashes={n:ds.digest(path/n) for n in ('model.json','model.npz')})
            ds.atomic_json(done,saved)
        outputs.update(saved['outputs']);reports.append({k:saved[k] for k in ('identity','model_hashes','training','training_events')})
    metrics=vision.evaluate_union(data['events'],outputs);baseline=data['peak_manifest']['baseline_metrics'];delta={}
    for k in ('1','3','8'):
        def good(m):return {e['event'] for e in m['per_event'] if e['truth']=='HIT' and e['combined_error'] is not None and e['combined_error']<=5}
        a,b=good(baseline[k]),good(metrics[k])
        delta[k]=dict(saves=sorted(b-a),regressions=sorted(a-b),
                      protected_lost=sorted((set(PROTECTED)&a)-b),
                      protected_not_covered=sorted(set(PROTECTED)-b))
    return dict(metrics=metrics,outputs=outputs,folds=reports,delta=delta,evaluation='NESTED_SESSION_OOF_DEVELOPMENT_ONLY')


def _initial(data):
    base=data['peak_manifest'];trial='frozen_base_'+base['intent']['trial']
    tracks={str(k):dict(trial=trial,budget=k,metrics=base['baseline_metrics'][str(k)],live_authority=False) for k in vision.BUDGETS}
    return dict(champion=trial,tracks=tracks,history={str(k):[] for k in vision.BUDGETS},live_authority=False)


def _compare(m,old,k,data):
    protected=set(data['peak_manifest']['baseline_metrics'][k]['unique_saves'])
    losses=protected-set(m['unique_saves'])
    if losses:return dict(accepted=False,reason='lost existing Vision saves: '+','.join(sorted(losses)))
    return vision.compare(m,old)


def _accept(root,state,result,data):
    decisions={}
    if result['status']!='COMPLETE':return dict(error=result.get('error'))
    for k in map(str,vision.BUDGETS):
        m=result['metrics'][k];decision=_compare(m,state['tracks'][k]['metrics'],k,data);decisions[k]=decision
        if decision['accepted']:
            entry=dict(trial=result['trial'],budget=int(k),metrics=m,cache_hash=data['dataset_hash'],
                       model_directory=str(root/'trials'/result['trial']),live_authority=False)
            target=root/'champions'/f'top{k}'/(result['trial']+'.json')
            if target.exists() and ds.read(target)!=entry:raise ValueError('Peak champion overwrite')
            if not target.exists():ds.atomic_json(target,entry)
            state['tracks'][k]=entry;state['history'][k].append(result['trial'])
    state['champion']=state['tracks']['3']['trial'];return decisions


def publish(root,result,data):
    state=ds.read(root/'champion.json');decisions=_accept(root,state,result,data)
    ds.atomic_json(root/'champion.json',state)
    with (root/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(trial=result['trial'],decisions=decisions))+'\n')
    return state


def refresh(root,data):
    state=_initial(data);records=lab.completed(root);leaderboard=[]
    for r in records:leaderboard.append(dict(trial=r['trial'],status=r['status'],decisions=_accept(root,state,r,data)))
    ds.atomic_json(root/'champion.json',state);ds.atomic_json(root/'leaderboard.json',leaderboard)
    top=sorted((r for r in records if r['status']=='COMPLETE'),key=lambda r:vision.policy(r['metrics']['3']),reverse=True)[:5]
    unresolved=sorted(set.intersection(*(set(r['metrics']['3']['still_unavailable5']) for r in top))) if top else []
    diagnoses={e:[] for e in unresolved}
    for r in top:
        full=ds.read(root/'trials'/r['trial']/'result.json')
        for e in unresolved:diagnoses[e].append(dict(trial=r['trial'],**full['outputs'][e]['diagnostic']))
    ds.atomic_json(root/'failure_wall.json',dict(common_unavailable5=unresolved,diagnostics=diagnoses,completed_trials=sum(r['status']=='COMPLETE' for r in records),
        top_models=[r['trial'] for r in top],baseline_top3=data['peak_manifest']['baseline_metrics']['3']['combined_oracle'],
        best_after={str(n):max(r['metrics']['3']['combined_oracle']['5'] for r in records[:n] if r['status']=='COMPLETE') for n in (100,1000,len(records)) if n and len(records)>=n and any(r['status']=='COMPLETE' for r in records[:n])}))
    ds.atomic_json(root/'journal.json',[{k:r[k] for k in ('trial','status','finished_at','error') if k in r} for r in records]);return state


def main(argv=None):
    ds.install_guard();parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('peak-prepare');q.add_argument('--vision-experiment',type=Path,required=True);q.add_argument('--base-trial',required=True)
    q.add_argument('--output',type=Path,required=True);q.add_argument('--resume',action='store_true')
    for name in ('peak-run','peak-resume','peak-status'):
        q=sub.add_parser(name);q.add_argument('--experiment',required=True);q.add_argument('--root',type=Path,default=lab.ROOT)
        if name!='peak-status':q.add_argument('--hours',type=float);q.add_argument('--max-trials',type=int)
        if name=='peak-run':q.add_argument('--dataset',type=Path,required=True);q.add_argument('--seed',type=int,default=2371)
    a=parser.parse_args(argv)
    if a.cmd=='peak-prepare':print(prepare(a.vision_experiment,a.base_trial,a.output,resume=a.resume)['dataset_hash']);return
    root=lab.location(a.root,a.experiment)
    if a.cmd=='peak-status':
        c=ds.read(root/'champion.json');print(json.dumps(dict(champions={k:v['trial'] for k,v in c['tracks'].items()},history=c['history'],failure_wall=ds.read(root/'failure_wall.json')),indent=2));return
    if a.cmd=='peak-run' and a.hours is None:parser.error('--hours required')
    lab.run(root,getattr(a,'dataset',None),hours=a.hours,seed=getattr(a,'seed',2371),max_trials=a.max_trials,resume=a.cmd=='peak-resume',adapter=sys.modules[__name__])


if __name__=='__main__':main()
