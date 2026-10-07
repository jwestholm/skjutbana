"""Deeper TRAIN-only proposal cache for the existing Fusion adapter.

Outer TRAIN={A,B,C}; predict C with base Vision(A,B) and reranker(A,B).
Reranker training peaks for A come from Vision(B), and vice versa. Outer
HELD/dev and C occur in no upstream fit. Frozen HELD proposals never change.
"""
import copy
from pathlib import Path
import numpy as np
import cv2
from automation import physical_ai_dataset as ds,physical_ai_fusion as fusion
from automation import physical_ai_vision as vision,physical_ai_vision_peaks as peaks
from src.engine.ai.training_v223.model import train_fusion_model,RankModelV223
from src.engine.ai.training_v223.heatmap_model_v2236 import HeatmapModelV2236,init_heatmap_model

SCHEMA='fusion-nested-train-1'
DEFAULT=fusion.lab.ROOT/'datasets/fusion_nested_top3_v1_20261007'
REFERENCE=fusion.lab.ROOT/'fusion_week_01/trials/trial_060456/result.json'


def nesting(fold,predicted):
    for s in fold['train']+fold['held']+fold['dev']:ds.guard(s)
    if len(fold['train'])!=3 or predicted not in fold['train']:raise ValueError('Three-session outer TRAIN required')
    if set(fold['train']) & set(fold['held']+fold['dev']):raise ValueError('Outer split overlaps')
    allowed=[s for s in fold['train'] if s!=predicted]
    return dict(train=allowed,held=[predicted],dev=fold['held']+fold['dev'])


def training_only(data,sessions):
    for s in sessions:ds.guard(s)
    return dict(data,events=[e for e in data['events'] if e['session'] in sessions])


def model_record(path,kind,sessions,events,upstream=()):
    return dict(directory=str(path),kind=kind,training_sessions=list(sessions),training_events=list(events),
                upstream=list(upstream),hashes={n:ds.digest(path/n) for n in ('model.json','model.npz')})


def assert_chain(p,event,outer,models):
    if not fusion.safe_training_proposal(p,event,outer):raise ValueError('Nested TRAIN proposal leakage')
    allowed=set(outer['train'])-{event['session']}
    base=models[p['model_sha256']];ranker=models[p['reranker_model_sha256']]
    if set(base['training_sessions'])!=allowed or set(ranker['training_sessions'])!=allowed:raise ValueError('Base/reranker model split differs')
    if base['training_sessions']!=p['vision_training_sessions'] or ranker['training_sessions']!=p['reranker_training_sessions']:raise ValueError('Proposal/model lineage mismatch')
    if set(p['reranker_upstream_models'])!=set(ranker['upstream']):raise ValueError('Missing transitive lineage')
    for h in ranker['upstream']:
        if not set(models[h]['training_sessions'])<allowed:raise ValueError('Reranker upstream leakage')


def prepare(output,*,resume=False,stop=lambda:False):
    output=ds.output_root(output);base,x=fusion.load(fusion.DEFAULT);data,shards=peaks.load(fusion.PEAKS)
    frozen=ds.read(fusion.VISION);ref=ds.read(REFERENCE)
    cfg=data['peak_manifest']['intent']['config'];rcfg=frozen['config'];cache=Path(data['peak_manifest']['intent']['vision_cache'])
    data=dict(data,dataset_hash=data['peak_manifest']['intent']['vision_cache_hash'])
    intent=dict(schema=SCHEMA,base=str(fusion.DEFAULT),base_hash=base['dataset_hash'],peak_cache_hash=data['dataset_hash'],
        vision_config=cfg,reranker_config=rcfg,code=fusion.code_identity(),reference=str(REFERENCE),reference_sha256=ds.digest(REFERENCE))
    if resume:
        if ds.read(output/'intent.json')!=intent:raise ValueError('Cross-fit preparation identity changed')
    else:output.mkdir(parents=True,exist_ok=False);ds.atomic_json(output/'intent.json',intent)
    cv2.setNumThreads(1);models={};singletons={};pair_predictions={}
    # Existing two-session base models for each outer TRAIN prediction are reused.
    for shard in shards:
        for rec in shard['inner_models']:
            path=Path(rec['directory']);meta=ds.read(path/'model.json')['metadata']
            m=model_record(path,'VISION_PAIR',meta['fold']['train'],meta['normalization']['training_events'])
            models[m['hashes']['model.npz']]=m
    for session in base['sessions']:
        if stop():raise InterruptedError('Cross-fit preparation paused')
        path=output/'singletons'/session;path.mkdir(parents=True,exist_ok=True);done=path/'complete.json'
        fold=dict(train=[session],held=[s for s in base['sessions'] if s!=session],dev=[])
        identity=dict(intent=ds.canonical_hash(intent),fold=fold)
        if done.exists():
            saved=ds.read(done)
            if saved['identity']!=identity:raise ValueError('Singleton identity changed')
            for n,h in saved['model']['hashes'].items():
                if ds.digest(path/n)!=h:raise ValueError('Singleton model changed')
            model=HeatmapModelV2236.load(path)
        else:
            subset=training_only(data,[session]);samples,norm=vision.training_samples(subset,cache,fold,cfg)
            model=init_heatmap_model(cfg['kind'],hidden=cfg['hidden'],seed=cfg['seed'])
            history=vision.fit_proposal_model(model,subset,cache,fold,cfg,norm,samples,path,seed=cfg['seed'],identity=identity,stop=stop)
            model.metadata.update(fold=fold,normalization=norm,code=fusion.code_identity(),role='VISION_FUSION_INNER',live_authority=False)
            model.save(path);saved=dict(identity=identity,history=history,model=model_record(path,'VISION_SINGLETON',[session],norm['training_events']))
            ds.atomic_json(done,saved)
        m=saved['model'];models[m['hashes']['model.npz']]=m;singletons[session]=(model,path)
        print('singleton complete',session,flush=True)
    # Hash-bound deterministic model predictions, reused across outer folds.
    def singleton_peaks(train_session,predicted_session):
        key=(train_session,predicted_session)
        if train_session==predicted_session:raise ValueError('In-sample singleton prediction')
        if key not in pair_predictions:
            path=output/'peaks'/f'{train_session}_to_{predicted_session}.json';model,directory=singletons[train_session]
            identity=dict(model=ds.digest(directory/'model.npz'),cache=data['dataset_hash'],predicted_session=predicted_session)
            if path.exists():
                saved=ds.read(path)
                if saved['identity']!=identity:raise ValueError('Prediction identity changed')
            else:
                events=[e for e in data['events'] if e['session']==predicted_session]
                # Inference receives no GT or physical truth fields.
                blind=[{k:v for k,v in e.items() if k not in ('gt_xy','truth','current_error','protected')} for e in events]
                pp=peaks._extract_events(data,cache,blind,model,directory,cfg,str(directory),[train_session])
                saved=dict(identity=identity,outputs=pp);ds.atomic_json(path,saved)
            pair_predictions[key]=saved['outputs']
        return pair_predictions[key]
    folds=[];extra=[];events_by_id={e['event']:e for e in base['events']}
    for i,outer in enumerate(base['folds']):
        train_events=[]
        for predicted in outer['train']:
            if stop():raise InterruptedError('Cross-fit preparation paused')
            inner=nesting(outer,predicted);train_peaks={};upstream=[]
            for target in inner['train']:
                source=next(s for s in inner['train'] if s!=target)
                train_peaks.update(singleton_peaks(source,target));upstream.append(ds.digest(singletons[source][1]/'model.npz'))
            shard=dict(fold=inner,train=train_peaks)
            names,samples,ids=peaks.samples(training_only(data,inner['train']),shard,rcfg)
            path=output/f'fold_{i:02d}'/predicted;path.mkdir(parents=True,exist_ok=True);done=path/'complete.json'
            identity=dict(intent=ds.canonical_hash(intent),outer=outer,inner=inner,upstream=upstream,train_peaks_hash=ds.canonical_hash(train_peaks))
            if done.exists():
                saved=ds.read(done)
                if saved['identity']!=identity:raise ValueError('Inner reranker identity changed')
                for n,h in saved['model']['hashes'].items():
                    if ds.digest(path/n)!=h:raise ValueError('Inner reranker changed')
                model=RankModelV223.load(path)
            else:
                model,history=train_fusion_model(samples,feature_names=names,seed=rcfg['seed']+i,checkpoint=path/'checkpoint.npz',checkpoint_identity=identity,stop=stop,
                    **{k:rcfg[k] for k in ('hidden','epochs','learning_rate','l2','hard_negative_count')})
                model.metadata.update(outer_fold=outer,fold=inner,training_events=ids,upstream=upstream,code=fusion.code_identity(),role='VISION_FUSION_RERANK',live_authority=False)
                model.save(path);saved=dict(identity=identity,history=history,model=model_record(path,'RERANKER',inner['train'],ids,upstream))
                ds.atomic_json(done,saved)
            mr=saved['model'];models[mr['hashes']['model.npz']]=mr
            for eid,modes in shards[i]['train'].items():
                e=events_by_id[eid]
                if e['session']!=predicted:continue
                pool=modes[rcfg['order']];_,px=peaks.feature_matrix(pool,rcfg['subset'])
                proposals,_=peaks.select(pool,model.score_matrix(px),order=rcfg['order'],nms_radius_px=cfg['nms_radius_px'])
                for p in proposals[:3]:
                    p.update(vision_training_sessions=p['training_sessions'],reranker_training_sessions=inner['train'],reranker_model_sha256=mr['hashes']['model.npz'],
                        reranker_model_id=str(path),reranker_upstream_models=upstream,evidence_outer_fold=i)
                    assert_chain(p,e,outer,models)
                classic=dict(e,candidates=e['candidates'][:e['classic_count']])
                item=fusion.union_event(classic,proposals[:3]);added=item['candidates'][e['classic_count']:]
                ep=path/(eid.replace(':','_')+'_evidence.json')
                ev_identity=dict(points=[c['xy'] for c in added],base=base['dataset_hash'],model=mr['hashes']['model.npz'])
                if ep.exists():
                    ev=ds.read(ep)
                    if ev['identity']!=ev_identity:raise ValueError('Evidence identity changed')
                else:
                    vectors,witnesses=fusion.existing_evidence(e,added,base['source_hashes']) if added else ([],[])
                    ev=dict(identity=ev_identity,vectors=vectors,witnesses=witnesses);ds.atomic_json(ep,ev)
                item['indices']=e['indices'][:e['classic_count']]+list(range(len(x)+len(extra),len(x)+len(extra)+len(added)))
                extra.extend(ev['vectors']);train_events.append(item)
            print('cross-fit TRAIN ready',i,predicted,flush=True)
        folds.append(dict(fold=outer,events=train_events))
    np.save(output/'training_features.npy',np.asarray(extra,np.float32),allow_pickle=False)
    manifest=dict(intent=intent,folds=folds,models=models,feature_sha256=ds.digest(output/'training_features.npy'),
        reference_metrics=ref['metrics'],reference_champion='frozen_80k_trial_060456',held_union_unchanged=True)
    manifest['dataset_hash']=ds.canonical_hash(manifest);ds.atomic_json(output/'nested.json',manifest)
    return manifest


def load(output):
    output=Path(output);m=ds.read(output/'nested.json');expected=m.pop('dataset_hash')
    if ds.canonical_hash(m)!=expected or m['intent']['schema']!=SCHEMA:raise ValueError('Nested manifest changed')
    base,x=fusion.load(m['intent']['base'])
    if base['dataset_hash']!=m['intent']['base_hash']:raise ValueError('Frozen HELD union changed')
    if ds.digest(output/'training_features.npy')!=m['feature_sha256']:raise ValueError('Nested features changed')
    extra=np.load(output/'training_features.npy',allow_pickle=False)
    if extra.ndim!=2 or extra.shape[1]!=x.shape[1] or not np.isfinite(extra).all():raise ValueError('Invalid training features')
    events={e['event']:e for e in base['events']}
    for h,r in m['models'].items():
        ds.guard(r['directory'])
        for s in r['training_sessions']:ds.guard(s)
        for name,digest in r['hashes'].items():
            if name not in ('model.json','model.npz') or ds.digest(Path(r['directory'])/name)!=digest:raise ValueError('Upstream model drift')
        meta=ds.read(Path(r['directory'])/'model.json')['metadata']
        if meta['fold']['train']!=r['training_sessions']:raise ValueError('Actual model sessions differ')
        recorded=meta.get('normalization',{}).get('training_events',meta.get('training_events'))
        if recorded!=r['training_events']:raise ValueError('Actual model events differ')
        if any(events[e]['session'] not in r['training_sessions'] for e in recorded):raise ValueError('Actual training event leakage')
        if meta.get('normalization') and meta['normalization']['training_sessions']!=r['training_sessions']:raise ValueError('Scaler leakage')
        if r['kind']=='RERANKER' and meta['upstream']!=r['upstream']:raise ValueError('Actual transitive lineage differs')
        if h!=r['hashes']['model.npz']:raise ValueError('Model identity differs')
    indices=[]
    if len(m['folds'])!=len(base['folds']):raise ValueError('Incomplete nested folds')
    for outer,record in zip(base['folds'],m['folds']):
        if record['fold']!=outer:raise ValueError('Evidence fold changed')
        expected_events={e['event'] for e in base['events'] if e['session'] in outer['train']}
        if {e['event'] for e in record['events']}!=expected_events or len(record['events'])!=len(expected_events):raise ValueError('Incomplete TRAIN events')
        for e in record['events']:
            ds.guard(e['event']);original=events[e['event']];n=original['classic_count']
            if e['truth']!=original['truth'] or e['gt_xy']!=original['gt_xy'] or e['session']!=original['session']:raise ValueError('Native truth changed')
            if e['candidates'][:n]!=original['candidates'][:n] or e['indices'][:n]!=original['indices'][:n]:raise ValueError('Classic bank changed')
            if len(e['vision_top3'])!=3 or len(e['candidates'])>n+3:raise ValueError('Vision cap changed')
            for p in e['vision_top3']:assert_chain(p,e,outer,m['models'])
            for c in e['candidates'][n:]:
                if c['xy']!=c['vision']['camera_xy'] or c['vision'] not in e['vision_top3']:raise ValueError('Unobserved coordinate')
            indices.extend(e['indices'][n:])
    if indices!=list(range(len(x),len(x)+len(extra))):raise ValueError('Training feature association differs')
    return (dict(base,dataset_hash=expected,nested_training=m['folds'],fusion=dict(base['fusion'],revision=SCHEMA,
        reference_metrics=m['reference_metrics'],reference_champion=m['reference_champion'],training_policy='Classic + genuinely nested Vision TRAIN; frozen HELD union')),
        np.concatenate([x,extra]))
