"""Frozen chronological conditional ranking experiments on historical V2.9 pools."""
from __future__ import annotations
import argparse,copy,json,platform
from pathlib import Path
import numpy as np
from src.engine.ai.canonical_challenger import digest,transform_matrix
from src.engine.ai.training_v223.schema import ShotTrainingRecord,CandidateTrainingRow,extract_physical_features
from src.engine.ai.training_v223.model import train_rank_model

FEATURES=['detector_score','area','radius','circularity','center_change','local_contrast','pre_shot_change','change_value','darkening','dog_value','zscore','persistence','existed_before','temporal_hits','same_frame_support','support_score','v2_rescue_temporal']


def load(path):
    records=[]
    for line in path.read_text().splitlines():
        r=json.loads(line);g=r['ground_truth'];cs=[]
        for c in r['candidates']:
            raw=c.get('raw',{});f=extract_physical_features(raw)
            if not all(np.isfinite([c['camera_x'],c['camera_y'],*f.values()])): raise ValueError('Nonfinite input')
            cs.append(CandidateTrainingRow(candidate_id=c['id'],camera_x=c['camera_x'],camera_y=c['camera_y'],features=f,
                       baseline_score=float(c['features']['baseline_score']),baseline_rank=c['ranks']['baseline']))
        record=ShotTrainingRecord(r['session_id'],str(r['sequence']),'projected_f2_historical',float(r['captured_at']),g['camera_x'],g['camera_y'],cs)
        record.finalize_labels();records.append(record)
    return records


def metrics(records,orders):
    if len(records)!=len(orders): raise ValueError('Ranking count mismatch')
    result={'shots':len(records),'candidate_count':sum(len(r.candidates) for r in records),'tolerances':{}}
    for radius in (5,10,20,42):
        ranks=[]
        for r,order in zip(records,orders):
            if sorted(order)!=list(range(len(r.candidates))): raise ValueError('Candidate membership changed')
            rank=next((j+1 for j,i in enumerate(order) if r.candidates[i].gt_distance_px<=radius),None)
            if rank is not None:ranks.append(rank)
        n=len(ranks)
        result['tolerances'][str(radius)]={'oracle':n,'oracle_fraction':n/len(records) if records else None,
          'excluded_no_positive':len(records)-n,**{f'conditional_top{k}':sum(r<=k for r in ranks)/n if n else None for k in (1,3,10)},
          'conditional_mrr':sum(1/r for r in ranks)/n if n else None,'mean_positive_rank':float(np.mean(ranks)) if n else None}
    result['sessions']={sid:metrics([r for r in records if r.session_id==sid],[o for r,o in zip(records,orders) if r.session_id==sid]) for sid in sorted({r.session_id for r in records})} if len({r.session_id for r in records})>1 else {}
    return result


def transformed(records,names,method):
    out=copy.deepcopy(records)
    for r in out:
        x=transform_matrix([[c.features[k] for k in names] for c in r.candidates],method)
        for c,row in zip(r.candidates,x):c.features.update(dict(zip(names,map(float,row))))
    return out


def objective(m):
    b=m['tolerances']['20'];return tuple(b[k] or 0 for k in ['conditional_top1','conditional_mrr','conditional_top3'])


def run(source,out):
    out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    records=load(source);sessions=sorted({r.session_id for r in records})
    if len(sessions)<3:raise ValueError('Need at least three complete sessions')
    split={'TRAIN':sessions[:-3],'DEVELOPMENT':sessions[-3:-1],'PROTECTED_HOLDOUT':sessions[-1:]}
    # Candidate coordinate fingerprints are label-blind; purge earlier exact duplicates.
    seen=set();purged=[];groups={}
    for key in ('PROTECTED_HOLDOUT','DEVELOPMENT','TRAIN'):
        groups[key]=[]
        for r in records:
            if r.session_id not in split[key]:continue
            fingerprint=tuple(sorted((round(c.camera_x,4),round(c.camera_y,4)) for c in r.candidates))
            if fingerprint in seen:purged.append([r.session_id,r.shot_id]);continue
            seen.add(fingerprint);groups[key].append(r)
    save('manifest.json',{'source':str(source),'sha256':digest(source),'splits':split,'purged_exact_candidate_duplicates':purged,
        'python':platform.python_version(),'numpy':np.__version__,'primary':'conditional Top1@20, MRR20 tie-break',
        'holdout_status':'Frozen for this run; historical dataset and newest session previously evaluated elsewhere, not a virgin holdout',
        'limitations':['Whole sessions, chronological. Repeated backgrounds/target sequences and unavailable images prevent near-duplicate certification.',
                       'Historical projected candidate snapshots only; no live-path-equivalent replay or physical validation.']})
    train=[r for r in groups['TRAIN'] if r.oracle20];dev=groups['DEVELOPMENT']
    baseline_orders=lambda rs:[sorted(range(len(r.candidates)),key=lambda i:-r.candidates[i].baseline_score) for r in rs]
    base=metrics(dev,baseline_orders(dev));save('baseline.json',base)
    configs=[('physical','identity',.001,FEATURES,'Conditional training removes impossible shots from scaler fitting.'),
             ('log','signed_log',.001,FEATURES,'Large evidence magnitudes may dominate standardized linear fitting; compress tails.'),
             ('percentile','within_shot_percentile',.001,FEATURES,'Shot-local relative evidence may transfer across camera exposure changes.'),
             ('geometry','identity',.001,['area','radius','circularity','detector_score'],'Geometry-only capacity tests whether noisy temporal features harm ranking.'),
             ('evidence','identity',.001,[f for f in FEATURES if f not in ['area','radius','circularity','detector_score']],'Evidence-only ablation tests reliance on geometry and detector score.'),
             ('regularized','identity',.1,FEATURES,'Stronger regularization tests overfitting with few oracle-positive training shots.')]
    results=[];best=None;previous=base
    for name,method,l2,names,hypothesis in configs:
        directory=out/name;directory.mkdir();(directory/'hypothesis.txt').write_text(hypothesis+'\n')
        tr=transformed(train,names,method)
        model,info=train_rank_model(tr,kind='linear',feature_names=names,epochs=80,learning_rate=.02,l2=l2,seed=2230,max_candidates_per_shot=512,
                                   metadata={'status':'OFFLINE_CHALLENGER','hypothesis':hypothesis})
        model.save(directory);dv=transformed(dev,names,method);m=metrics(dev,[model.rank_indices(r).tolist() for r in dv])
        guard=all((m['tolerances'][str(r)]['conditional_top10'] or 0)>=(base['tolerances'][str(r)]['conditional_top10'] or 0) for r in (5,10,20,42))
        result={'name':name,'hypothesis':hypothesis,'metrics':m,'training':info,'transform':method,'l2':l2,'guardrails':guard,
                'hashes':{f:digest(directory/f) for f in ['model.json','model.npz']},
                'objective':objective(m),'baseline_objective':objective(base),'previous_objective':objective(previous),
                'best_before_objective':objective(best['metrics']) if best else objective(base)}
        if guard and (best is None or objective(m)>objective(best['metrics'])):best=result
        result['decision']='best_eligible_ai_so_far' if best is result else 'discard';results.append(result);save(name+'/result.json',result);previous=m
        print(name,objective(m),'guard',guard,flush=True)
    # Freeze an AI candidate even if baseline wins; that distinction is explicit.
    if best is None:best=max(results,key=lambda r:objective(r['metrics']))
    frozen={'mode':'SHADOW','status':'OFFLINE_CHALLENGER','model_directory':best['name'],'hashes':best['hashes'],'transform':best['transform'],
            'beats_baseline_development':objective(best['metrics'])>objective(base),'guardrails':best['guardrails']}
    save('challenger.json',frozen)
    from src.engine.ai.training_v223.model import RankModelV223
    model=RankModelV223.load(out/best['name']);hold=groups['PROTECTED_HOLDOUT'];hh=transformed(hold,list(model.feature_names),best['transform'])
    save('holdout.json',{'baseline':metrics(hold,baseline_orders(hold)),'challenger':metrics(hold,[model.rank_indices(r).tolist() for r in hh])})
    save('summary.json',{'best':best['name'],'train_shots':len(groups['TRAIN']),'train_oracle20':len(train),'results':results})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('content/ai/ranking_v29/ranking_dataset.jsonl'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source,a.output)
