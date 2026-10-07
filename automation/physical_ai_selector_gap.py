"""Bounded selector audit; fixed cached evidence, never proposal generation.

GT is confined to supervised labels and diagnostic evaluation. Direction/scaling
are fitted on outer TRAIN only. Producer scores/history are audit-only.
"""
import argparse
from pathlib import Path
import numpy as np
from automation import physical_ai_dataset as ds, physical_ai_fusion as fusion
from automation import physical_ai_fusion_crossfit as crossfit
from src.engine.ai.canonical_challenger import transform_matrix

REFERENCE=fusion.lab.ROOT/'fusion_week_01/trials/trial_060456/result.json'
PHYSICAL=tuple(n for family,names in ds.FAMILIES.items() if family!='producer_history' for n in names)
NAMES=PHYSICAL+tuple(n+'__missing' for n in PHYSICAL)
COLS=[ds.FEATURE_NAMES.index(n) for n in NAMES]
FAMILY={n:f for f,ns in ds.FAMILIES.items() for n in ns}


def physical_matrix(x):
    if x.ndim!=2 or x.shape[1]!=len(ds.FEATURE_NAMES):raise ValueError('Unknown schema')
    return transform_matrix(x[:,COLS],'signed_log')


def positive_mask(event):
    """Supervised target construction only; never an inference feature."""
    ds.guard(event['event']);ds.guard(event['session'])
    if event['truth']=='UNKNOWN':return None
    if event['truth']=='NO_IMPACT':return np.zeros(len(event['indices']),bool)
    if event['truth']!='HIT':raise ValueError('Unknown truth')
    return np.array([ds.distance(c['xy'],event['gt_xy'])<=5 for c in event['candidates']])


def train_events(data,fold):
    found=next(r for r in data['nested_training'] if r['fold']==fold)['events']
    if any(e['session'] not in fold['train'] for e in found):raise ValueError('Held row in TRAIN')
    safe=fusion.training_filter(found,fold)
    if any(a['indices']!=b['indices'] for a,b in zip(found,safe)):raise ValueError('Unsafe upstream rows')
    return [e for e in found if e['truth']!='UNKNOWN']


def train_statistics(events,x,fold):
    if any(e['session'] not in fold['train'] for e in events):raise ValueError('Held scaler/direction leakage')
    rows=np.concatenate([x[e['indices']] for e in events]);mean=rows.mean(0);scale=rows.std(0);scale[scale<1e-6]=1
    evidence={}
    for j,name in enumerate(PHYSICAL):
        auc=[];pairs=[]
        for e in events:
            pos=positive_mask(e);v=x[e['indices']];valid=v[:,j+len(PHYSICAL)]==0
            a=v[pos&valid,j];b=v[(~pos)&valid,j]
            if len(a) and len(b):
                auc.append(float(np.mean((a[:,None]>b)+.5*(a[:,None]==b))))
                pairs.append(e['event'])
        avg=float(np.mean(auc)) if auc else .5;direction=1 if avg>.5 else -1 if avg<.5 else 0
        evidence[name]=dict(direction=direction,train_auc=avg,oriented_auc=max(avg,1-avg),events=pairs,
             consistency=float(np.mean((np.array(auc)-.5)*direction>0)) if auc else 0,
             eligible=len(auc)>=4 and max(avg,1-avg)>=.65)
    correlations=[]
    for a in range(len(PHYSICAL)):
        for b in range(a):
            valid=(rows[:,a+len(PHYSICAL)]==0)&(rows[:,b+len(PHYSICAL)]==0)
            z=rows[valid]
            if len(z)>=10 and z[:,a].std()>1e-6 and z[:,b].std()>1e-6:
                r=float(np.corrcoef(z[:,a],z[:,b])[0,1])
                if abs(r)>=.9:correlations.append(dict(a=PHYSICAL[a],b=PHYSICAL[b],r=r))
    return mean,scale,evidence,correlations


def rank(values,direction):
    # Equal measurements share best rank; no arbitrary candidate-order advantage.
    return np.array([1+np.sum(direction*values>direction*v) for v in values],int)


def audit(output):
    output=ds.output_root(output);output.mkdir(parents=True,exist_ok=True)
    data,raw=fusion.load(crossfit.DEFAULT);x=physical_matrix(raw);ref=ds.read(REFERENCE)
    identity=dict(dataset=data['dataset_hash'],reference=ds.digest(REFERENCE),code=ds.digest(__file__),
        policy='TRAIN-event mean AUC>=.65 or <=.35 with >=4 observed positive/negative events; held rank<=3 and directed standardized delta>=.5: descriptive support only')
    destination=output/'audit.json'
    if destination.exists():
        old=ds.read(destination)
        if old['identity']!=identity:raise ValueError('Audit identity changed; use new output root')
        return old
    results={r['event']:r for r in ref['metrics']['per_event']};details={r['event']:r for f in ref['folds'] for r in f['detail']}
    classic=ds.read(fusion.CLASSIC/'provenance.json');vision=ds.read(fusion.DEFAULT/'provenance.json')
    provenance={(r['event'],str(r['candidate_id'])):r for r in classic+vision}
    reports=[];fold_reports=[]
    for fold in data['folds']:
        train=train_events(data,fold);mean,scale,directions,corr=train_statistics(train,x,fold)
        fold_reports.append(dict(fold=fold,training_events=[e['event'] for e in train],mean=mean.tolist(),scale=scale.tolist(),directions=directions,correlations=corr))
        for e in data['events']:
            r=results[e['event']]
            if e['session'] not in fold['held'] or e['truth']!='HIT' or r['oracle_error']>5 or(r['error'] is not None and r['error']<=5):continue
            d=details[e['event']];scores=np.array(d['scores']);v=x[e['indices']];rawv=raw[e['indices']];pos=positive_mask(e)
            if d['candidate_ids']!=[c['id'] for c in e['candidates']]:raise ValueError('Champion/candidate association changed')
            good=int(np.flatnonzero(pos)[np.argmax(scores[pos])]);bad=int(np.flatnonzero(~pos)[np.argmax(scores[~pos])]);win=int(np.argmax(scores))
            z=(v-mean)/scale;separators=[];dimensions=[]
            for j,n in enumerate(PHYSICAL):
                direction=directions[n]['direction'];rr=rank(v[:,j],direction) if direction else np.ones(len(v),int)
                delta=float(z[good,j]-z[bad,j]);observed=v[good,j+len(PHYSICAL)]==v[bad,j+len(PHYSICAL)]==0
                useful=observed and directions[n]['eligible'] and rr[good]<=3 and delta*direction>=.5
                if useful:separators.append(n)
                dimensions.append(dict(feature=n,family=FAMILY[n],train_direction=directions[n],good_rank=int(rr[good]),
                    good=float(rawv[good,ds.FEATURE_NAMES.index(n)]),wrong=float(rawv[bad,ds.FEATURE_NAMES.index(n)]),
                    raw_delta=float(rawv[good,ds.FEATURE_NAMES.index(n)]-rawv[bad,ds.FEATURE_NAMES.index(n)]),
                    normalized_delta=delta,both_observed=bool(observed),descriptive_separation=bool(useful)))
            collisions={str(j):[int(k) for k in np.flatnonzero(~pos) if np.array_equal(v[j],v[k])] for j in np.flatnonzero(pos)}
            all_collide=all(collisions[str(j)] for j in np.flatnonzero(pos))
            source_missing=[n for j,n in enumerate(PHYSICAL) if v[good,j+len(PHYSICAL)]!=0]
            classification=('ABSTENTION' if pos[win] and r['state']=='UNKNOWN' else
                'EVIDENCE_INDISTINGUISHABLE' if all_collide else 'RANKING_POTENTIAL' if separators else
                'MISSING_EVIDENCE' if source_missing else 'SEPARATION_UNPROVEN')
            candidates=[]
            for j,c in enumerate(e['candidates']):
                candidates.append(dict(id=c['id'],xy=c['xy'],error=ds.distance(c['xy'],e['gt_xy']),source='VISION' if 'vision' in c else 'CLASSIC',
                    provenance=provenance.get((e['event'],str(c['id'])),c.get('vision')),score=float(scores[j]),rank=int(rank(scores,1)[j]),
                    values=dict(zip(ds.FEATURE_NAMES,map(float,rawv[j]))),physical_feature_ranks={n:int(rank(v[:,i],directions[n]['direction'])[j]) if directions[n]['direction'] else None for i,n in enumerate(PHYSICAL)}))
            reports.append(dict(event=e['event'],gt_xy=e['gt_xy'],champion=r,good_index=good,wrong_index=bad,winning_index=win,
                good_indices=np.flatnonzero(pos).tolist(),nearest_index=int(np.argmin([c['error'] for c in candidates])),
                candidates=candidates,dimensions=dimensions,separators=separators,separating_families=sorted({FAMILY[n] for n in separators}),
                normalized_distance=float(np.linalg.norm(z[good]-z[bad])),exact_collisions=collisions,
                missing_good=source_missing,classification=classification))
    out=dict(identity=identity,features=list(NAMES),excluded_audit_only=list(ds.FAMILIES['producer_history']),gaps=reports,folds=fold_reports,
             proposal_failures=[r['event'] for r in results.values() if r['truth']=='HIT' and r['oracle_error']>5])
    ds.atomic_json(destination,out);return out


def main():
    ds.install_guard();p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.output)
    for e in r['gaps']:
        g=e['candidates'][e['good_index']];b=e['candidates'][e['wrong_index']]
        print(e['event'],e['classification'],g['source'],'rank',g['rank'],'score',round(g['score'],3),'wrong',round(b['score'],3),'sep',e['separators'],'distance',round(e['normalized_distance'],3),'collision',e['exact_collisions'])

if __name__=='__main__':main()
