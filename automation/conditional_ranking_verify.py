"""Rebuild a frozen challenger and verify its bytes, without reading holdout metrics."""
import argparse,json
from pathlib import Path
from automation.conditional_ranking import load,transformed,metrics
from src.engine.ai.training_v223.model import train_rank_model,RankModelV223
from src.engine.ai.canonical_challenger import digest


def verify(run,output):
    manifest=json.loads((run/'manifest.json').read_text())
    source=Path(manifest['source'])
    if digest(source)!=manifest['sha256']:raise ValueError('Dataset changed')
    records=load(source)
    purged={tuple(row) for row in manifest['purged_exact_candidate_duplicates']}
    records=[r for r in records if (r.session_id,r.shot_id) not in purged]
    frozen=json.loads((run/'challenger.json').read_text())
    best=json.loads((run/frozen['model_directory']/'result.json').read_text())
    old=RankModelV223.load(run/frozen['model_directory'])
    train=[r for r in records if r.session_id in manifest['splits']['TRAIN'] and r.oracle20]
    model,info=train_rank_model(transformed(train,list(old.feature_names),best['transform']),
        kind='linear',feature_names=list(old.feature_names),epochs=80,learning_rate=.02,
        l2=best['l2'],seed=2230,max_candidates_per_shot=512,metadata=old.metadata)
    output.mkdir(parents=True,exist_ok=False);model.save(output)
    checks={f:digest(output/f)==best['hashes'][f] for f in ['model.json','model.npz']}
    dependencies=['automation/conditional_ranking.py','src/engine/ai/canonical_challenger.py',
                  'src/engine/ai/training_v223/model.py','src/engine/ai/training_v223/schema.py']
    result={'hash_equal':checks,'source_hashes':{p:digest(p) for p in dependencies}}
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    profile={}
    for split,sids in manifest['splits'].items():
        rs=[r for r in records if r.session_id in sids]
        profile[split]=metrics(rs,[sorted(range(len(r.candidates)),key=lambda i:-r.candidates[i].baseline_score) for r in rs])
    (output/'benchmark_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    if not all(checks.values()):raise ValueError('Rebuilt model differs from frozen bytes')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.run,a.output)
