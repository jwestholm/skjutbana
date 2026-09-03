from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np

from src.engine.ai.training_v223.dense_v2233 import DenseShotV2233, REDUCER_FEATURE_NAMES
from src.engine.ai.training_v223.reducer_v2233 import ReducerModelV2233, evaluate_reducer, train_reducer
from src.engine.ai.training_v223.rich_v2233 import RICH_FEATURE_NAMES, compute_rich_feature_matrix
from src.engine.ai.training_v223.schema import FEATURE_NAMES, FORBIDDEN_FEATURE_NAMES
from src.engine.ai.training_v223.trainer_v2233 import _shot_partition


def ok(msg: str) -> None: print(f'[PASS] {msg}')

def fail(msg: str) -> None: raise AssertionError(msg)


def synthetic_shots(seed: int = 1, shots: int = 24, n: int = 1400) -> list[DenseShotV2233]:
    rng=np.random.default_rng(seed); d=len(REDUCER_FEATURE_NAMES); out=[]
    ir=REDUCER_FEATURE_NAMES.index('v2233_newhole_heuristic')
    ia=REDUCER_FEATURE_NAMES.index('v2233_abs_r2')
    ip=REDUCER_FEATURE_NAMES.index('v2233_persist_r2')
    idense=REDUCER_FEATURE_NAMES.index('dense_score')
    for s in range(shots):
        x=rng.normal(0,1,(n,d)).astype(np.float32)
        distances=rng.uniform(60,900,n).astype(np.float32)
        # Several actual positive hypotheses. Baseline dense score deliberately
        # prefers negatives so learning must use the physical NEW-hole features.
        pos=rng.choice(n,size=4,replace=False)
        distances[pos]=rng.uniform(1,12,4)
        x[:,idense]=rng.normal(1.5,0.7,n)
        x[pos,idense]=rng.normal(-0.8,0.1,4)
        x[pos,ir]+=6.0; x[pos,ia]+=5.0; x[pos,ip]+=4.0
        xy=rng.uniform(0,3000,(n,2)).astype(np.float32)
        baseline=x[:,idense].copy()
        out.append(DenseShotV2233(f's{s//8}',str(s+1),s+1,xy,x,distances,baseline,(1000.0,1000.0)))
    return out


def main() -> None:
    print('V2.23.3 SELFTEST')
    print('================')
    assert all(name not in FORBIDDEN_FEATURE_NAMES for name in REDUCER_FEATURE_NAMES)
    assert not any(name.startswith('gt_') for name in REDUCER_FEATURE_NAMES)
    ok('reducer feature contract contains no GT/rank/model leakage')

    pre=np.full((180,240),180,np.uint8); posts=[]
    for k in range(3):
        p=pre.copy(); yy,xx=np.ogrid[:180,:240]; mask=(xx-120)**2+(yy-90)**2 <= (3+k//2)**2; p[mask]=35; posts.append(p)
    candidates=[{'camera_x':120,'camera_y':90},{'camera_x':30,'camera_y':30}]
    m=compute_rich_feature_matrix(pre,posts,candidates)
    assert m.shape==(2,len(RICH_FEATURE_NAMES))
    hidx=RICH_FEATURE_NAMES.index('v2233_newhole_heuristic')
    assert float(m[0,hidx]) > float(m[1,hidx]) + 0.2
    ok('GT-free rich PRE/POST maps react strongly at a synthetic persistent new hole')

    shots=synthetic_shots()
    before=[]
    didx=REDUCER_FEATURE_NAMES.index('dense_score')
    for shot in shots[16:]:
        order=np.argsort(-shot.features[:,didx]); rank=next(i for i,j in enumerate(order,1) if shot.distances[int(j)]<=20); before.append(rank)
    model,info=train_reducer(shots[:16],kind='mlp',hidden=24,epochs=18,learning_rate=0.008,seed=77)
    metrics=evaluate_reducer(model,shots[16:])
    assert metrics['retention20_at_k']['32'] >= 0.75, metrics
    assert metrics['median_positive_rank20'] is not None and metrics['median_positive_rank20'] <= 8, metrics
    assert float(np.median(before)) > 100
    ok('pairwise reducer moves positives from bad baseline ranks into a small top-K pool')


    # The second stage must be able to train on reducer-retained rows without
    # reintroducing GT into model features.
    from src.engine.ai.training_v223.trainer_v2233 import _to_reduced_records
    from src.engine.ai.training_v223.model import train_rank_model, evaluate_model
    train_records=_to_reduced_records(shots[:16],model,top_k=64)
    val_records=_to_reduced_records(shots[16:],model,top_k=64)
    final,_=train_rank_model(train_records,kind='mlp',hidden=12,epochs=8,learning_rate=0.01,seed=78,max_candidates_per_shot=64,feature_names=REDUCER_FEATURE_NAMES)
    final_metrics=evaluate_model(final,val_records)
    assert final_metrics['conditional_top1_20_rate'] >= 0.5, final_metrics
    ok('reducer -> final listwise ranker cascade trains on retained candidates')

    with tempfile.TemporaryDirectory() as td:
        model.save(td); loaded=ReducerModelV2233.load(td)
        a=model.score_matrix(shots[20].features[:20]); b=loaded.score_matrix(shots[20].features[:20])
        assert np.allclose(a,b,atol=1e-5)
    ok('reducer model NPZ+JSON round-trip uses allow_pickle=False-compatible arrays')

    class R:
        def __init__(self,i): self.session_id='same'; self.shot_id=str(i)
    tr,val=_shot_partition([R(i) for i in range(1,101)])
    assert len(tr)>=70 and len(val)>=10 and len(tr)+len(val)==100
    ok('single-session bootstrap split is deterministic and leaves validation data')

    assert 'v2233_newhole_heuristic' in FEATURE_NAMES
    ok('stable V2.23 schema exports V2.23.3 rich evidence fields')
    print('\nAll V2.23.3 selftests passed.')

if __name__=='__main__': main()
