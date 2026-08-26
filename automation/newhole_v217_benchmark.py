from __future__ import annotations
import argparse,json
from pathlib import Path
from src.engine.ai.new_hole_ai_v217 import NewHoleAIV217
from src.engine.offline.new_hole_training_v217 import DEFAULT_HARDNEG_MANIFEST, DEFAULT_ROOT, _evaluate_samples, _ranking_eval, _shot_split_keys_v217, discover_samples_v217, split_samples_v217

def main() -> int:
    p=argparse.ArgumentParser(description='Re-benchmark a frozen V2.17 NEW-hole model without retraining')
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT); p.add_argument('--hard-negatives',type=Path,default=DEFAULT_HARDNEG_MANIFEST)
    p.add_argument('--model',type=Path,default=Path('content/ai/reports/v217/new_hole_ai_v217.npz')); p.add_argument('--out',type=Path,default=Path('content/ai/reports/v217/new_hole_v217_rebenchmark.json'))
    args=p.parse_args(); model,meta=NewHoleAIV217.load(args.model); threshold=float(meta.get('metadata',{}).get('threshold',0.5))
    samples,dataset=discover_samples_v217(args.root,hardnegative_manifest=args.hard_negatives); split,provisional=split_samples_v217(args.root,samples); split_keys,_=_shot_split_keys_v217(args.root)
    evaluations={name:_evaluate_samples(model,rows,threshold=threshold) for name,rows in split.items()}; ranking={name:_ranking_eval(args.root,model,split_rounds=keys,radius=20.0) for name,keys in split_keys.items()}
    report={'schema_version':'2.17','model':str(args.model),'shadow_only':True,'threshold':threshold,'dataset':dataset,'split_is_provisional':provisional,'evaluations':evaluations,'candidate_ranking_20px':ranking}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('V2.17 FROZEN MODEL BENCHMARK'); print('============================'); print(f'Model       : {args.model}'); print(f'Provisional : {provisional}')
    for name,row in ranking.items(): print(f"  {name:12s} top1={row.get('top1')} top3={row.get('top3')} oracle={row.get('oracle_recall')} median_rank={row.get('median_gt_rank')}")
    print(f'Report      : {args.out}'); return 0
if __name__=='__main__': raise SystemExit(main())
