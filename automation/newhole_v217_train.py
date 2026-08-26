from __future__ import annotations
import argparse
from pathlib import Path
from src.engine.ai.new_hole_ai_v217 import NewHoleAIConfigV217
from src.engine.offline.new_hole_training_v217 import DEFAULT_HARDNEG_MANIFEST, DEFAULT_OUT, DEFAULT_ROOT, TrainingConfigV217, run_training_v217

def main() -> int:
    p=argparse.ArgumentParser(description='Train V2.17 before/after NEW-hole AI from V2.16 candidate packs')
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT); p.add_argument('--hard-negatives',type=Path,default=DEFAULT_HARDNEG_MANIFEST); p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--epochs',type=int,default=18); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--seed',type=int,default=21701)
    p.add_argument('--hidden',type=int,default=96); p.add_argument('--input-size',type=int,default=22); p.add_argument('--positive-radius',type=float,default=16.0); p.add_argument('--negative-min',type=float,default=55.0); p.add_argument('--max-negatives-per-shot',type=int,default=10); p.add_argument('--jitter',type=int,default=6)
    args=p.parse_args()
    if args.negative_min <= args.positive_radius+8:
        print('ERROR: keep a wide ambiguous band between positive and negative labels'); return 2
    tc=TrainingConfigV217(positive_radius_px=args.positive_radius,negative_min_distance_px=args.negative_min,max_negatives_per_shot=args.max_negatives_per_shot,jitter_px=args.jitter,batch_size=args.batch_size,epochs=args.epochs,seed=args.seed)
    mc=NewHoleAIConfigV217(hidden_size=args.hidden,input_size=args.input_size)
    report=run_training_v217(root=args.root,output_dir=args.out,hardnegative_manifest=args.hard_negatives,model_config=mc,training_config=tc)
    print('\nV2.17 NEW-HOLE RESULT'); print('=====================')
    print(f"Model       : {report['model_path']}"); print(f"Provisional : {report['split_is_provisional']}"); print(f"Threshold   : {report['selected_threshold']}")
    print('\nPatch classification')
    for name,result in report['evaluations'].items():
        m=result.get('classification') or {}; print(f"  {name:12s} AUC={m.get('auc')} F1={m.get('f1')} recall={m.get('recall')} specificity={m.get('specificity')}")
    print('\nCandidate ranking / <=20px')
    for name,result in report['candidate_ranking_20px'].items():
        print(f"  {name:12s} top1={result.get('top1')} top3={result.get('top3')} oracle={result.get('oracle_recall')} median_rank={result.get('median_gt_rank')}")
    print('\nGATE')
    for key,value in report['gate'].items(): print(f'  {key}: {value}')
    print('\nV2.17 stays OFFLINE/SHADOW ONLY. Old holes are valid NOT-NEW negatives, never static non-hole labels.')
    return 0
if __name__=='__main__': raise SystemExit(main())
