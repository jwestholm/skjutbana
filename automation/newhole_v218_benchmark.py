from __future__ import annotations
import argparse
from pathlib import Path
from src.engine.offline.candidate_ranking_training_v218 import DEFAULT_CACHE,DEFAULT_MODEL,DEFAULT_ROOT,DEFAULT_V217_MODEL,benchmark_frozen_v218

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=DEFAULT_ROOT); p.add_argument("--model",type=Path,default=DEFAULT_MODEL); p.add_argument("--v217-model",type=Path,default=DEFAULT_V217_MODEL); p.add_argument("--cache",type=Path,default=DEFAULT_CACHE); a=p.parse_args()
    r=benchmark_frozen_v218(root=a.root,model_path=a.model,v217_model_path=a.v217_model,cache_root=a.cache)
    print("V2.18 FROZEN CANDIDATE-RANK BENCHMARK\n====================================")
    for split,rows in r["results20"].items():
        print(f"\n{split.upper()} / <=20px")
        for pool in ("ranked","union"):
            print(" ",pool)
            for src in ("current","v217","v218"):
                m=rows[pool][src]; print(f"    {src:8s} top1={m['top1']:.4f} top3={m['top3']:.4f} refined_top1={m['refined_top1']:.4f} oracle={m['oracle_recall']:.4f} refined_oracle={m['refined_oracle_recall']:.4f} median={m['median_gt_rank']}")
    report_path = Path(a.model).parent / "new_hole_v218_rebenchmark.json"
    print(f"\nReport: {report_path}")
    return 0
if __name__=="__main__": raise SystemExit(main())
