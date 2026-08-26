from __future__ import annotations
import argparse
from pathlib import Path
from src.engine.offline.candidate_ranking_training_v218 import DEFAULT_CACHE,DEFAULT_OUT,DEFAULT_ROOT,DEFAULT_V217_MODEL,TrainingConfigV218,run_training_v218

def main()->int:
    p=argparse.ArgumentParser(description="V2.18 candidate-aware listwise NEW-hole training")
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT); p.add_argument("--out",type=Path,default=DEFAULT_OUT)
    p.add_argument("--v217-model",type=Path,default=DEFAULT_V217_MODEL); p.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    p.add_argument("--epochs",type=int,default=32); p.add_argument("--rebuild-cache",action="store_true")
    a=p.parse_args(); cfg=TrainingConfigV218(epochs=max(1,a.epochs))
    print("V2.18 CANDIDATE-AWARE NEW-HOLE RANKING")
    print("========================================")
    print("Objective : whole-shot listwise ranking + candidate->GT offset refinement")
    print("Backbone  : frozen V2.17 before/after representation")
    print("Authority : OFFLINE/SHADOW ONLY")
    report=run_training_v218(root=a.root,output_dir=a.out,v217_model_path=a.v217_model,cache_root=a.cache,training_config=cfg,rebuild_cache=a.rebuild_cache)
    print("\nV2.18 RESULT")
    print("============")
    print("Model:",report["model_path"]); print("Provisional:",report["split_is_provisional"]); print("Cache:",report["cache"])
    for split in ("development","confirmation","holdout"):
        print(f"\n{split.upper()} / raw+ranked union / <=20px")
        for src in ("current","v217","v218"):
            m=report["results"][split]["union"][src]["r20"]
            print(f"{src:8s} top1={m['top1']:.4f} top3={m['top3']:.4f} oracle={m['oracle_recall']:.4f} refined_top1={m['refined_top1']:.4f} refined_oracle={m['refined_oracle_recall']:.4f} median_rank={m['median_gt_rank']}")
    print("\nEXISTING V2.16/V2.17 CONFIRMATION BASELINES")
    print(report.get("existing_confirmation_baselines"))
    print("\nGATE")
    for k,v in report["gate"].items(): print(f"  {k}: {v}")
    print("V2.18 remains OFFLINE/SHADOW ONLY. Candidate order in the live game is unchanged.")
    return 0

if __name__=="__main__": raise SystemExit(main())
