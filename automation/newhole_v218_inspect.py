from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from src.engine.offline.candidate_ranking_training_v218 import DEFAULT_CACHE,DEFAULT_ROOT,DEFAULT_V217_MODEL,prepare_groups_v218,split_groups_v218

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=DEFAULT_ROOT); p.add_argument("--v217-model",type=Path,default=DEFAULT_V217_MODEL); p.add_argument("--cache",type=Path,default=DEFAULT_CACHE); p.add_argument("--rebuild-cache",action="store_true"); a=p.parse_args()
    groups,info=prepare_groups_v218(a.root,v217_model_path=a.v217_model,cache_root=a.cache,rebuild_cache=a.rebuild_cache); split,prov=split_groups_v218(a.root,groups)
    print("V2.18 CANDIDATE GROUP INSPECTION\n================================")
    print("Groups:",len(groups)); print("Candidates:",sum(g.size for g in groups)); print("Cache:",info); print("Split provisional:",prov)
    print("Sessions:",sorted({g.session_id for g in groups}))
    print("Groups with candidate <=20px:",sum(bool(np.any(g.distances<=20)) for g in groups))
    print("Groups with candidate <=42px:",sum(bool(np.any(g.distances<=42)) for g in groups))
    print("Known-hole distance present:",sum(int(np.any(np.isfinite(g.known_hole_distance))) for g in groups),"shots (diagnostic only)")
    for name,rows in split.items(): print(f"  {name}: shots={len(rows)} <=20={sum(bool(np.any(g.distances<=20)) for g in rows)} <=42={sum(bool(np.any(g.distances<=42)) for g in rows)}")
    return 0
if __name__=="__main__": raise SystemExit(main())
