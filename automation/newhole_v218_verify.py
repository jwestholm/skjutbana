from pathlib import Path
from src.engine.ai.new_hole_ranker_v218 import NewHoleRankerV218
from src.engine.offline.candidate_ranking_training_v218 import DEFAULT_MODEL

def main()->int:
    print("V2.18 NEW-HOLE RANKER VERIFY\n============================")
    if not Path(DEFAULT_MODEL).exists(): print("Model missing:",DEFAULT_MODEL); return 2
    model,meta=NewHoleRankerV218.load(DEFAULT_MODEL); md=meta.get("metadata") or {}
    print("Model:",DEFAULT_MODEL); print("Embedding dim:",model.embedding_dim); print("Context dim:",model.context_dim); print("Shadow only:",md.get("shadow_only")); print("Split provisional:",md.get("split_is_provisional")); print("V2.17 backbone:",md.get("v217_backbone")); print("Semantic:",md.get("semantic_contract"));
    ok=bool(md.get("shadow_only")) and not bool(md.get("eligible_for_live_authority",False)); print("Verified:",ok,"- V2.18 cannot grant live authority."); return 0 if ok else 3
if __name__=="__main__": raise SystemExit(main())
