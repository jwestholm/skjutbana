from __future__ import annotations
import tempfile
from pathlib import Path
import json
import numpy as np
from src.engine.ai.new_hole_ai_v217 import NewHoleAIV217
from src.engine.ai.new_hole_ranker_v218 import NewHoleRankerConfigV218,NewHoleRankerV218
from src.engine.offline.candidate_ranking_training_v218 import CandidateGroupV218,context_matrix_v218,ranking_metrics_v218,relevance_from_distances,prepare_groups_v218


def _groups(seed=1,count=48):
    rng=np.random.default_rng(seed); groups=[]
    for shot in range(count):
        n=14; pos=int(rng.integers(0,n)); emb=rng.normal(0,0.7,(n,8)).astype(np.float32)
        # Hidden signal says which candidate contains the compact new change.
        emb[pos,0]+=3.0; emb[pos,1]-=2.0
        # Base pointwise probability is deliberately fooled by candidate 0/1.
        base=np.clip(rng.uniform(.15,.55,n),0,1).astype(np.float32); wrong=(pos+3)%n; base[wrong]=.98; base[pos]=.46
        d=rng.uniform(70,180,n).astype(np.float32); d[pos]=float(rng.uniform(12,34))
        angles=rng.uniform(0,6.28,n).astype(np.float32)
        # Ground truth is at (0,0); every candidate has a geometrically correct
        # candidate->GT offset, not just the positive candidate.
        xy=np.column_stack([np.cos(angles)*d,np.sin(angles)*d]).astype(np.float32)
        target=-xy.copy()
        baseoff=target*0.03
        baseoff[pos]=target[pos]*.28
        scal=rng.uniform(0,1,(n,8)).astype(np.float32); scal[pos,2]+=1.5; scal[pos,7]+=1.0
        ranks=np.arange(1,n+1,dtype=np.int32); ranks[pos]=n; ranked=np.ones(n,bool)
        groups.append(CandidateGroupV218("s",shot,"",emb,base,baseoff,scal,d,xy,target,ranks,ranked,np.full(n,np.nan,np.float32)))
    return groups


def main()->int:
    print("V2.18 SELFTEST\n==============")
    groups=_groups(); train=groups[:34]; test=groups[34:]
    model=NewHoleRankerV218(8,context_matrix_v218(groups[0]).shape[1],NewHoleRankerConfigV218(hidden_size=24,learning_rate=.006,pairwise_loss_weight=.45,offset_loss_weight=.35),seed=7)
    emb=np.vstack([g.embedding for g in train]); ctx=np.vstack([context_matrix_v218(g) for g in train]); model.set_normalisation(np.mean(emb,0),np.std(emb,0)+1e-3,np.mean(ctx,0),np.std(ctx,0)+1e-3)
    base=ranking_metrics_v218(test,None,source="v217",pool="union",radius=20)
    rng=np.random.default_rng(4)
    for _ in range(42):
        for i in rng.permutation(len(train)):
            g=train[int(i)]; model.train_group(g.embedding,context_matrix_v218(g),relevance_from_distances(g.distances),g.target_offsets,g.base_offsets,g.distances)
    after=ranking_metrics_v218(test,model,source="v218",pool="union",radius=20)
    assert base["top1"]==0.0 and after["top1"]>=max(0.20, after["oracle_recall"]*0.85), (base,after)
    assert after["refined_top1"]>=after["top1"] and after["refined_oracle_recall"]>=after["oracle_recall"], after
    print(f"[PASS] listwise candidate training beats fooled pointwise score: top1 {base['top1']:.2f} -> {after['top1']:.2f} (raw oracle {after['oracle_recall']:.2f})")
    print(f"[PASS] offset refinement is measured separately: top1 {after['top1']:.2f} -> refined_top1 {after['refined_top1']:.2f}; oracle {after['oracle_recall']:.2f} -> refined {after['refined_oracle_recall']:.2f}")
    # Save/load must preserve rank and offsets exactly.
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"m.npz"; model.save(p,metadata={"shadow_only":True,"split_is_provisional":True}); loaded,meta=NewHoleRankerV218.load(p)
        g=test[0]; a=model.predict(g.embedding,context_matrix_v218(g),g.base_offsets); b=loaded.predict(g.embedding,context_matrix_v218(g),g.base_offsets)
        assert np.allclose(a[0],b[0]) and np.allclose(a[1],b[1]) and meta["metadata"]["shadow_only"]
    print("[PASS] V2.18 model save/load preserves listwise head, normalisation and shadow metadata")
    # Known-hole context is intentionally not part of the neural context matrix.
    g=test[1]; dim=context_matrix_v218(g).shape[1]; g.known_hole_distance[:]=0.0; assert context_matrix_v218(g).shape[1]==dim
    print("[PASS] known-hole registry remains soft diagnostic context; no hard exclusion/policy leakage in V2.18 head")

    # Real file-format plumbing: V2.16 pack -> frozen V2.17 embedding cache.
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)/"candidate"; session=root/"sessions"/"s1"; session.mkdir(parents=True)
        v217=Path(td)/"v217.npz"; NewHoleAIV217(seed=3).save(v217,metadata={"shadow_only":True})
        n=4; post_count=2; pre=np.full((n,64,64),180,np.uint8); post=np.repeat(pre[:,None,:,:],post_count,axis=1)
        post[2,:,30:35,30:35]=80
        np.savez_compressed(session/"shot_000001.npz",pre_patches=pre,post_patches=post,gt_post_patches=post[2],post_timestamps=np.array([1.0,1.1]),candidate_xy=np.array([[80,80],[110,100],[120,120],[180,180]],np.float32),gt_pre_patch=pre[2])
        candidates=[]
        for i,(x,y) in enumerate([[80,80],[110,100],[120,120],[180,180]]):
            candidates.append({"capture_index":i,"camera_x":x,"camera_y":y,"current_rank":i+1,"in_ranked_pool":True,"in_raw_pool":True,"capture_forced_gt_nearest":False,"distance_gt_px":float(((x-120)**2+(y-120)**2)**.5),"candidate":{"camera_x":x,"camera_y":y}})
        payload={"schema_version":"2.17","session_id":"s1","round_id":1,"ground_truth":{"camera_x":120,"camera_y":120},"array_file":"shot_000001.npz","candidates":candidates,"extra":{"known_holes_before_shot":[{"camera_x":80,"camera_y":80}]}}
        (session/"shot_000001.json").write_text(json.dumps(payload))
        cache=Path(td)/"cache"; first,info1=prepare_groups_v218(root,v217_model_path=v217,cache_root=cache); second,info2=prepare_groups_v218(root,v217_model_path=v217,cache_root=cache)
        assert len(first)==1 and first[0].size==4 and info1["cache_built"]==1 and info2["cache_hits"]==1
        assert np.isfinite(first[0].known_hole_distance[0]) and first[0].known_hole_distance[0] < 1e-3
    print("[PASS] V2.16 pack -> frozen V2.17 embedding cache roundtrip works and preserves known-hole provenance diagnostically")
    print("\nAll V2.18 selftests passed."); return 0
if __name__=="__main__": raise SystemExit(main())
