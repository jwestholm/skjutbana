from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.engine.ai.new_hole_ai_v217 import NewHoleAIV217
from src.engine.ai.new_hole_ranker_v218 import NewHoleRankerConfigV218, NewHoleRankerV218
from src.engine.offline.candidate_pack_v216 import CandidatePackV216, discover_candidate_packs
from src.engine.offline.new_hole_training_v217 import _candidate_pre_bank, _median_post, _safe_float, _shot_split_keys_v217


DEFAULT_ROOT = Path("content/ai/candidate_shadow_v216")
DEFAULT_V217_MODEL = Path("content/ai/reports/v217/new_hole_ai_v217.npz")
DEFAULT_V216_REPORT = Path("content/ai/reports/v216/candidate_shadow_report.json")
DEFAULT_OUT = Path("content/ai/reports/v218")
DEFAULT_CACHE = DEFAULT_OUT / "embedding_cache"
DEFAULT_MODEL = DEFAULT_OUT / "new_hole_ranker_v218.npz"
DEFAULT_REPORT = DEFAULT_OUT / "new_hole_v218_report.json"
CACHE_SCHEMA = "2.18-embedding-v1"


@dataclass(frozen=True)
class TrainingConfigV218:
    epochs: int = 32
    seed: int = 21801
    positive_radius_px: float = 42.0
    ranked_pool_primary: bool = True
    cache_float16: bool = True


@dataclass
class CandidateGroupV218:
    session_id: str
    round_id: int
    json_path: str
    embedding: np.ndarray
    base_probability: np.ndarray
    base_offsets: np.ndarray
    temporal_scalars: np.ndarray
    distances: np.ndarray
    candidate_xy: np.ndarray
    target_offsets: np.ndarray
    current_rank: np.ndarray
    in_ranked_pool: np.ndarray
    known_hole_distance: np.ndarray

    @property
    def key(self) -> tuple[str, int]:
        return self.session_id, self.round_id

    @property
    def size(self) -> int:
        return int(len(self.distances))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) <= 1:
        return np.full_like(values, 0.5, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    result = np.empty_like(values, dtype=np.float32)
    result[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return result


def _known_distance(row: dict[str, Any], pack: CandidatePackV216) -> float:
    candidate = row.get("candidate") or {}
    for key in ("near_known_hole_dist", "known_hole_distance_px", "nearest_known_hole_dist"):
        value = _safe_float(candidate.get(key)) if isinstance(candidate, dict) else None
        if value is not None:
            return float(value)
    # Future V2.17 capture stores the session-local registry in metadata. Use it
    # only as diagnostic context; the ranker does NOT hard reject known holes.
    extra = pack.metadata.get("extra") or {}
    known = (extra.get("known_holes_before_shot") if isinstance(extra, dict) else None) or pack.metadata.get("known_holes_before_shot") or []
    x = _safe_float(row.get("camera_x"), _safe_float(candidate.get("camera_x"))) if isinstance(candidate, dict) else _safe_float(row.get("camera_x"))
    y = _safe_float(row.get("camera_y"), _safe_float(candidate.get("camera_y"))) if isinstance(candidate, dict) else _safe_float(row.get("camera_y"))
    if x is None or y is None:
        return float("nan")
    distances = []
    for hole in known:
        if not isinstance(hole, dict):
            continue
        hx = _safe_float(hole.get("camera_x"), _safe_float(hole.get("x")))
        hy = _safe_float(hole.get("camera_y"), _safe_float(hole.get("y")))
        if hx is not None and hy is not None:
            distances.append(math.hypot(x - hx, y - hy))
    return float(min(distances)) if distances else float("nan")


def _cache_signature(pack_path: Path, v217_model_sha256: str) -> str:
    pack_path = Path(pack_path); npz = pack_path.with_suffix(".npz")
    parts = [CACHE_SCHEMA, str(v217_model_sha256)]
    for p in (pack_path, npz):
        st = p.stat(); parts.append(f"{p.name}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _cache_path(cache_root: Path, pack: CandidatePackV216) -> Path:
    session = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(pack.metadata.get("session_id", "session")))
    return Path(cache_root) / session / f"shot_{int(pack.metadata.get('round_id',0)):06d}.npz"


def _eligible_candidate_indices(pack: CandidatePackV216) -> list[int]:
    result = []
    n = min(len(pack.candidates), len(pack.pre_patches), len(pack.post_patches))
    for idx in range(n):
        row = pack.candidates[idx]
        if bool(row.get("capture_forced_gt_nearest")):
            continue
        if not (bool(row.get("in_ranked_pool")) or bool(row.get("in_raw_pool"))):
            continue
        if len(pack.post_patches[idx]) <= 0:
            continue
        result.append(idx)
    return result


def _build_group_uncached(pack: CandidatePackV216, backbone: NewHoleAIV217) -> CandidateGroupV218 | None:
    gt = pack.gt_xy
    if gt is None or pack.pre_patches.ndim != 3 or pack.post_patches.ndim != 4:
        return None
    indices = _eligible_candidate_indices(pack)
    if not indices:
        return None
    pre_bank = _candidate_pre_bank(pack)
    pairs = []
    distances=[]; xy=[]; ranks=[]; ranked=[]; known=[]
    for idx in indices:
        row = pack.candidates[idx]
        stack = [np.asarray(p, dtype=np.uint8) for p in pack.post_patches[idx]]
        pairs.append((np.asarray(pre_bank[idx], dtype=np.uint8), _median_post(stack), stack))
        candidate = row.get("candidate") or {}
        cx = float(_safe_float(row.get("camera_x"), _safe_float(candidate.get("camera_x"), 0.0)) or 0.0)
        cy = float(_safe_float(row.get("camera_y"), _safe_float(candidate.get("camera_y"), 0.0)) or 0.0)
        xy.append((cx,cy))
        distances.append(float(_safe_float(row.get("distance_gt_px"), math.hypot(cx-gt[0],cy-gt[1])) or 0.0))
        rank = row.get("current_rank")
        ranks.append(-1 if rank is None else int(rank))
        ranked.append(bool(row.get("in_ranked_pool")))
        known.append(_known_distance(row, pack))
    features = backbone.feature_batch(pairs)
    probs, offsets, hidden = backbone._forward(features)
    scalars = features[:, -int(backbone.config.scalar_features):]
    xy_arr=np.asarray(xy,dtype=np.float32)
    target=np.column_stack([float(gt[0])-xy_arr[:,0], float(gt[1])-xy_arr[:,1]]).astype(np.float32)
    return CandidateGroupV218(
        session_id=str(pack.metadata.get("session_id","unknown")),
        round_id=int(pack.metadata.get("round_id",0)), json_path=str(pack.json_path or ""),
        embedding=hidden.astype(np.float32), base_probability=probs.astype(np.float32),
        base_offsets=offsets.astype(np.float32), temporal_scalars=scalars.astype(np.float32),
        distances=np.asarray(distances,dtype=np.float32), candidate_xy=xy_arr, target_offsets=target,
        current_rank=np.asarray(ranks,dtype=np.int32), in_ranked_pool=np.asarray(ranked,dtype=bool),
        known_hole_distance=np.asarray(known,dtype=np.float32),
    )


def _save_group_cache(path: Path, signature: str, group: CandidateGroupV218, *, float16: bool) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    dtype=np.float16 if float16 else np.float32
    temp=path.with_suffix(path.suffix+".tmp")
    meta={"schema":CACHE_SCHEMA,"signature":signature,"session_id":group.session_id,"round_id":group.round_id,"json_path":group.json_path}
    with temp.open("wb") as f:
        np.savez_compressed(f,
            metadata_json=np.array(json.dumps(meta)),
            embedding=group.embedding.astype(dtype), base_probability=group.base_probability,
            base_offsets=group.base_offsets.astype(dtype), temporal_scalars=group.temporal_scalars.astype(dtype),
            distances=group.distances, candidate_xy=group.candidate_xy, target_offsets=group.target_offsets,
            current_rank=group.current_rank, in_ranked_pool=group.in_ranked_pool.astype(np.uint8),
            known_hole_distance=group.known_hole_distance,
        )
    temp.replace(path)


def _load_group_cache(path: Path, signature: str) -> CandidateGroupV218 | None:
    path=Path(path)
    if not path.exists(): return None
    try:
        with np.load(path,allow_pickle=False) as d:
            meta=json.loads(str(d["metadata_json"].item()))
            if meta.get("schema")!=CACHE_SCHEMA or meta.get("signature")!=signature: return None
            return CandidateGroupV218(
                session_id=str(meta["session_id"]),round_id=int(meta["round_id"]),json_path=str(meta.get("json_path","")),
                embedding=np.asarray(d["embedding"],dtype=np.float32),base_probability=np.asarray(d["base_probability"],dtype=np.float32),
                base_offsets=np.asarray(d["base_offsets"],dtype=np.float32),temporal_scalars=np.asarray(d["temporal_scalars"],dtype=np.float32),
                distances=np.asarray(d["distances"],dtype=np.float32),candidate_xy=np.asarray(d["candidate_xy"],dtype=np.float32),
                target_offsets=np.asarray(d["target_offsets"],dtype=np.float32),current_rank=np.asarray(d["current_rank"],dtype=np.int32),
                in_ranked_pool=np.asarray(d["in_ranked_pool"],dtype=np.uint8).astype(bool),known_hole_distance=np.asarray(d["known_hole_distance"],dtype=np.float32),
            )
    except Exception:
        return None


def prepare_groups_v218(
    root: Path=DEFAULT_ROOT, *, v217_model_path: Path=DEFAULT_V217_MODEL,
    cache_root: Path=DEFAULT_CACHE, rebuild_cache: bool=False, float16: bool=True,
) -> tuple[list[CandidateGroupV218], dict[str,Any]]:
    root=Path(root); v217_model_path=Path(v217_model_path); cache_root=Path(cache_root)
    if not v217_model_path.exists(): raise FileNotFoundError(f"V2.17 model not found: {v217_model_path}")
    backbone,_=NewHoleAIV217.load(v217_model_path)
    paths=discover_candidate_packs(root)
    if not paths: raise RuntimeError(f"No candidate packs under {root}")
    groups=[]; hits=misses=0; started=time.time(); model_sha=_sha256(v217_model_path)
    for number,path in enumerate(paths,start=1):
        pack=CandidatePackV216.load(path); sig=_cache_signature(Path(path),model_sha); cpath=_cache_path(cache_root,pack)
        group=None if rebuild_cache else _load_group_cache(cpath,sig)
        if group is None:
            group=_build_group_uncached(pack,backbone); misses+=1
            if group is not None: _save_group_cache(cpath,sig,group,float16=float16)
        else: hits+=1
        if group is not None: groups.append(group)
        if misses and number%10==0:
            print(f"Embedding cache: {number}/{len(paths)} shots | hits={hits} built={misses}")
    info={"groups":len(groups),"candidates":int(sum(g.size for g in groups)),"cache_hits":hits,"cache_built":misses,"seconds":round(time.time()-started,3),"v217_model":str(v217_model_path)}
    return groups,info


def context_matrix_v218(group: CandidateGroupV218) -> np.ndarray:
    p=group.base_probability.astype(np.float32)
    off=group.base_offsets.astype(np.float32)
    scal=group.temporal_scalars.astype(np.float32)
    p_pct=_percentile(p)[:,None]
    scalar_pct=np.column_stack([_percentile(scal[:,i]) for i in range(scal.shape[1])]).astype(np.float32)
    off_mag=np.linalg.norm(off,axis=1,keepdims=True)/48.0
    # No current-rank feature and no hard known-hole exclusion: V2.18 remains a
    # NEW-hole evidence model rather than policy leakage/final fusion.
    return np.concatenate([p[:,None],p_pct,off/48.0,off_mag,scal,scalar_pct],axis=1).astype(np.float32)


def relevance_from_distances(distances: np.ndarray) -> np.ndarray:
    d=np.asarray(distances,dtype=np.float32)
    return np.where(d<=8,1.0,np.where(d<=12,0.92,np.where(d<=20,0.78,np.where(d<=32,0.42,np.where(d<=42,0.16,0.0))))).astype(np.float32)


def split_groups_v218(root: Path, groups: Sequence[CandidateGroupV218]) -> tuple[dict[str,list[CandidateGroupV218]],bool]:
    keys,provisional=_shot_split_keys_v217(Path(root)); result={k:[] for k in keys}
    for g in groups:
        for name,subset in keys.items():
            if g.key in subset: result[name].append(g); break
    return result,provisional


def _normalisation(groups: Sequence[CandidateGroupV218]) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    emb=np.vstack([g.embedding for g in groups]); ctx=np.vstack([context_matrix_v218(g) for g in groups])
    em=np.mean(emb,axis=0); es=np.std(emb,axis=0); cm=np.mean(ctx,axis=0); cs=np.std(ctx,axis=0)
    return em.astype(np.float32),np.maximum(es,1e-3).astype(np.float32),cm.astype(np.float32),np.maximum(cs,1e-3).astype(np.float32)


def _mask(group: CandidateGroupV218,pool:str)->np.ndarray:
    return group.in_ranked_pool.copy() if pool=="ranked" else np.ones((group.size,),dtype=bool)


def ranking_metrics_v218(groups: Sequence[CandidateGroupV218], model: NewHoleRankerV218 | None, *, source:str="v218", pool:str="union", radius:float=20.0) -> dict[str,Any]:
    shots=oracle=refined_oracle=top1=top3=top5=ref_top1=ref_top3=0; ranks=[]; rranks=[]; selected=[]; selected_ref=[]
    for g in groups:
        mask=_mask(g,pool); idx=np.flatnonzero(mask)
        if not len(idx): continue
        shots+=1; d=g.distances[idx]
        oracle+=int(np.any(d<=radius))
        if source=="v217":
            scores=g.base_probability[idx]; pred_off=g.base_offsets[idx]
        elif source=="current":
            ranks_raw=g.current_rank[idx]; scores=np.where(ranks_raw>0,-ranks_raw.astype(np.float32),-1e9); pred_off=g.base_offsets[idx]
        elif source=="v218":
            if model is None: raise ValueError("model required")
            scores,pred_off=model.predict(g.embedding[idx],context_matrix_v218(g)[idx],g.base_offsets[idx])
        else: raise KeyError(source)
        ref_err=np.linalg.norm(pred_off-g.target_offsets[idx],axis=1)
        # Offset training is defined only for candidates <=42px from GT.  Do not
        # let an untrained far-candidate residual accidentally inflate the
        # diagnostic refined oracle by "teleporting" a false candidate.
        ref_eligible = d <= 42.0
        refined_oracle+=int(np.any(ref_eligible & (ref_err<=radius)))
        order=np.argsort(-scores,kind="mergesort")
        selected.append(float(d[order[0]]))
        selected_ref.append(float(ref_err[order[0]]) if ref_eligible[order[0]] else float(d[order[0]]))
        raw_positions=np.flatnonzero(d[order]<=radius); ref_positions=np.flatnonzero(ref_eligible[order] & (ref_err[order]<=radius))
        if len(raw_positions):
            r=int(raw_positions[0])+1; ranks.append(r); top1+=int(r<=1); top3+=int(r<=3); top5+=int(r<=5)
        if len(ref_positions):
            r=int(ref_positions[0])+1; rranks.append(r); ref_top1+=int(r<=1); ref_top3+=int(r<=3)
    denom=max(1,shots)
    return {
        "shots":shots,"oracle_recall":round(oracle/denom,6),"refined_oracle_recall":round(refined_oracle/denom,6),
        "top1":round(top1/denom,6),"top3":round(top3/denom,6),"top5":round(top5/denom,6),
        "refined_top1":round(ref_top1/denom,6),"refined_top3":round(ref_top3/denom,6),
        "median_gt_rank":None if not ranks else float(np.median(ranks)),"median_refined_gt_rank":None if not rranks else float(np.median(rranks)),
        "median_selected_error_px":None if not selected else round(float(np.median(selected)),6),
        "median_selected_refined_error_px":None if not selected_ref else round(float(np.median(selected_ref)),6),
    }


def selection_score_v218(metrics20:dict[str,Any], metrics42:dict[str,Any])->float:
    return (0.42*float(metrics20.get("top1",0))+0.18*float(metrics20.get("top3",0))+0.25*float(metrics20.get("refined_top1",0))+0.10*float(metrics42.get("top1",0))+0.05*float(metrics42.get("refined_top1",0)))


def _v216_context_report(path:Path=DEFAULT_V216_REPORT)->dict[str,Any]|None:
    try: return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception: return None


def run_training_v218(
    *, root:Path=DEFAULT_ROOT, output_dir:Path=DEFAULT_OUT, v217_model_path:Path=DEFAULT_V217_MODEL,
    cache_root:Path=DEFAULT_CACHE, model_config:NewHoleRankerConfigV218|None=None,
    training_config:TrainingConfigV218|None=None, rebuild_cache:bool=False,
)->dict[str,Any]:
    root=Path(root); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    cfg=training_config or TrainingConfigV218()
    groups,cache_info=prepare_groups_v218(root,v217_model_path=v217_model_path,cache_root=cache_root,rebuild_cache=rebuild_cache,float16=cfg.cache_float16)
    split,provisional=split_groups_v218(root,groups)
    if not split.get("development") or not split.get("confirmation"): raise RuntimeError("Need development and confirmation candidate groups")
    embedding_dim=groups[0].embedding.shape[1]; context_dim=context_matrix_v218(groups[0]).shape[1]
    model=NewHoleRankerV218(embedding_dim,context_dim,model_config or NewHoleRankerConfigV218(),seed=cfg.seed)
    model.set_normalisation(*_normalisation(split["development"]))
    rng=np.random.default_rng(cfg.seed); history=[]; best_state=None; best_key=None; best_selection=-1.0

    eligible_dev=[g for g in split["development"] if np.any(g.distances<=cfg.positive_radius_px)]
    if not eligible_dev: raise RuntimeError("No development groups contain an actual candidate <= positive radius")
    for epoch in range(1,int(cfg.epochs)+1):
        order=rng.permutation(len(eligible_dev)); losses=[]
        for pos in order:
            g=eligible_dev[int(pos)]; rel=relevance_from_distances(g.distances)
            try:
                m=model.train_group(g.embedding,context_matrix_v218(g),rel,g.target_offsets,g.base_offsets,g.distances,group_weight=float(np.max(rel)))
                losses.append(m.loss)
            except ValueError: pass
        c20=ranking_metrics_v218(split["confirmation"],model,source="v218",pool="union",radius=20.0)
        c42=ranking_metrics_v218(split["confirmation"],model,source="v218",pool="union",radius=42.0)
        select=selection_score_v218(c20,c42)
        median=c20.get("median_gt_rank") or 9999.0
        key=(select,float(c20.get("top1",0)),float(c20.get("refined_top1",0)),-float(median))
        row={"epoch":epoch,"loss":round(float(np.mean(losses)) if losses else 0.0,6),"confirmation_selection":round(select,6),"confirmation_top1":c20["top1"],"confirmation_top3":c20["top3"],"confirmation_refined_top1":c20["refined_top1"],"confirmation_median_rank":c20["median_gt_rank"]}
        history.append(row)
        print(f"Epoch {epoch:02d}/{cfg.epochs}: loss={row['loss']:.4f} conf_top1={c20['top1']:.4f} top3={c20['top3']:.4f} refined_top1={c20['refined_top1']:.4f} select={select:.4f}")
        if best_key is None or key>best_key:
            best_key=key; best_selection=select; best_state=model.state()
    if best_state is not None: model.restore(best_state)

    results={}
    for name,rows in split.items():
        result={}
        for pool in ("ranked","union"):
            result[pool]={}
            for source in ("current","v217","v218"):
                result[pool][source]={"r20":ranking_metrics_v218(rows,model if source=="v218" else None,source=source,pool=pool,radius=20.0),"r42":ranking_metrics_v218(rows,model if source=="v218" else None,source=source,pool=pool,radius=42.0)}
        results[name]=result
    conf_base=results["confirmation"]["union"]["v217"]["r20"]; conf_new=results["confirmation"]["union"]["v218"]["r20"]
    base_med=conf_base.get("median_gt_rank") or 9999.0; new_med=conf_new.get("median_gt_rank") or 9999.0
    ranking_improved=(float(conf_new["top1"])>float(conf_base["top1"]) or float(conf_new["top3"])>float(conf_base["top3"]) or new_med<base_med)
    refinement_useful=float(conf_new.get("refined_top1",0))>float(conf_new.get("top1",0)) or float(conf_new.get("refined_oracle_recall",0))>float(conf_new.get("oracle_recall",0))
    v216_context=_v216_context_report()
    v9_confirmation=None; fusion_confirmation=None
    if isinstance(v216_context,dict):
        try: v9_confirmation=v216_context["results"]["confirmation"]["raw_plus_ranked_union"]["v9"]["r20"]
        except Exception: pass
        try: fusion_confirmation=v216_context["results"]["confirmation"]["raw_plus_ranked_union"]["fusion"]["r20"]
        except Exception: pass
    best_existing_top1=max([float(conf_base.get("top1",0.0))]+([float(v9_confirmation.get("top1",0.0))] if isinstance(v9_confirmation,dict) else [])+([float(fusion_confirmation.get("top1",0.0))] if isinstance(fusion_confirmation,dict) else []))
    reaches_existing_top1=float(conf_new.get("top1",0.0))>=best_existing_top1
    model_path=output_dir/"new_hole_ranker_v218.npz"
    metadata={"shadow_only":True,"split_is_provisional":provisional,"v217_backbone":str(v217_model_path),"v217_backbone_sha256":_sha256(v217_model_path),"semantic_contract":{"ranking":"rank candidates within the SAME shot; do not interpret scores as globally calibrated P(hole)","offset":"candidate-to-current-new-hole refinement","known_holes":"diagnostic/soft context only; never hard exclusion"}}
    model.save(model_path,metadata=metadata)
    gate={"confirmation_candidate_ranking_improved_vs_v217":bool(ranking_improved),"confirmation_top1_reaches_best_v216_existing_source":bool(reaches_existing_top1),"offset_refinement_adds_signal":bool(refinement_useful),"has_listwise_training_groups":bool(len(eligible_dev)>0),"eligible_for_live_authority":False,"eligible_for_next_offline_iteration":bool(ranking_improved or refinement_useful)}
    report={"schema_version":"2.18","model_path":str(model_path),"shadow_only":True,"split_is_provisional":provisional,"cache":cache_info,"groups":{k:len(v) for k,v in split.items()},"listwise_development_groups":len(eligible_dev),"model_config":(model_config or NewHoleRankerConfigV218()).__dict__,"training_config":cfg.__dict__,"history":history,"selected_confirmation_score":round(best_selection,6),"results":results,"v216_context":v216_context,"existing_confirmation_baselines":{"v217":conf_base,"v9_v216":v9_confirmation,"fusion_v216":fusion_confirmation,"best_existing_top1":round(best_existing_top1,6)},"gate":gate,"next_requirement":"V2.18 must improve candidate ordering/refinement offline before champion/challenger automation. >=3 unseen capture sessions remain required before authority."}
    (output_dir/"new_hole_v218_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    return report


def benchmark_frozen_v218(*,root:Path=DEFAULT_ROOT,model_path:Path=DEFAULT_MODEL,v217_model_path:Path=DEFAULT_V217_MODEL,cache_root:Path=DEFAULT_CACHE)->dict[str,Any]:
    model,meta=NewHoleRankerV218.load(model_path); groups,cache=prepare_groups_v218(root,v217_model_path=v217_model_path,cache_root=cache_root)
    split,provisional=split_groups_v218(root,groups); results={}
    for name,rows in split.items():
        results[name]={}
        for pool in ("ranked","union"):
            results[name][pool]={src:ranking_metrics_v218(rows,model if src=="v218" else None,source=src,pool=pool,radius=20.0) for src in ("current","v217","v218")}
    report={"schema_version":"2.18","model_path":str(model_path),"model_metadata":meta,"split_is_provisional":provisional,"cache":cache,"results20":results}
    out=Path(model_path).parent/"new_hole_v218_rebenchmark.json"; out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8"); return report


__all__=["TrainingConfigV218","CandidateGroupV218","prepare_groups_v218","context_matrix_v218","relevance_from_distances","split_groups_v218","ranking_metrics_v218","run_training_v218","benchmark_frozen_v218"]
