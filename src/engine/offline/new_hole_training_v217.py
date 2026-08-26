from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from src.engine.ai.hole_patch_ensemble_v215 import HolePatchEnsembleV215
from src.engine.ai.new_hole_ai_v217 import NewHoleAIConfigV217, NewHoleAIV217
from src.engine.offline.candidate_pack_v216 import CandidatePackV216, discover_candidate_packs
from src.engine.offline.candidate_shadow_analysis_v216 import (
    DEFAULT_ENSEMBLE_CONFIG,
    hard_negative_rows_v216,
    score_pack_v216,
)
from src.engine.offline.hole_training_v213 import choose_threshold, threshold_metrics


DEFAULT_ROOT = Path("content/ai/candidate_shadow_v216")
DEFAULT_OUT = Path("content/ai/reports/v217")
DEFAULT_MODEL = DEFAULT_OUT / "new_hole_ai_v217.npz"
DEFAULT_REPORT = DEFAULT_OUT / "new_hole_v217_report.json"
DEFAULT_HARDNEG_MANIFEST = Path("content/ai/reports/v216/hard_negatives/hard_negatives.jsonl")


@dataclass(frozen=True)
class TrainingConfigV217:
    positive_radius_px: float = 16.0
    ambiguous_radius_px: float = 42.0
    negative_min_distance_px: float = 55.0
    max_negatives_per_shot: int = 10
    include_gt_positive: bool = True
    include_nearest_candidate_positive: bool = True
    jitter_px: int = 6
    batch_size: int = 128
    epochs: int = 18
    seed: int = 21701


@dataclass
class PairSampleV217:
    shot_key: str
    session_id: str
    round_id: int
    label: int
    pre: np.ndarray
    post: np.ndarray
    post_stack: list[np.ndarray]
    offset_xy: tuple[float, float]
    source: str
    capture_index: int | None
    distance_gt_px: float | None
    likely_old_hole: bool = False
    known_hole_distance_px: float | None = None


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except Exception:
        return default


def _median_post(stack: Sequence[np.ndarray]) -> np.ndarray:
    if not stack:
        raise ValueError("empty post stack")
    if len(stack) == 1:
        return np.ascontiguousarray(stack[0])
    return np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)


def _shift_pair(pre: np.ndarray, posts: Sequence[np.ndarray], dx: int, dy: int) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = pre.shape[:2]
    matrix = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    def warp(image: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return warp(pre), [warp(item) for item in posts]


def _augment_pair(pre: np.ndarray, posts: Sequence[np.ndarray], rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
    pre_f = pre.astype(np.float32)
    posts_f = [p.astype(np.float32) for p in posts]
    # Apply the same broad photometric transform to pre/post so temporal change
    # survives while absolute projector/camera intensity varies.
    gain = float(rng.uniform(0.88, 1.12))
    bias = float(rng.uniform(-12.0, 12.0))
    gamma = float(rng.uniform(0.88, 1.12))
    sigma = float(rng.uniform(0.0, 2.4))
    blur = bool(rng.random() < 0.22)
    blur_sigma = float(rng.uniform(0.25, 0.7))

    def transform(arr: np.ndarray) -> np.ndarray:
        x = np.clip(arr * gain + bias, 0.0, 255.0) / 255.0
        x = np.power(np.clip(x, 0.0, 1.0), gamma) * 255.0
        if sigma > 0.0:
            # independent sensor noise per frame is realistic; the broad
            # transform remains shared.
            x = x + rng.normal(0.0, sigma, size=x.shape)
        x = np.clip(x, 0.0, 255.0)
        if blur:
            x = cv2.GaussianBlur(x.astype(np.float32), (3, 3), blur_sigma)
        return x.astype(np.uint8)

    return transform(pre_f), [transform(item) for item in posts_f]


def _known_hole_distance_from_row(row: dict[str, Any]) -> float | None:
    candidate = row.get("candidate") or {}
    if not isinstance(candidate, dict):
        return None
    for key in ("near_known_hole_dist", "known_hole_distance_px", "nearest_known_hole_dist"):
        value = _safe_float(candidate.get(key))
        if value is not None:
            return value
    return None


def _likely_old_hole(pre_patch: np.ndarray, post_patch: np.ndarray, ensemble: HolePatchEnsembleV215 | None, known_distance: float | None) -> bool:
    known = known_distance is not None and known_distance <= 18.0
    if ensemble is None:
        return bool(known)
    try:
        before = ensemble.score_patches([pre_patch])[0]
        after = ensemble.score_patches([post_patch])[0]
        threshold = max(0.55, float(ensemble.config.fused_threshold))
        high_before = before.fused_probability >= threshold
        high_after = after.fused_probability >= threshold
        low_gain = (after.fused_probability - before.fused_probability) <= 0.18
        return bool((high_before and high_after and low_gain) or known)
    except Exception:
        return bool(known)


def _candidate_pre_bank(pack: CandidatePackV216) -> np.ndarray:
    recent = getattr(pack, "recent_pre_patches", None)
    if isinstance(recent, np.ndarray) and recent.ndim == 3 and len(recent) == len(pack.pre_patches):
        return recent
    return pack.pre_patches


def _gt_pre_patch(pack: CandidatePackV216) -> np.ndarray | None:
    recent = getattr(pack, "gt_recent_pre_patch", None)
    return recent if recent is not None else pack.gt_pre_patch


def samples_from_pack_v217(
    pack: CandidatePackV216,
    ensemble: HolePatchEnsembleV215 | None,
    config: TrainingConfigV217,
    hard_negative_indices: Sequence[int] | None = None,
) -> list[PairSampleV217]:
    gt = pack.gt_xy
    gt_pre = _gt_pre_patch(pack)
    if gt is None or gt_pre is None or pack.gt_post_patches.size == 0:
        return []
    shot_key = f"{pack.metadata.get('session_id','unknown')}:{int(pack.metadata.get('round_id',0))}"
    session_id = str(pack.metadata.get("session_id", "unknown"))
    round_id = int(pack.metadata.get("round_id", 0))
    samples: list[PairSampleV217] = []

    if config.include_gt_positive:
        gt_stack = [np.asarray(p, dtype=np.uint8) for p in pack.gt_post_patches]
        samples.append(PairSampleV217(
            shot_key=shot_key, session_id=session_id, round_id=round_id, label=1,
            pre=np.asarray(gt_pre, dtype=np.uint8), post=_median_post(gt_stack), post_stack=gt_stack,
            offset_xy=(0.0, 0.0), source="gt_patch", capture_index=None, distance_gt_px=0.0,
        ))

    if pack.pre_patches.ndim != 3 or pack.post_patches.ndim != 4:
        return samples
    pre_bank = _candidate_pre_bank(pack)

    # Add at most one actual live candidate as positive. GT-only patches remain
    # diagnostic/training positives even when detector recall was zero.
    nearest: tuple[int, float] | None = None
    for idx, row in enumerate(pack.candidates):
        if bool(row.get("capture_forced_gt_nearest")):
            continue
        dist = _safe_float(row.get("distance_gt_px"))
        if dist is None:
            continue
        if nearest is None or dist < nearest[1]:
            nearest = (idx, dist)
    if config.include_nearest_candidate_positive and nearest is not None and nearest[1] <= config.positive_radius_px:
        idx, dist = nearest
        if idx < len(pre_bank) and idx < len(pack.post_patches):
            stack = [np.asarray(p, dtype=np.uint8) for p in pack.post_patches[idx]]
            if stack:
                row = pack.candidates[idx]
                cx = float(row.get("camera_x", 0.0)); cy = float(row.get("camera_y", 0.0))
                samples.append(PairSampleV217(
                    shot_key=shot_key, session_id=session_id, round_id=round_id, label=1,
                    pre=np.asarray(pre_bank[idx], dtype=np.uint8), post=_median_post(stack), post_stack=stack,
                    offset_xy=(float(gt[0]-cx), float(gt[1]-cy)), source="nearest_live_candidate",
                    capture_index=idx, distance_gt_px=float(dist),
                ))

    if hard_negative_indices is not None:
        hard_rows: list[dict[str, Any]] = []
        for idx in list(hard_negative_indices)[: max(1, int(config.max_negatives_per_shot))]:
            if not 0 <= int(idx) < len(pack.candidates):
                continue
            row = dict(pack.candidates[int(idx)])
            dist = _safe_float(row.get("distance_gt_px"))
            if dist is None or dist < float(config.negative_min_distance_px):
                continue
            row["capture_index"] = int(idx)
            hard_rows.append(row)
    else:
        # Fallback only. Existing V2.16 exported manifest is preferred because
        # rescoring all ~38k candidates with Hole-AI is slower.
        if ensemble is None:
            hard_rows = []
        else:
            scored = score_pack_v216(pack, ensemble)
            hard_rows = hard_negative_rows_v216(scored, min_distance_px=config.negative_min_distance_px, max_per_shot=config.max_negatives_per_shot)

    for row in hard_rows:
        idx = int(row.get("capture_index", -1))
        if idx < 0 or idx >= len(pre_bank) or idx >= len(pack.post_patches):
            continue
        dist = _safe_float(row.get("distance_gt_px"))
        if dist is None or dist < float(config.negative_min_distance_px):
            continue
        stack = [np.asarray(p, dtype=np.uint8) for p in pack.post_patches[idx]]
        if not stack:
            continue
        pre = np.asarray(pre_bank[idx], dtype=np.uint8)
        post = _median_post(stack)
        known_distance = _known_hole_distance_from_row(row)
        samples.append(PairSampleV217(
            shot_key=shot_key, session_id=session_id, round_id=round_id, label=0,
            pre=pre, post=post, post_stack=stack, offset_xy=(0.0,0.0),
            source="real_detector_far_candidate", capture_index=idx, distance_gt_px=float(dist),
            likely_old_hole=_likely_old_hole(pre, post, ensemble, known_distance),
            known_hole_distance_px=known_distance,
        ))
    return samples


def _load_hardnegative_index(manifest_path: Path) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    path = Path(manifest_path)
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            source = str(row.get("source_pack") or "")
            index = int(row.get("capture_index"))
            if source:
                result.setdefault(source, []).append(index)
                result.setdefault(str(Path(source).resolve()), []).append(index)
        except Exception:
            continue
    return result


def discover_samples_v217(
    root: Path = DEFAULT_ROOT,
    *,
    ensemble_path: Path = DEFAULT_ENSEMBLE_CONFIG,
    hardnegative_manifest: Path = DEFAULT_HARDNEG_MANIFEST,
    config: TrainingConfigV217 | None = None,
) -> tuple[list[PairSampleV217], dict[str, Any]]:
    cfg = config or TrainingConfigV217()
    paths = discover_candidate_packs(Path(root))
    hard_index = _load_hardnegative_index(Path(hardnegative_manifest))
    ensemble: HolePatchEnsembleV215 | None = None
    if not hard_index:
        try:
            ensemble = HolePatchEnsembleV215.load(Path(ensemble_path))
        except Exception:
            ensemble = None
    else:
        # Load it only for old-hole diagnostic tagging; failure is non-fatal.
        try:
            ensemble = HolePatchEnsembleV215.load(Path(ensemble_path))
        except Exception:
            ensemble = None

    samples: list[PairSampleV217] = []
    sessions: set[str] = set()
    usable_shots = 0
    recent_pre_shots = 0
    for path in paths:
        pack = CandidatePackV216.load(path)
        if _gt_pre_patch(pack) is not pack.gt_pre_patch:
            recent_pre_shots += 1
        selected = hard_index.get(str(path)) or hard_index.get(str(path.resolve()))
        shot_samples = samples_from_pack_v217(pack, ensemble, cfg, selected)
        if shot_samples:
            usable_shots += 1
            sessions.add(str(pack.metadata.get("session_id", "unknown")))
            samples.extend(shot_samples)

    summary = {
        "candidate_packs": len(paths),
        "usable_shots": usable_shots,
        "sessions": sorted(sessions),
        "samples": len(samples),
        "positives": sum(s.label == 1 for s in samples),
        "negatives_newhole": sum(s.label == 0 for s in samples),
        "likely_old_hole_negatives": sum(s.label == 0 and s.likely_old_hole for s in samples),
        "hardnegative_manifest": str(hardnegative_manifest),
        "hardnegative_manifest_used": bool(hard_index),
        "recent_pre_shots": int(recent_pre_shots),
        "semantic_note": "label=0 means NOT THE CURRENT NEW HOLE; it does not mean non-hole",
        "preference": "true recent-pre camera patches when present; legacy/reference PRE is backward-compatible fallback",
    }
    return samples, summary


def _shot_split_keys_v217(root: Path) -> tuple[dict[str, set[tuple[str,int]]], bool]:
    rows: list[tuple[str,int]] = []
    for path in discover_candidate_packs(root):
        try:
            meta = json.loads(Path(path).read_text(encoding="utf-8"))
            rows.append((str(meta.get("session_id", "unknown")), int(meta.get("round_id", 0))))
        except Exception:
            continue
    sessions = sorted({s for s,_ in rows})
    if len(sessions) >= 3:
        holdout = sessions[-1]; confirmation = sessions[-2]
        return {
            "development": {r for r in rows if r[0] not in {confirmation,holdout}},
            "confirmation": {r for r in rows if r[0] == confirmation},
            "holdout": {r for r in rows if r[0] == holdout},
        }, False
    ordered = sorted(rows)
    result = {"development":set(), "confirmation":set(), "holdout":set()}
    for index,row in enumerate(ordered):
        bucket=index%5
        if bucket==4: result["holdout"].add(row)
        elif bucket==3: result["confirmation"].add(row)
        else: result["development"].add(row)
    return result, True


def split_samples_v217(root: Path, samples: Sequence[PairSampleV217], *, ensemble_path: Path = DEFAULT_ENSEMBLE_CONFIG) -> tuple[dict[str,list[PairSampleV217]], bool]:
    del ensemble_path
    split_rounds, provisional = _shot_split_keys_v217(root)
    result = {name:[] for name in split_rounds}
    for sample in samples:
        key=(sample.session_id,sample.round_id)
        for name,keys in split_rounds.items():
            if key in keys:
                result[name].append(sample); break
    return result, provisional


def _make_training_feature(model: NewHoleAIV217, sample: PairSampleV217, rng: np.random.Generator, *, augment: bool, jitter_px: int) -> tuple[np.ndarray,np.ndarray]:
    posts=[np.asarray(p,dtype=np.uint8) for p in sample.post_stack] or [sample.post]
    pre=np.asarray(sample.pre,dtype=np.uint8)
    offset=np.asarray(sample.offset_xy,dtype=np.float32)
    if augment:
        dx=int(rng.integers(-jitter_px,jitter_px+1)) if jitter_px>0 else 0
        dy=int(rng.integers(-jitter_px,jitter_px+1)) if jitter_px>0 else 0
        pre,posts=_shift_pair(pre,posts,dx,dy)
        if sample.label==1:
            offset=offset+np.asarray([dx,dy],dtype=np.float32)
        pre,posts=_augment_pair(pre,posts,rng)
    return model.features_from_pair(pre,_median_post(posts),posts),offset


def _evaluate_samples(model: NewHoleAIV217, samples: Sequence[PairSampleV217], *, threshold: float) -> dict[str,Any]:
    if not samples:
        return {"count":0,"classification":threshold_metrics(np.array([]),np.array([]),threshold),"offset":{}}
    pairs=[(s.pre,s.post,s.post_stack) for s in samples]
    labels=np.asarray([s.label for s in samples],dtype=np.int32)
    offsets=np.asarray([s.offset_xy for s in samples],dtype=np.float32)
    probs,pred_offsets=model.predict_pairs(pairs)
    metrics=threshold_metrics(labels,probs,threshold)
    pos=labels==1
    errors=np.linalg.norm(pred_offsets[pos]-offsets[pos],axis=1) if np.any(pos) else np.array([])
    return {
        "count":len(samples),
        "classification":metrics,
        "offset":{
            "mean_error_px":None if not len(errors) else round(float(np.mean(errors)),6),
            "median_error_px":None if not len(errors) else round(float(np.median(errors)),6),
            "p95_error_px":None if not len(errors) else round(float(np.percentile(errors,95)),6),
        },
        "likely_old_hole_negative_count":int(sum(s.label==0 and s.likely_old_hole for s in samples)),
    }


def _ranking_eval(root: Path, model: NewHoleAIV217, *, split_rounds: set[tuple[str,int]], radius: float=20.0) -> dict[str,Any]:
    shots=top1=top3=oracle=0; ranks=[]
    for path in discover_candidate_packs(root):
        pack=CandidatePackV216.load(path)
        key=(str(pack.metadata.get("session_id","unknown")),int(pack.metadata.get("round_id",0)))
        if key not in split_rounds or pack.gt_xy is None or pack.pre_patches.ndim!=3 or pack.post_patches.ndim!=4:
            continue
        shots+=1
        pre_bank=_candidate_pre_bank(pack)
        rows=[]; pairs=[]
        for idx,row in enumerate(pack.candidates):
            if bool(row.get("capture_forced_gt_nearest")) or not bool(row.get("in_ranked_pool")):
                continue
            if idx>=len(pre_bank) or idx>=len(pack.post_patches): continue
            stack=[np.asarray(p,dtype=np.uint8) for p in pack.post_patches[idx]]
            if not stack: continue
            rows.append(row); pairs.append((np.asarray(pre_bank[idx],dtype=np.uint8),_median_post(stack),stack))
        if not rows: continue
        distances=[_safe_float(row.get("distance_gt_px")) for row in rows]
        if any(d is not None and d<=radius for d in distances): oracle+=1
        probs,_=model.predict_pairs(pairs)
        rank=None
        for pos,index in enumerate(np.argsort(-probs),start=1):
            d=_safe_float(rows[int(index)].get("distance_gt_px"))
            if d is not None and d<=radius:
                rank=pos; break
        if rank is not None:
            ranks.append(rank); top1+=int(rank<=1); top3+=int(rank<=3)
    denom=max(1,shots)
    return {"shots":shots,"oracle_recall":round(oracle/denom,6),"top1":round(top1/denom,6),"top3":round(top3/denom,6),"median_gt_rank":None if not ranks else float(np.median(ranks))}


def run_training_v217(
    *, root: Path=DEFAULT_ROOT, output_dir: Path=DEFAULT_OUT,
    ensemble_path: Path=DEFAULT_ENSEMBLE_CONFIG, hardnegative_manifest: Path=DEFAULT_HARDNEG_MANIFEST,
    model_config: NewHoleAIConfigV217|None=None, training_config: TrainingConfigV217|None=None,
) -> dict[str,Any]:
    root=Path(root); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    cfg=training_config or TrainingConfigV217()
    samples,dataset=discover_samples_v217(root,ensemble_path=ensemble_path,hardnegative_manifest=hardnegative_manifest,config=cfg)
    if not samples: raise RuntimeError(f"No usable V2.16 pre/post samples under {root}")
    split,provisional=split_samples_v217(root,samples)
    if not split.get("development") or not split.get("confirmation"):
        raise RuntimeError("Need development and confirmation samples")

    model_cfg=model_config or NewHoleAIConfigV217(); model=NewHoleAIV217(model_cfg,seed=cfg.seed); rng=np.random.default_rng(cfg.seed)
    dev=list(split["development"]); history=[]; best_state=None; best_key=None; best_threshold=0.5
    for epoch in range(1,int(cfg.epochs)+1):
        order=rng.permutation(len(dev)); losses=[]
        for start in range(0,len(order),int(cfg.batch_size)):
            batch=[dev[int(i)] for i in order[start:start+int(cfg.batch_size)]]
            feats=[]; labels=[]; offsets=[]
            for sample in batch:
                f,o=_make_training_feature(model,sample,rng,augment=True,jitter_px=int(cfg.jitter_px)); feats.append(f); labels.append(sample.label); offsets.append(o)
            metrics=model.train_batch(np.stack(feats),np.asarray(labels),np.stack(offsets)); losses.append(metrics.loss)
        conf_pairs=[(s.pre,s.post,s.post_stack) for s in split["confirmation"]]
        conf_labels=np.asarray([s.label for s in split["confirmation"]],dtype=np.int32)
        conf_probs,_=model.predict_pairs(conf_pairs)
        threshold,conf_metrics=choose_threshold(conf_labels,conf_probs)
        auc=float(conf_metrics.get("auc") or 0.0); f1=float(conf_metrics.get("f1") or 0.0); recall=float(conf_metrics.get("recall") or 0.0)
        key=(auc,f1,recall)
        row={"epoch":epoch,"loss":round(float(np.mean(losses)) if losses else 0.0,6),"confirmation_auc":conf_metrics.get("auc"),"confirmation_f1":conf_metrics.get("f1"),"confirmation_recall":conf_metrics.get("recall"),"threshold":round(float(threshold),6)}
        history.append(row)
        print(f"Epoch {epoch:02d}/{cfg.epochs}: loss={row['loss']:.4f} conf_auc={auc:.4f} conf_f1={f1:.4f} recall={recall:.4f}")
        if best_key is None or key>best_key:
            best_key=key; best_threshold=float(threshold); best_state=[p.copy() for p in model.parameters()]
    if best_state is not None:
        for param,saved in zip(model.parameters(),best_state): param[...] = saved

    evaluations={name:_evaluate_samples(model,rows,threshold=best_threshold) for name,rows in split.items()}
    split_rounds,_=_shot_split_keys_v217(root)
    ranking={name:_ranking_eval(root,model,split_rounds=keys,radius=20.0) for name,keys in split_rounds.items()}

    model_path=output_dir/"new_hole_ai_v217.npz"
    metadata={
        "threshold":best_threshold,"shadow_only":True,"split_is_provisional":provisional,"sessions":dataset.get("sessions"),
        "semantic_contract":{"positive":"current shot's NEW hole","negative":"not the current NEW hole; may be an old real hole or a non-hole","forbidden_use":"do not use V2.17 negative labels to retrain static Hole-AI as non-hole"},
    }
    model.save(model_path,metadata=metadata)
    conf_auc=float(evaluations["confirmation"]["classification"].get("auc") or 0.0)
    gate={
        "confirmation_auc_ge_0_70":bool(conf_auc>=0.70),
        "has_real_detector_negatives":bool(dataset.get("negatives_newhole",0)>0),
        "old_holes_not_mislabeled_as_nonholes":True,
        "eligible_for_live_authority":False,
        "eligible_for_next_offline_iteration":bool(conf_auc>=0.70 and dataset.get("negatives_newhole",0)>0),
    }
    report={
        "schema_version":"2.17","model_path":str(model_path),"shadow_only":True,"dataset":dataset,
        "split":{k:len(v) for k,v in split.items()},"split_is_provisional":provisional,
        "training_config":cfg.__dict__,"model_config":model_cfg.__dict__,"selected_threshold":round(best_threshold,6),
        "history":history,"evaluations":evaluations,"candidate_ranking_20px":ranking,"gate":gate,
        "next_requirement":">=3 independent physical/projected capture sessions before authority; V2.17 is offline/shadow only",
    }
    (output_dir/"new_hole_v217_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    return report


__all__=["TrainingConfigV217","PairSampleV217","discover_samples_v217","samples_from_pack_v217","split_samples_v217","run_training_v217","_evaluate_samples","_ranking_eval","_shot_split_keys_v217"]
