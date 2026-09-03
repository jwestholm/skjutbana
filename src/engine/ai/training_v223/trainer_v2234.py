from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dense_v2233 import REDUCER_FEATURE_NAMES, DenseShotRefV2233, load_dense_shot
from .model import evaluate_model, train_rank_model
from .patch_model_v2234 import PatchModelV2234, evaluate_patch_model, train_patch_model
from .patch_v2234 import PatchShotRefV2234, compile_patch_session, discover_patch_sessions, load_patch_shot
from .proposal import expand_session
from .rich_v2233 import enrich_session
from .dense_v2233 import compile_session
from .schema import CandidateTrainingRow, ShotTrainingRecord
from .trainer_v2233 import DenseSplitV2233, prepare_dense_sessions, select_dense_split

ROOT = Path("content/ai/training_v223/patch_v2234")
REPORT_ROOT = ROOT / "reports"
MODEL_ROOT = ROOT / "models"
REGISTRY_PATH = ROOT / "registry.json"
FINAL_FEATURE_NAMES_V2234: tuple[str, ...] = tuple(REDUCER_FEATURE_NAMES) + (
    "v2234_patch_score",
    "v2234_patch_rank_norm",
)


@dataclass
class PatchSplitV2234:
    mode: str
    train_refs: list[PatchShotRefV2234]
    validation_refs: list[PatchShotRefV2234]
    domain_refs: list[PatchShotRefV2234]
    train_sessions: list[str]
    validation_sessions: list[str]
    domain_session: str | None
    notes: list[str]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _map_dense_to_patch(refs: Sequence[DenseShotRefV2233], groups: dict[str, list[PatchShotRefV2234]]) -> list[PatchShotRefV2234]:
    by_key = {(p.session_id, p.sequence): p for values in groups.values() for p in values}
    return [by_key[(r.session_id, r.sequence)] for r in refs if (r.session_id, r.sequence) in by_key]


def select_patch_split(*, min_session_shots: int = 50) -> PatchSplitV2234:
    dense = select_dense_split(min_session_shots=min_session_shots)
    groups = discover_patch_sessions(min_shots=1)
    train = _map_dense_to_patch(dense.train_refs, groups)
    val = _map_dense_to_patch(dense.validation_refs, groups)
    domain = _map_dense_to_patch(dense.domain_refs, groups)
    notes = list(dense.notes)
    if len(train) != len(dense.train_refs) or len(val) != len(dense.validation_refs) or len(domain) != len(dense.domain_refs):
        notes.append("Some V2.23.3 dense refs do not yet have V2.23.4 patch banks; run v2234_prepare.")
    return PatchSplitV2234(dense.mode, train, val, domain, dense.train_sessions, dense.validation_sessions, dense.domain_session, notes)


def prepare_patch_sessions(*, session: str | None = None, force: bool = False) -> dict[str, Any]:
    """Ensure proposal -> rich -> dense cache -> V2.23.4 patch banks.

    Existing V2.23.2/V2.23.3 caches are reused.  The only expensive new step is
    candidate-centred patch-bank extraction, which is performed once per shot.
    """
    prep = prepare_dense_sessions(session=session, force_rich=False, force_cache=False, min_session_shots=1)
    if prep.get("status") != "ok":
        return {"status": prep.get("status"), "dense_prepare": prep, "patch": {}}
    sessions = list(prep.get("sessions", {}).keys())
    reports = {}
    for sid in sessions:
        print(f"[V2.23.4 PREP] patch-bank session={sid}")
        reports[sid] = compile_patch_session(sid, force=force, min_shots=1)
    return {"status": "ok", "dense_prepare": prep, "patch": reports}


def _patch_objective(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    r = metrics.get("retention20_at_k", {})
    median = metrics.get("median_positive_rank20")
    return (
        float(r.get("128", 0.0)),
        float(r.get("256", 0.0)),
        float(r.get("512", 0.0)),
        -(float(median) if median is not None else 1e9),
    )


def _to_final_records(refs: Sequence[PatchShotRefV2234], model: PatchModelV2234, *, top_k: int = 512) -> list[ShotTrainingRecord]:
    out: list[ShotTrainingRecord] = []
    base_names = tuple(REDUCER_FEATURE_NAMES)
    for idx, ref in enumerate(refs, start=1):
        shot = load_patch_shot(ref)
        scores = model.score_patches(shot.patches)
        order = np.argsort(-scores, kind="stable")[: min(int(top_k), len(scores))]
        rows: list[CandidateTrainingRow] = []
        denom = max(1, min(int(top_k), len(scores))-1)
        for rank, raw_idx in enumerate(order, start=1):
            i = int(raw_idx)
            features = {name: float(shot.dense.features[i,j]) for j,name in enumerate(base_names)}
            features["v2234_patch_score"] = float(scores[i])
            features["v2234_patch_rank_norm"] = float((rank-1)/denom)
            dist = float(shot.dense.distances[i])
            relevance = math.exp(-(dist*dist)/(2.0*8.0*8.0)) if dist <= 42.0 else 0.0
            rows.append(CandidateTrainingRow(
                candidate_id=f"{shot.dense.shot_id}:{i}",
                camera_x=float(shot.dense.xy[i,0]), camera_y=float(shot.dense.xy[i,1]),
                features=features, baseline_rank=rank, baseline_score=float(scores[i]),
                source="v2234_patch_topk", provenance=["v2234_patch_model"],
                gt_distance_px=dist, relevance=float(relevance),
            ))
        out.append(ShotTrainingRecord(
            session_id=shot.dense.session_id, shot_id=shot.dense.shot_id,
            source_kind="v2234_patch_reduced", timestamp=0.0,
            gt_camera_x=float(shot.dense.gt_xy[0]), gt_camera_y=float(shot.dense.gt_xy[1]),
            candidates=rows,
            metadata={"v2234_patch_top_k": int(top_k), "patch_model_kind": model.kind},
            schema_version="2.23.4",
        ))
        if idx == 1 or idx == len(refs) or idx % 25 == 0:
            print(f"[V2.23.4 FINAL-DATA] {idx}/{len(refs)} shot={shot.dense.shot_id} top{top_k}")
    return out


def _registry() -> dict[str, Any]:
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version":"2.23.4", "runs":[], "research_patch_champion":None, "bootstrap_best":None, "live_authority":False}


def _save_models(run_id: str, patch_model: PatchModelV2234, final_model: Any, stub: dict[str, Any]) -> dict[str, str]:
    run_dir = MODEL_ROOT / run_id
    patch_dir = run_dir / "patch_model"; patch_model.save(patch_dir)
    final_dir = run_dir / "final_ranker"; final_model.save(final_dir)
    _atomic_json(run_dir / "run.json", stub)
    return {"run_dir":str(run_dir), "patch_model_dir":str(patch_dir), "final_ranker_dir":str(final_dir)}


def train_patch_cascade_v2234(*, quick: bool = False, prepare: bool = True, seed_base: int = 2340) -> dict[str, Any]:
    started = time.perf_counter()
    if prepare:
        prep = prepare_patch_sessions(session=None)
        print(f"[V2.23.4] prepare status={prep.get('status')}")
    split = select_patch_split()
    if not split.train_refs or not split.validation_refs:
        report = {"schema_version":"2.23.4","status":"insufficient_patch_sessions","split_mode":split.mode,"notes":split.notes,"live_authority":False}
        REPORT_ROOT.mkdir(parents=True,exist_ok=True); _atomic_json(REPORT_ROOT/"latest.json", report)
        return report

    print(f"[V2.23.4 SPLIT] mode={split.mode} train={len(split.train_refs)} validation={len(split.validation_refs)} domain={len(split.domain_refs)}")
    trials = [
        {"kind":"patch_mlp","hidden":48,"filters":0,"epochs":8 if quick else 24,"lr":0.0018,"seed":seed_base},
        {"kind":"tiny_cnn","hidden":32,"filters":6,"epochs":9 if quick else 28,"lr":0.0018,"seed":seed_base+1},
        {"kind":"tiny_cnn","hidden":48,"filters":8,"epochs":11 if quick else 34,"lr":0.0013,"seed":seed_base+2},
    ]
    entries=[]; models=[]
    for tidx,cfg in enumerate(trials, start=1):
        print(f"[V2.23.4 PATCHMODEL] trial {tidx}/{len(trials)} kind={cfg['kind']} hidden={cfg['hidden']} filters={cfg['filters']} epochs={cfg['epochs']}")
        t0=time.perf_counter()
        def progress(ep:int,total:int,loss:float)->None:
            print(f"[V2.23.4 PATCHMODEL] trial={tidx} epoch={ep}/{total} loss={loss:.5f}")
        model, info = train_patch_model(
            split.train_refs, kind=cfg["kind"], hidden=cfg["hidden"], filters=max(1,cfg["filters"]),
            epochs=cfg["epochs"], learning_rate=cfg["lr"], seed=cfg["seed"], progress=progress,
        )
        val = evaluate_patch_model(model, split.validation_refs)
        entry={"trial":tidx,**cfg,"validation":val,"final_loss":info.get("loss_history",[None])[-1],"seconds":time.perf_counter()-t0}
        entries.append(entry); models.append(model)
        print(
            f"[V2.23.4 PATCHMODEL] trial={tidx} "
            f"valR128={val.get('retention20_at_k',{}).get('128',0):.3f} "
            f"valR512={val.get('retention20_at_k',{}).get('512',0):.3f} "
            f"medianRank={val.get('median_positive_rank20')} top1={val.get('conditional_top1_20_rate',0):.3f} "
            f"time={entry['seconds']:.1f}s"
        )
    # IMPORTANT: the fresh-domain session is never used to select among patch
    # trials. Selection is validation-only; domain is evaluated once afterward.
    best_idx=max(range(len(entries)), key=lambda i:_patch_objective(entries[i]["validation"]))
    best_patch=models[best_idx]; best_patch_entry=entries[best_idx]
    domain_patch = evaluate_patch_model(best_patch, split.domain_refs) if split.domain_refs else None

    top_k=512
    train_records=_to_final_records(split.train_refs,best_patch,top_k=top_k)
    val_records=_to_final_records(split.validation_refs,best_patch,top_k=top_k)
    domain_records=_to_final_records(split.domain_refs,best_patch,top_k=top_k) if split.domain_refs else []
    print(f"[V2.23.4 FINAL] patch top{top_k}: train oracle20={sum(int(r.oracle20) for r in train_records)}/{len(train_records)} validation={sum(int(r.oracle20) for r in val_records)}/{len(val_records)}")

    final_cfgs=[
        {"kind":"linear","hidden":16,"epochs":24 if quick else 65,"lr":0.012,"seed":seed_base+20},
        {"kind":"mlp","hidden":32,"epochs":32 if quick else 85,"lr":0.006,"seed":seed_base+21},
    ]
    final_entries=[]; final_models=[]
    for idx,cfg in enumerate(final_cfgs,start=1):
        print(f"[V2.23.4 FINAL] trial {idx}/{len(final_cfgs)} kind={cfg['kind']} epochs={cfg['epochs']}")
        t0=time.perf_counter()
        fm,info=train_rank_model(
            train_records,kind=cfg["kind"],hidden=cfg["hidden"],epochs=cfg["epochs"],learning_rate=cfg["lr"],
            seed=cfg["seed"],max_candidates_per_shot=top_k,feature_names=FINAL_FEATURE_NAMES_V2234,
            metadata={"schema_version":"2.23.4-final-1","patch_top_k":top_k,"split_mode":split.mode,"gt_in_model_features":False},
        )
        val=evaluate_model(fm,val_records)
        ent={"trial":idx,**cfg,"validation":val,"final_loss":info.get("loss_history",[None])[-1],"seconds":time.perf_counter()-t0}
        final_entries.append(ent); final_models.append(fm)
        print(f"[V2.23.4 FINAL] trial={idx} valTop1={val.get('conditional_top1_20_rate',0):.3f} valTop3={val.get('conditional_top3_20_rate',0):.3f} median={val.get('median_positive_rank')} time={ent['seconds']:.1f}s")
    best_final_idx=max(range(len(final_entries)),key=lambda i:(
        float(final_entries[i]["validation"].get("conditional_top1_20_rate",0)),
        float(final_entries[i]["validation"].get("conditional_top3_20_rate",0)),
        float(final_entries[i]["validation"].get("mrr20",0)),
    ))
    best_final=final_models[best_final_idx]; best_final_entry=final_entries[best_final_idx]
    domain_final=evaluate_model(best_final,domain_records) if domain_records else None

    val_patch=best_patch_entry["validation"]
    bootstrap_gate=bool(
        float(val_patch.get("retention20_at_k",{}).get("512",0.0))>=0.80
        and float(val_patch.get("retention20_at_k",{}).get("128",0.0))>=0.60
        and val_patch.get("median_positive_rank20") is not None
        and float(val_patch["median_positive_rank20"])<=100.0
    )
    domain_validated=bool(split.domain_refs)
    research_gate=bool(
        domain_validated and domain_patch and domain_final
        and float(domain_patch.get("retention20_at_k",{}).get("512",0.0))>=0.85
        and float(domain_patch.get("retention20_at_k",{}).get("128",0.0))>=0.55
        and domain_patch.get("median_positive_rank20") is not None
        and float(domain_patch["median_positive_rank20"])<=150.0
        and float(domain_final.get("conditional_top1_20_rate",0.0))>=0.10
    )
    run_id=time.strftime("%Y%m%d_%H%M%S")+f"_{split.mode}"
    stub={"schema_version":"2.23.4","run_id":run_id,"split_mode":split.mode,"bootstrap_learnability_gate":bootstrap_gate,"domain_validated":domain_validated,"research_gate_passed":research_gate,"live_authority":False}
    paths=_save_models(run_id,best_patch,best_final,stub)
    report={
        "schema_version":"2.23.4","status":"ok","run_id":run_id,
        "split":{"mode":split.mode,"train":len(split.train_refs),"validation":len(split.validation_refs),"fresh_domain":len(split.domain_refs),"train_sessions":split.train_sessions,"validation_sessions":split.validation_sessions,"domain_session":split.domain_session,"notes":split.notes},
        "patch_contract":{"channels":5,"patch_size":16,"crop_size":32,"gt_anchor_training_only":True,"candidate_pool_modified_by_gt":False},
        "patch_trials":entries,"best_patch_model":best_patch_entry,"best_patch_fresh_domain":domain_patch,
        "patch_top_k":top_k,"final_ranker_trials":final_entries,"best_final_ranker":best_final_entry,"best_final_fresh_domain":domain_final,
        "bootstrap_learnability_gate_passed":bootstrap_gate,
        "bootstrap_gate_contract":{"validation_retention20_at_512_min":0.80,"validation_retention20_at_128_min":0.60,"validation_median_positive_rank20_max":100.0},
        "domain_validated":domain_validated,"research_gate_passed":research_gate,
        "research_gate_contract":{"domain_retention20_at_512_min":0.85,"domain_retention20_at_128_min":0.55,"domain_median_positive_rank20_max":150.0,"final_domain_conditional_top1_20_min":0.10},
        "selection_discipline":"patch/final trial selection uses engineering validation only; fresh-domain is evaluated only after selection",
        "model_paths":paths,"eligible_for_live_authority":False,"elapsed_seconds":time.perf_counter()-started,
    }
    REPORT_ROOT.mkdir(parents=True,exist_ok=True); _atomic_json(REPORT_ROOT/f"run_{run_id}.json",report); _atomic_json(REPORT_ROOT/"latest.json",report)
    reg=_registry(); reg.setdefault("runs",[]).append({"run_id":run_id,"split_mode":split.mode,"bootstrap_gate":bootstrap_gate,"domain_validated":domain_validated,"research_gate":research_gate,"model_paths":paths})
    if split.mode=="single_session_bootstrap" and bootstrap_gate: reg["bootstrap_best"]=run_id
    if research_gate: reg["research_patch_champion"]=run_id
    reg["live_authority"]=False; _atomic_json(REGISTRY_PATH,reg)
    print(f"[V2.23.4 DONE] run={run_id} mode={split.mode} bootstrap_gate={bootstrap_gate} domain_validated={domain_validated} research_gate={research_gate} elapsed={report['elapsed_seconds']:.1f}s")
    return report


def cycle_v2234(*, session: str | None="latest", quick: bool=False) -> dict[str,Any]:
    print("[V2.23.4 CYCLE] phase 1/5 proposal")
    proposal=expand_session(session); print(f"[V2.23.4 CYCLE] proposal status={proposal.get('status')} processed={proposal.get('processed')}")
    print("[V2.23.4 CYCLE] phase 2/5 rich evidence")
    rich=enrich_session(session); print(f"[V2.23.4 CYCLE] rich status={rich.get('status')} processed={rich.get('processed')}")
    print("[V2.23.4 CYCLE] phase 3/5 dense numeric cache")
    dense=compile_session(session,min_shots=1); print(f"[V2.23.4 CYCLE] dense status={dense.get('status')} processed={dense.get('processed')}")
    print("[V2.23.4 CYCLE] phase 4/5 candidate patch bank")
    patch=compile_patch_session(session,min_shots=1); print(f"[V2.23.4 CYCLE] patch status={patch.get('status')} processed={patch.get('processed')}")
    print("[V2.23.4 CYCLE] phase 5/5 patch learner + final ranker")
    train=train_patch_cascade_v2234(quick=quick,prepare=False)
    return {"proposal":proposal,"rich":rich,"dense":dense,"patch":patch,"train":train}


_BG_THREAD=None

def schedule_cycle_v2234(*,session_id:str,quick:bool=True)->bool:
    global _BG_THREAD
    if _BG_THREAD is not None and _BG_THREAD.is_alive(): return False
    def worker()->None:
        print(f"[V2.23.4 AUTOTRAIN] cycle started session={session_id} (shadow only)")
        try:
            result=cycle_v2234(session=session_id,quick=quick); train=result.get("train",{})
            print(f"[V2.23.4 AUTOTRAIN] finished status={train.get('status')} mode={train.get('split',{}).get('mode')} bootstrap={train.get('bootstrap_learnability_gate_passed',False)} research={train.get('research_gate_passed',False)} elapsed={train.get('elapsed_seconds',0):.1f}s")
        except Exception as exc:
            print(f"[V2.23.4 AUTOTRAIN] failed open: {type(exc).__name__}: {exc}")
    _BG_THREAD=threading.Thread(target=worker,name="V2234Autotrain",daemon=True); _BG_THREAD.start(); return True
