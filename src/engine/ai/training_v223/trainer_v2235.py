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

from .dense_v2233 import REDUCER_FEATURE_NAMES, DenseShotRefV2233
from .evidence_model_v2235 import EvidenceModelV2235, evaluate_evidence_model, train_evidence_model
from .evidence_patch_v2235 import (
    EvidenceShotRefV2235,
    compile_evidence_session,
    discover_evidence_sessions,
    load_evidence_shot,
)
from .model import evaluate_model, train_rank_model
from .proposal import expand_session
from .rich_v2233 import enrich_session
from .dense_v2233 import compile_session
from .schema import CandidateTrainingRow, ShotTrainingRecord
from .trainer_v2233 import prepare_dense_sessions, select_dense_split

ROOT = Path("content/ai/training_v223/evidence_v2235")
REPORT_ROOT = ROOT / "reports"
MODEL_ROOT = ROOT / "models"
REGISTRY_PATH = ROOT / "registry.json"
FINAL_FEATURE_NAMES_V2235: tuple[str, ...] = tuple(REDUCER_FEATURE_NAMES) + (
    "v2235_evidence_score",
    "v2235_evidence_rank_norm",
)


@dataclass
class EvidenceSplitV2235:
    mode: str
    train_refs: list[EvidenceShotRefV2235]
    validation_refs: list[EvidenceShotRefV2235]
    domain_refs: list[EvidenceShotRefV2235]
    train_sessions: list[str]
    validation_sessions: list[str]
    domain_session: str | None
    notes: list[str]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _map_dense_to_evidence(refs: Sequence[DenseShotRefV2233], groups: dict[str, list[EvidenceShotRefV2235]]) -> list[EvidenceShotRefV2235]:
    by_key = {(p.session_id, p.sequence): p for values in groups.values() for p in values}
    return [by_key[(r.session_id, r.sequence)] for r in refs if (r.session_id, r.sequence) in by_key]


def select_evidence_split(*, min_session_shots: int = 50) -> EvidenceSplitV2235:
    dense = select_dense_split(min_session_shots=min_session_shots)
    groups = discover_evidence_sessions(min_shots=1)
    train = _map_dense_to_evidence(dense.train_refs, groups)
    val = _map_dense_to_evidence(dense.validation_refs, groups)
    domain = _map_dense_to_evidence(dense.domain_refs, groups)
    notes = list(dense.notes)
    if len(train) != len(dense.train_refs) or len(val) != len(dense.validation_refs) or len(domain) != len(dense.domain_refs):
        notes.append("Some dense refs do not yet have V2.23.5 registered-evidence banks; run v2235_prepare.")
    return EvidenceSplitV2235(
        dense.mode, train, val, domain,
        dense.train_sessions, dense.validation_sessions, dense.domain_session, notes,
    )


def prepare_evidence_sessions(*, session: str | None = None, force: bool = False) -> dict[str, Any]:
    # Reuse V2.23.2 proposal + V2.23.3 rich/dense cache.  V2.23.5's new
    # expensive work is rebuilding the registered physical maps and compiling
    # local evidence patches once per shot.
    prep = prepare_dense_sessions(session=session, force_rich=False, force_cache=False, min_session_shots=1)
    if prep.get("status") != "ok":
        return {"status": prep.get("status"), "dense_prepare": prep, "evidence": {}}
    sessions = list(prep.get("sessions", {}).keys())
    reports: dict[str, Any] = {}
    for sid in sessions:
        print(f"[V2.23.5 PREP] registered-evidence session={sid}")
        reports[sid] = compile_evidence_session(sid, force=force, min_shots=1)
    return {"status": "ok", "dense_prepare": prep, "evidence": reports}


def _objective(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    r = metrics.get("retention20_at_k", {})
    median = metrics.get("median_positive_rank20")
    # The first-stage job is recall preservation.  Prefer 512 survival first,
    # then 128 compression, before fine rank position.
    return (
        float(r.get("512", 0.0)),
        float(r.get("128", 0.0)),
        float(r.get("1024", 0.0)),
        -(float(median) if median is not None else 1e9),
    )


def _to_final_records(refs: Sequence[EvidenceShotRefV2235], model: EvidenceModelV2235, *, top_k: int = 512) -> list[ShotTrainingRecord]:
    out: list[ShotTrainingRecord] = []
    base_names = tuple(REDUCER_FEATURE_NAMES)
    for idx, ref in enumerate(refs, start=1):
        shot = load_evidence_shot(ref)
        scores = model.score_patches(shot.patches)
        order = np.argsort(-scores, kind="stable")[: min(int(top_k), len(scores))]
        rows: list[CandidateTrainingRow] = []
        denom = max(1, min(int(top_k), len(scores)) - 1)
        for rank, raw_idx in enumerate(order, start=1):
            i = int(raw_idx)
            features = {name: float(shot.dense.features[i, j]) for j, name in enumerate(base_names)}
            features["v2235_evidence_score"] = float(scores[i])
            features["v2235_evidence_rank_norm"] = float((rank - 1) / denom)
            dist = float(shot.dense.distances[i])
            relevance = math.exp(-(dist * dist) / (2.0 * 8.0 * 8.0)) if dist <= 42.0 else 0.0
            rows.append(CandidateTrainingRow(
                candidate_id=f"{shot.dense.shot_id}:{i}",
                camera_x=float(shot.dense.xy[i, 0]),
                camera_y=float(shot.dense.xy[i, 1]),
                features=features,
                baseline_rank=rank,
                baseline_score=float(scores[i]),
                source="v2235_registered_evidence_topk",
                provenance=["v2235_registered_evidence_model"],
                gt_distance_px=dist,
                relevance=float(relevance),
            ))
        out.append(ShotTrainingRecord(
            session_id=shot.dense.session_id,
            shot_id=shot.dense.shot_id,
            source_kind="v2235_registered_evidence_reduced",
            timestamp=0.0,
            gt_camera_x=float(shot.dense.gt_xy[0]),
            gt_camera_y=float(shot.dense.gt_xy[1]),
            candidates=rows,
            metadata={"v2235_top_k": int(top_k), "evidence_model_kind": model.kind},
            schema_version="2.23.5",
        ))
        if idx == 1 or idx == len(refs) or idx % 25 == 0:
            print(f"[V2.23.5 FINAL-DATA] {idx}/{len(refs)} shot={shot.dense.shot_id} top{top_k}")
    return out


def _registry() -> dict[str, Any]:
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": "2.23.5", "runs": [], "research_evidence_champion": None, "bootstrap_best": None, "live_authority": False}


def _save_models(run_id: str, evidence_model: EvidenceModelV2235, final_model: Any, stub: dict[str, Any]) -> dict[str, str]:
    run_dir = MODEL_ROOT / run_id
    evidence_dir = run_dir / "evidence_model"
    evidence_model.save(evidence_dir)
    final_dir = run_dir / "final_ranker"
    final_model.save(final_dir)
    _atomic_json(run_dir / "run.json", stub)
    return {"run_dir": str(run_dir), "evidence_model_dir": str(evidence_dir), "final_ranker_dir": str(final_dir)}


def train_registered_evidence_cascade_v2235(*, quick: bool = False, prepare: bool = True, seed_base: int = 2350) -> dict[str, Any]:
    started = time.perf_counter()
    if prepare:
        prep = prepare_evidence_sessions(session=None)
        print(f"[V2.23.5] prepare status={prep.get('status')}")
    split = select_evidence_split()
    if not split.train_refs or not split.validation_refs:
        report = {"schema_version": "2.23.5", "status": "insufficient_evidence_sessions", "split_mode": split.mode, "notes": split.notes, "live_authority": False}
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        _atomic_json(REPORT_ROOT / "latest.json", report)
        return report

    print(f"[V2.23.5 SPLIT] mode={split.mode} train={len(split.train_refs)} validation={len(split.validation_refs)} domain={len(split.domain_refs)}")
    trials = [
        {"kind": "linear", "hidden": 0, "lr": 0.010, "seed": seed_base},
        {"kind": "mlp", "hidden": 48, "lr": 0.0040, "seed": seed_base + 1},
        {"kind": "mlp", "hidden": 72, "lr": 0.0032, "seed": seed_base + 2},
    ]
    entries: list[dict[str, Any]] = []
    models: list[EvidenceModelV2235] = []
    for tidx, cfg in enumerate(trials, start=1):
        print(f"[V2.23.5 MODEL] trial {tidx}/{len(trials)} kind={cfg['kind']} hidden={cfg['hidden']} hard-negative-mining=2 rounds")
        t0 = time.perf_counter()

        def progress(stage: str, ep: int, total: int, loss: float) -> None:
            print(f"[V2.23.5 MODEL] trial={tidx} {stage} epoch={ep}/{total} loss={loss:.5f}")

        model, info = train_evidence_model(
            split.train_refs,
            kind=cfg["kind"],
            hidden=max(1, cfg["hidden"]),
            quick=quick,
            learning_rate=cfg["lr"],
            seed=cfg["seed"],
            progress=progress,
        )
        val = evaluate_evidence_model(model, split.validation_refs)
        entry = {"trial": tidx, **cfg, "validation": val, "training": info, "seconds": time.perf_counter() - t0}
        entries.append(entry)
        models.append(model)
        print(
            f"[V2.23.5 MODEL] trial={tidx} "
            f"valR128={val.get('retention20_at_k',{}).get('128',0):.3f} "
            f"valR512={val.get('retention20_at_k',{}).get('512',0):.3f} "
            f"medianRank={val.get('median_positive_rank20')} top1={val.get('conditional_top1_20_rate',0):.3f} "
            f"time={entry['seconds']:.1f}s"
        )

    # Trial selection is engineering-validation only.  Fresh domain remains
    # untouched until the model has been selected.
    best_idx = max(range(len(entries)), key=lambda i: _objective(entries[i]["validation"]))
    best_model = models[best_idx]
    best_entry = entries[best_idx]
    domain_evidence = evaluate_evidence_model(best_model, split.domain_refs) if split.domain_refs else None

    top_k = 512
    train_records = _to_final_records(split.train_refs, best_model, top_k=top_k)
    val_records = _to_final_records(split.validation_refs, best_model, top_k=top_k)
    domain_records = _to_final_records(split.domain_refs, best_model, top_k=top_k) if split.domain_refs else []
    print(
        f"[V2.23.5 FINAL] evidence top{top_k}: "
        f"train oracle20={sum(int(r.oracle20) for r in train_records)}/{len(train_records)} "
        f"validation={sum(int(r.oracle20) for r in val_records)}/{len(val_records)}"
    )

    final_cfgs = [
        {"kind": "linear", "hidden": 16, "epochs": 24 if quick else 65, "lr": 0.012, "seed": seed_base + 20},
        {"kind": "mlp", "hidden": 32, "epochs": 32 if quick else 85, "lr": 0.006, "seed": seed_base + 21},
    ]
    final_entries: list[dict[str, Any]] = []
    final_models: list[Any] = []
    for idx, cfg in enumerate(final_cfgs, start=1):
        print(f"[V2.23.5 FINAL] trial {idx}/{len(final_cfgs)} kind={cfg['kind']} epochs={cfg['epochs']}")
        t0 = time.perf_counter()
        fm, info = train_rank_model(
            train_records,
            kind=cfg["kind"], hidden=cfg["hidden"], epochs=cfg["epochs"], learning_rate=cfg["lr"],
            seed=cfg["seed"], max_candidates_per_shot=top_k, feature_names=FINAL_FEATURE_NAMES_V2235,
            metadata={"schema_version": "2.23.5-final-1", "evidence_top_k": top_k, "split_mode": split.mode, "gt_in_model_features": False},
        )
        val = evaluate_model(fm, val_records)
        ent = {"trial": idx, **cfg, "validation": val, "final_loss": info.get("loss_history", [None])[-1], "seconds": time.perf_counter() - t0}
        final_entries.append(ent)
        final_models.append(fm)
        print(
            f"[V2.23.5 FINAL] trial={idx} valTop1={val.get('conditional_top1_20_rate',0):.3f} "
            f"valTop3={val.get('conditional_top3_20_rate',0):.3f} median={val.get('median_positive_rank')} time={ent['seconds']:.1f}s"
        )
    best_final_idx = max(range(len(final_entries)), key=lambda i: (
        float(final_entries[i]["validation"].get("conditional_top1_20_rate", 0.0)),
        float(final_entries[i]["validation"].get("conditional_top3_20_rate", 0.0)),
        float(final_entries[i]["validation"].get("mrr20", 0.0)),
    ))
    best_final = final_models[best_final_idx]
    best_final_entry = final_entries[best_final_idx]
    domain_final = evaluate_model(best_final, domain_records) if domain_records else None

    val = best_entry["validation"]
    bootstrap_gate = bool(
        float(val.get("retention20_at_k", {}).get("512", 0.0)) >= 0.70
        and float(val.get("retention20_at_k", {}).get("128", 0.0)) >= 0.45
        and val.get("median_positive_rank20") is not None
        and float(val["median_positive_rank20"]) <= 200.0
    )
    domain_validated = bool(split.domain_refs)
    research_gate = bool(
        domain_validated and domain_evidence and domain_final
        and float(domain_evidence.get("retention20_at_k", {}).get("512", 0.0)) >= 0.80
        and float(domain_evidence.get("retention20_at_k", {}).get("128", 0.0)) >= 0.55
        and domain_evidence.get("median_positive_rank20") is not None
        and float(domain_evidence["median_positive_rank20"]) <= 150.0
        and float(domain_final.get("conditional_top1_20_rate", 0.0)) >= 0.10
    )

    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{split.mode}"
    stub = {
        "schema_version": "2.23.5", "run_id": run_id, "split_mode": split.mode,
        "bootstrap_learnability_gate": bootstrap_gate,
        "domain_validated": domain_validated,
        "research_gate_passed": research_gate,
        "live_authority": False,
    }
    paths = _save_models(run_id, best_model, best_final, stub)
    report = {
        "schema_version": "2.23.5",
        "status": "ok",
        "run_id": run_id,
        "split": {
            "mode": split.mode,
            "train": len(split.train_refs),
            "validation": len(split.validation_refs),
            "fresh_domain": len(split.domain_refs),
            "train_sessions": split.train_sessions,
            "validation_sessions": split.validation_sessions,
            "domain_session": split.domain_session,
            "notes": split.notes,
        },
        "evidence_contract": {
            "registered_map_channels": 8,
            "patch_size": 9,
            "crop_size": 27,
            "candidate_positive_radius_px": 6.0,
            "neutral_band_px": [6.0, 42.0],
            "negative_radius_gt_px": 42.0,
            "gt_anchor_training_only": True,
            "candidate_pool_modified_by_gt": False,
            "hard_negative_mining_rounds": 2,
        },
        "evidence_trials": entries,
        "best_evidence_model": best_entry,
        "best_evidence_fresh_domain": domain_evidence,
        "evidence_top_k": top_k,
        "final_ranker_trials": final_entries,
        "best_final_ranker": best_final_entry,
        "best_final_fresh_domain": domain_final,
        "bootstrap_learnability_gate_passed": bootstrap_gate,
        "bootstrap_gate_contract": {
            "validation_retention20_at_512_min": 0.70,
            "validation_retention20_at_128_min": 0.45,
            "validation_median_positive_rank20_max": 200.0,
        },
        "domain_validated": domain_validated,
        "research_gate_passed": research_gate,
        "research_gate_contract": {
            "domain_retention20_at_512_min": 0.80,
            "domain_retention20_at_128_min": 0.55,
            "domain_median_positive_rank20_max": 150.0,
            "final_domain_conditional_top1_20_min": 0.10,
        },
        "selection_discipline": "evidence/final trial selection uses engineering validation only; fresh-domain evaluated only after selection",
        "model_paths": paths,
        "eligible_for_live_authority": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_json(REPORT_ROOT / f"run_{run_id}.json", report)
    _atomic_json(REPORT_ROOT / "latest.json", report)
    reg = _registry()
    reg.setdefault("runs", []).append({
        "run_id": run_id,
        "split_mode": split.mode,
        "bootstrap_gate": bootstrap_gate,
        "domain_validated": domain_validated,
        "research_gate": research_gate,
        "model_paths": paths,
    })
    if split.mode == "single_session_bootstrap" and bootstrap_gate:
        reg["bootstrap_best"] = run_id
    if research_gate:
        reg["research_evidence_champion"] = run_id
    reg["live_authority"] = False
    _atomic_json(REGISTRY_PATH, reg)
    print(
        f"[V2.23.5 DONE] run={run_id} mode={split.mode} bootstrap_gate={bootstrap_gate} "
        f"domain_validated={domain_validated} research_gate={research_gate} elapsed={report['elapsed_seconds']:.1f}s"
    )
    return report


def cycle_v2235(*, session: str | None = "latest", quick: bool = False) -> dict[str, Any]:
    print("[V2.23.5 CYCLE] phase 1/5 proposal")
    proposal = expand_session(session)
    print(f"[V2.23.5 CYCLE] proposal status={proposal.get('status')} processed={proposal.get('processed')}")
    print("[V2.23.5 CYCLE] phase 2/5 rich numeric evidence")
    rich = enrich_session(session)
    print(f"[V2.23.5 CYCLE] rich status={rich.get('status')} processed={rich.get('processed')}")
    print("[V2.23.5 CYCLE] phase 3/5 dense numeric cache")
    dense = compile_session(session, min_shots=1)
    print(f"[V2.23.5 CYCLE] dense status={dense.get('status')} processed={dense.get('processed')}")
    print("[V2.23.5 CYCLE] phase 4/5 registered evidence patch bank")
    evidence = compile_evidence_session(session, min_shots=1)
    print(f"[V2.23.5 CYCLE] evidence status={evidence.get('status')} processed={evidence.get('processed')}")
    print("[V2.23.5 CYCLE] phase 5/5 hard-mined evidence learner + final ranker")
    train = train_registered_evidence_cascade_v2235(quick=quick, prepare=False)
    return {"proposal": proposal, "rich": rich, "dense": dense, "evidence": evidence, "train": train}


_BG_THREAD = None


def schedule_cycle_v2235(*, session_id: str, quick: bool = True) -> bool:
    global _BG_THREAD
    if _BG_THREAD is not None and _BG_THREAD.is_alive():
        return False

    def worker() -> None:
        print(f"[V2.23.5 AUTOTRAIN] cycle started session={session_id} (shadow only)")
        try:
            result = cycle_v2235(session=session_id, quick=quick)
            train = result.get("train", {})
            print(
                f"[V2.23.5 AUTOTRAIN] finished status={train.get('status')} mode={train.get('split',{}).get('mode')} "
                f"bootstrap={train.get('bootstrap_learnability_gate_passed',False)} "
                f"research={train.get('research_gate_passed',False)} elapsed={train.get('elapsed_seconds',0):.1f}s"
            )
        except Exception as exc:
            print(f"[V2.23.5 AUTOTRAIN] failed open: {type(exc).__name__}: {exc}")

    _BG_THREAD = threading.Thread(target=worker, name="V2235Autotrain", daemon=True)
    _BG_THREAD.start()
    return True
