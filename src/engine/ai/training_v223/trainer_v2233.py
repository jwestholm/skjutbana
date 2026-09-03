from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dense_v2233 import (
    REDUCER_FEATURE_NAMES,
    DenseShotRefV2233,
    DenseShotV2233,
    compile_session,
    discover_cached_sessions,
    discover_compilable_sessions,
    load_dense_shot,
)
from .model import evaluate_model, train_rank_model
from .proposal import expand_session
from .reducer_v2233 import ReducerModelV2233, evaluate_reducer, train_reducer
from .rich_v2233 import discover_proposal_sessions, enrich_session
from .schema import CandidateTrainingRow, ShotTrainingRecord

ROOT = Path("content/ai/training_v223/reducer_v2233")
REPORT_ROOT = ROOT / "reports"
MODEL_ROOT = ROOT / "models"
REGISTRY_PATH = ROOT / "registry.json"


@dataclass
class DenseSplitV2233:
    mode: str
    train_refs: list[DenseShotRefV2233]
    validation_refs: list[DenseShotRefV2233]
    domain_refs: list[DenseShotRefV2233]
    train_sessions: list[str]
    validation_sessions: list[str]
    domain_session: str | None
    notes: list[str]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _shot_partition(refs: Sequence[DenseShotRefV2233], fraction: float = 0.80) -> tuple[list[DenseShotRefV2233], list[DenseShotRefV2233]]:
    train = []; val = []
    for ref in refs:
        token = f"{ref.session_id}|{ref.shot_id}|v2233".encode("utf-8")
        u = int(hashlib.sha256(token).hexdigest()[:8], 16) / 0xFFFFFFFF
        (train if u < fraction else val).append(ref)
    # Deterministic safety if a tiny pathological hash split happens.
    if not val and train:
        val.append(train.pop())
    if not train and val:
        train.append(val.pop())
    return train, val


def _session_sort_key(sid: str, refs: Sequence[DenseShotRefV2233]) -> tuple[float, str]:
    mt = 0.0
    for ref in refs:
        try: mt = max(mt, ref.proposal_path.stat().st_mtime)
        except Exception: pass
    return mt, sid


def select_dense_split(*, min_session_shots: int = 50) -> DenseSplitV2233:
    groups = discover_cached_sessions(min_shots=min_session_shots)
    if not groups:
        return DenseSplitV2233("none", [], [], [], [], [], None, ["No >=50-shot rich dense F2 session available."])
    ordered = sorted(groups, key=lambda sid: _session_sort_key(sid, groups[sid]))
    if len(ordered) == 1:
        sid = ordered[0]
        train, val = _shot_partition(groups[sid])
        return DenseSplitV2233(
            "single_session_bootstrap", train, val, [], [sid], [sid], None,
            [
                "Only one substantial dense F2 session exists.",
                "Use deterministic 80/20 same-session split only to prove learnability.",
                "No research champion/domain-generalisation claim is allowed until a newer F2 session exists.",
            ],
        )
    domain_sid = ordered[-1]
    engineering = ordered[:-1]
    if len(engineering) == 1:
        train, val = _shot_partition(groups[engineering[0]])
        val_sessions = [engineering[0]]
    else:
        val_sid = engineering[-1]
        train = [ref for sid in engineering[:-1] for ref in groups[sid]]
        val = list(groups[val_sid])
        val_sessions = [val_sid]
    return DenseSplitV2233(
        "fresh_session_domain",
        list(train), list(val), list(groups[domain_sid]),
        sorted(set(r.session_id for r in train)), val_sessions, domain_sid,
        ["Newest substantial dense F2 session is untouched fresh-domain validation."],
    )


def prepare_dense_sessions(*, session: str | None = None, force_rich: bool = False, force_cache: bool = False, min_session_shots: int = 50) -> dict[str, Any]:
    """Ensure proposal -> rich features -> compact numeric cache exists.

    If session is omitted, prepare every substantial proposal session.  Cached
    steps are cheap and make subsequent overnight training repeatable.
    """
    proposal_groups = discover_proposal_sessions()
    substantial = {sid: paths for sid, paths in proposal_groups.items() if len(paths) >= min_session_shots}
    if session not in (None, "all"):
        if session == "latest" and substantial:
            session = max(substantial, key=lambda sid: max(p.stat().st_mtime for p in substantial[sid]))
        substantial = {str(session): substantial.get(str(session), [])} if str(session) in substantial else {}
    if not substantial:
        return {"status": "no_substantial_proposal_sessions", "sessions": {}}
    reports: dict[str, Any] = {}
    for sid in sorted(substantial):
        print(f"[V2.23.3 PREP] session={sid} shots={len(substantial[sid])}")
        rich = enrich_session(sid, force=force_rich)
        cache = compile_session(sid, force=force_cache, min_shots=1)
        reports[sid] = {"rich": rich, "cache": cache}
    return {"status": "ok", "sessions": reports}


def _load_refs(refs: Sequence[DenseShotRefV2233], *, label: str) -> list[DenseShotV2233]:
    shots: list[DenseShotV2233] = []
    t0 = time.perf_counter()
    for idx, ref in enumerate(refs, start=1):
        shots.append(load_dense_shot(ref))
        if idx == 1 or idx == len(refs) or idx % 25 == 0:
            print(f"[V2.23.3 LOAD] {label} {idx}/{len(refs)}")
    print(f"[V2.23.3 LOAD] {label} loaded={len(shots)} time={time.perf_counter()-t0:.1f}s")
    return shots


def _to_reduced_records(shots: Sequence[DenseShotV2233], reducer: ReducerModelV2233, *, top_k: int = 512) -> list[ShotTrainingRecord]:
    out: list[ShotTrainingRecord] = []
    names = tuple(REDUCER_FEATURE_NAMES)
    for shot in shots:
        order = reducer.rank_indices(shot)[: min(int(top_k), len(shot.features))]
        rows = []
        for rank, idx in enumerate(order, start=1):
            i = int(idx)
            features = {name: float(shot.features[i, j]) for j, name in enumerate(names)}
            dist = float(shot.distances[i])
            relevance = math.exp(-(dist * dist) / (2.0 * 8.0 * 8.0)) if dist <= 42.0 else 0.0
            row = CandidateTrainingRow(
                candidate_id=f"{shot.shot_id}:{i}", camera_x=float(shot.xy[i,0]), camera_y=float(shot.xy[i,1]),
                features=features, baseline_rank=rank, baseline_score=None,
                source="v2233_reduced_dense", provenance=["v2233_reducer_topk"],
                gt_distance_px=dist, relevance=float(relevance),
            )
            rows.append(row)
        record = ShotTrainingRecord(
            session_id=shot.session_id, shot_id=shot.shot_id, source_kind="v2233_dense_reduced",
            timestamp=0.0, gt_camera_x=float(shot.gt_xy[0]), gt_camera_y=float(shot.gt_xy[1]),
            candidates=rows, metadata={"v2233_reduced_top_k": int(top_k)}, schema_version="2.23.3",
        )
        # Distances/relevance come from the dense cache's GT labels. Do not
        # recompute them here; the model never receives those label fields.
        out.append(record)
    return out


def _registry() -> dict[str, Any]:
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": "2.23.3", "runs": [], "research_cascade_champion": None, "live_authority": False}


def _save_run_models(run_id: str, reducer: ReducerModelV2233, final_model: Any | None, report_stub: dict[str, Any]) -> dict[str, str]:
    run_dir = MODEL_ROOT / run_id
    reducer_dir = run_dir / "reducer"
    reducer.save(reducer_dir)
    paths = {"run_dir": str(run_dir), "reducer_dir": str(reducer_dir)}
    if final_model is not None:
        final_dir = run_dir / "final_ranker"
        final_model.save(final_dir)
        paths["final_ranker_dir"] = str(final_dir)
    _atomic_json(run_dir / "run.json", report_stub)
    return paths


def _reducer_objective(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    r = metrics.get("retention20_at_k", {})
    median = metrics.get("median_positive_rank20")
    return (
        float(r.get("128", 0.0)),
        float(r.get("256", 0.0)),
        float(r.get("512", 0.0)),
        -(float(median) if median is not None else 1e9),
    )


def train_cascade_v2233(*, quick: bool = False, prepare: bool = True, seed_base: int = 2330) -> dict[str, Any]:
    started = time.perf_counter()
    if prepare:
        prep = prepare_dense_sessions()
        print(f"[V2.23.3] prepare status={prep.get('status')}")
    split = select_dense_split()
    if not split.train_refs or not split.validation_refs:
        report = {"schema_version": "2.23.3", "status": "insufficient_dense_sessions", "split_mode": split.mode, "notes": split.notes, "live_authority": False}
        REPORT_ROOT.mkdir(parents=True, exist_ok=True); _atomic_json(REPORT_ROOT / "latest.json", report)
        return report

    print(f"[V2.23.3 SPLIT] mode={split.mode} train={len(split.train_refs)} validation={len(split.validation_refs)} domain={len(split.domain_refs)}")
    train_shots = _load_refs(split.train_refs, label="train")
    val_shots = _load_refs(split.validation_refs, label="validation")
    domain_shots = _load_refs(split.domain_refs, label="fresh-domain") if split.domain_refs else []

    reducer_trials = [
        {"kind": "linear", "hidden": 0, "epochs": 18 if quick else 42, "lr": 0.010, "seed": seed_base},
        {"kind": "mlp", "hidden": 32, "epochs": 24 if quick else 58, "lr": 0.006, "seed": seed_base + 1},
        {"kind": "mlp", "hidden": 64, "epochs": 28 if quick else 72, "lr": 0.0045, "seed": seed_base + 2},
    ]
    reducer_entries: list[dict[str, Any]] = []
    best_reducer: ReducerModelV2233 | None = None
    best_reducer_entry: dict[str, Any] | None = None
    for tidx, cfg in enumerate(reducer_trials, start=1):
        print(f"[V2.23.3 REDUCER] trial {tidx}/{len(reducer_trials)} kind={cfg['kind']} hidden={cfg['hidden']} epochs={cfg['epochs']}")
        t0 = time.perf_counter()
        def progress(ep: int, total: int, loss: float) -> None:
            print(f"[V2.23.3 REDUCER] trial={tidx} epoch={ep}/{total} loss={loss:.5f}")
        model, info = train_reducer(
            train_shots, kind=cfg["kind"], hidden=cfg["hidden"] or 32, epochs=cfg["epochs"],
            learning_rate=cfg["lr"], seed=cfg["seed"], progress=progress,
            metadata={"split_mode": split.mode, "train_sessions": split.train_sessions, "domain_session": split.domain_session},
        )
        val_metrics = evaluate_reducer(model, val_shots)
        domain_metrics = evaluate_reducer(model, domain_shots) if domain_shots else None
        entry = {
            "trial": tidx, "kind": cfg["kind"], "hidden": cfg["hidden"], "seed": cfg["seed"],
            "validation": val_metrics, "fresh_domain": domain_metrics,
            "final_loss": info.get("loss_history", [None])[-1], "seconds": time.perf_counter() - t0,
        }
        reducer_entries.append(entry)
        selection_metrics = domain_metrics if domain_metrics is not None else val_metrics
        print(
            f"[V2.23.3 REDUCER] trial={tidx} "
            f"valR128={val_metrics.get('retention20_at_k',{}).get('128',0):.3f} "
            f"valR512={val_metrics.get('retention20_at_k',{}).get('512',0):.3f} "
            f"valMedianRank={val_metrics.get('median_positive_rank20')} "
            + (f"domainR512={domain_metrics.get('retention20_at_k',{}).get('512',0):.3f} " if domain_metrics else "")
            + f"time={entry['seconds']:.1f}s"
        )
        if best_reducer_entry is None or _reducer_objective(selection_metrics) > _reducer_objective((best_reducer_entry.get("fresh_domain") or best_reducer_entry["validation"])):
            best_reducer = model; best_reducer_entry = entry

    assert best_reducer is not None and best_reducer_entry is not None
    top_k = 512
    train_reduced = _to_reduced_records(train_shots, best_reducer, top_k=top_k)
    val_reduced = _to_reduced_records(val_shots, best_reducer, top_k=top_k)
    domain_reduced = _to_reduced_records(domain_shots, best_reducer, top_k=top_k) if domain_shots else []
    retained_train = sum(int(r.oracle20) for r in train_reduced)
    retained_val = sum(int(r.oracle20) for r in val_reduced)
    print(f"[V2.23.3 FINAL] reducer top{top_k}: train oracle20={retained_train}/{len(train_reduced)} validation={retained_val}/{len(val_reduced)}")

    final_trials = [
        {"kind": "linear", "hidden": 16, "epochs": 25 if quick else 70, "lr": 0.014, "seed": seed_base + 20},
        {"kind": "mlp", "hidden": 32, "epochs": 34 if quick else 90, "lr": 0.007, "seed": seed_base + 21},
    ]
    final_entries = []; best_final = None; best_final_entry = None
    for idx, cfg in enumerate(final_trials, start=1):
        print(f"[V2.23.3 FINAL] trial {idx}/{len(final_trials)} kind={cfg['kind']} epochs={cfg['epochs']} (this phase is quieter)")
        t0 = time.perf_counter()
        model, info = train_rank_model(
            train_reduced, kind=cfg["kind"], hidden=cfg["hidden"], epochs=cfg["epochs"],
            learning_rate=cfg["lr"], seed=cfg["seed"], max_candidates_per_shot=top_k,
            feature_names=REDUCER_FEATURE_NAMES,
            metadata={"schema_version": "2.23.3-final-1", "reducer_top_k": top_k, "split_mode": split.mode, "gt_in_model_features": False},
        )
        val_metrics = evaluate_model(model, val_reduced)
        domain_metrics = evaluate_model(model, domain_reduced) if domain_reduced else None
        entry = {
            "trial": idx, "kind": cfg["kind"], "hidden": cfg["hidden"], "seed": cfg["seed"],
            "validation": val_metrics, "fresh_domain": domain_metrics,
            "final_loss": info.get("loss_history", [None])[-1], "seconds": time.perf_counter() - t0,
        }
        final_entries.append(entry)
        chosen = domain_metrics if domain_metrics is not None else val_metrics
        objective = (float(chosen.get("conditional_top1_20_rate",0.0)), float(chosen.get("conditional_top3_20_rate",0.0)), float(chosen.get("mrr20",0.0)))
        if best_final_entry is None:
            best_final = model; best_final_entry = entry; best_obj = objective
        elif objective > best_obj:
            best_final = model; best_final_entry = entry; best_obj = objective
        print(
            f"[V2.23.3 FINAL] trial={idx} valTop1={val_metrics.get('conditional_top1_20_rate',0):.3f} "
            f"valTop3={val_metrics.get('conditional_top3_20_rate',0):.3f} "
            + (f"domainTop1={domain_metrics.get('conditional_top1_20_rate',0):.3f} domainTop3={domain_metrics.get('conditional_top3_20_rate',0):.3f} " if domain_metrics else "")
            + f"time={entry['seconds']:.1f}s"
        )

    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{split.mode}"
    domain_validated = bool(domain_shots)
    reducer_domain = best_reducer_entry.get("fresh_domain")
    final_domain = best_final_entry.get("fresh_domain") if best_final_entry else None
    research_gate = bool(
        domain_validated
        and reducer_domain
        and float(reducer_domain.get("retention20_at_k",{}).get("512",0.0)) >= 0.90
        and final_domain
        and float(final_domain.get("conditional_top1_20_rate",0.0)) >= 0.10
    )
    report_stub = {
        "schema_version": "2.23.3", "run_id": run_id, "split_mode": split.mode,
        "domain_validated": domain_validated, "research_gate_passed": research_gate,
        "live_authority": False,
    }
    model_paths = _save_run_models(run_id, best_reducer, best_final, report_stub)
    report = {
        "schema_version": "2.23.3", "status": "ok", "run_id": run_id,
        "split": {
            "mode": split.mode, "train": len(split.train_refs), "validation": len(split.validation_refs), "fresh_domain": len(split.domain_refs),
            "train_sessions": split.train_sessions, "validation_sessions": split.validation_sessions, "domain_session": split.domain_session, "notes": split.notes,
        },
        "feature_count": len(REDUCER_FEATURE_NAMES), "feature_names": list(REDUCER_FEATURE_NAMES),
        "best_reducer": best_reducer_entry, "reducer_trials": reducer_entries,
        "reducer_top_k": top_k, "best_final_ranker": best_final_entry, "final_ranker_trials": final_entries,
        "model_paths": model_paths,
        "domain_validated": domain_validated, "research_gate_passed": research_gate,
        "research_gate_contract": {"reducer_domain_retention20_at_512_min": 0.90, "final_domain_conditional_top1_20_min": 0.10},
        "eligible_for_live_authority": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_json(REPORT_ROOT / f"run_{run_id}.json", report); _atomic_json(REPORT_ROOT / "latest.json", report)
    reg = _registry(); reg.setdefault("runs", []).append({
        "run_id": run_id, "split_mode": split.mode, "domain_validated": domain_validated,
        "research_gate_passed": research_gate, "model_paths": model_paths,
        "best_reducer": best_reducer_entry, "best_final_ranker": best_final_entry,
    })
    if research_gate:
        reg["research_cascade_champion"] = run_id
    reg["live_authority"] = False
    _atomic_json(REGISTRY_PATH, reg)
    print(f"[V2.23.3 DONE] run={run_id} mode={split.mode} domain_validated={domain_validated} research_gate={research_gate} elapsed={report['elapsed_seconds']:.1f}s")
    return report


def cycle_v2233(*, session: str | None = "latest", quick: bool = False) -> dict[str, Any]:
    print("[V2.23.3 CYCLE] phase 1/4 proposal cache")
    proposal = expand_session(session)
    print(f"[V2.23.3 CYCLE] proposal status={proposal.get('status')} processed={proposal.get('processed')}")
    print("[V2.23.3 CYCLE] phase 2/4 rich PRE/POST evidence")
    rich = enrich_session(session)
    print(f"[V2.23.3 CYCLE] rich status={rich.get('status')} processed={rich.get('processed')} cached={rich.get('cached')}")
    print("[V2.23.3 CYCLE] phase 3/4 numeric reducer cache")
    cache = compile_session(session, min_shots=1)
    print(f"[V2.23.3 CYCLE] cache status={cache.get('status')} processed={cache.get('processed')}")
    print("[V2.23.3 CYCLE] phase 4/4 learned reducer + final ranker")
    train = train_cascade_v2233(quick=quick, prepare=False)
    return {"proposal": proposal, "rich": rich, "cache": cache, "train": train}

_BG_THREAD = None

def schedule_cycle_v2233(*, session_id: str, quick: bool = True) -> bool:
    """Start the V2.23.3 offline cycle after F2, fail-open and shadow-only."""
    global _BG_THREAD
    import threading
    if _BG_THREAD is not None and _BG_THREAD.is_alive():
        return False
    def worker() -> None:
        print(f"[V2.23.3 AUTOTRAIN] cycle started session={session_id} (shadow only)")
        try:
            result = cycle_v2233(session=session_id, quick=quick)
            train = result.get("train", {})
            print(
                f"[V2.23.3 AUTOTRAIN] finished status={train.get('status')} "
                f"mode={train.get('split',{}).get('mode')} gate={train.get('research_gate_passed',False)} "
                f"elapsed={train.get('elapsed_seconds',0):.1f}s"
            )
        except Exception as exc:
            print(f"[V2.23.3 AUTOTRAIN] failed open: {type(exc).__name__}: {exc}")
    _BG_THREAD = threading.Thread(target=worker, name="V2233Autotrain", daemon=True)
    _BG_THREAD.start()
    return True
