from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .heatmap_model_v2236 import (
    HeatmapModelV2236,
    build_training_samples,
    evaluate_heatmap_baselines,
    evaluate_heatmap_model,
    init_heatmap_model,
    mine_heatmap_negatives,
    objective,
    train_stage,
)
from .heatmap_v2236 import (
    HeatmapShotRefV2236,
    discover_heatmap_sessions,
    prepare_heatmap_sessions,
)

ROOT = Path("content/ai/training_v223/heatmap_v2236")
REPORT_ROOT = ROOT / "reports"
MODEL_ROOT = ROOT / "models"
REGISTRY_PATH = ROOT / "registry.json"


@dataclass
class HeatmapSplitV2236:
    mode: str
    train_refs: list[HeatmapShotRefV2236]
    validation_refs: list[HeatmapShotRefV2236]
    domain_refs: list[HeatmapShotRefV2236]
    train_sessions: list[str]
    validation_sessions: list[str]
    domain_session: str | None
    notes: list[str]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _shot_partition(refs: Sequence[HeatmapShotRefV2236], fraction: float = 0.80) -> tuple[list[HeatmapShotRefV2236], list[HeatmapShotRefV2236]]:
    train: list[HeatmapShotRefV2236] = []
    val: list[HeatmapShotRefV2236] = []
    for ref in refs:
        token = f"{ref.session_id}|{ref.shot_id}|v2236".encode("utf-8")
        u = int(hashlib.sha256(token).hexdigest()[:8], 16) / 0xFFFFFFFF
        (train if u < fraction else val).append(ref)
    if not val and train: val.append(train.pop())
    if not train and val: train.append(val.pop())
    return train, val


def _session_key(sid: str, refs: Sequence[HeatmapShotRefV2236]) -> tuple[float, str]:
    mt = 0.0
    for ref in refs:
        try: mt = max(mt, ref.dense_ref.proposal_path.stat().st_mtime)
        except Exception: pass
    return mt, sid


def select_heatmap_split(*, min_session_shots: int = 50) -> HeatmapSplitV2236:
    groups = discover_heatmap_sessions(min_shots=min_session_shots)
    if not groups:
        return HeatmapSplitV2236("none", [], [], [], [], [], None, ["No >=50-shot direct-heatmap session available."])
    ordered = sorted(groups, key=lambda sid: _session_key(sid, groups[sid]))
    if len(ordered) == 1:
        sid = ordered[0]
        train, val = _shot_partition(groups[sid])
        return HeatmapSplitV2236(
            "single_session_bootstrap", train, val, [], [sid], [sid], None,
            [
                "Only one substantial heatmap F2 session exists.",
                "Deterministic same-session split is learnability-only.",
                "No domain-generalisation or live-authority claim is allowed.",
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
    return HeatmapSplitV2236(
        "fresh_session_domain", list(train), list(val), list(groups[domain_sid]),
        sorted(set(r.session_id for r in train)), val_sessions, domain_sid,
        ["Newest substantial F2 session is untouched fresh-domain validation."],
    )


def _best_baseline(baselines: dict[str, dict[str, Any]]) -> tuple[str | None, dict[str, Any] | None]:
    if not baselines:
        return None, None
    name = max(baselines, key=lambda k: objective(baselines[k]))
    return name, baselines[name]


def _registry() -> dict[str, Any]:
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "schema_version": "2.23.6",
            "runs": [],
            "bootstrap_best": None,
            "research_heatmap_champion": None,
            "live_authority": False,
        }


def _save_model(run_id: str, model: HeatmapModelV2236, stub: dict[str, Any]) -> dict[str, str]:
    run_dir = MODEL_ROOT / run_id
    model_dir = run_dir / "heatmap_model"
    model.save(model_dir)
    _atomic_json(run_dir / "run.json", stub)
    return {"run_dir": str(run_dir), "model_dir": str(model_dir)}


def train_direct_heatmap_v2236(*, quick: bool = False, prepare: bool = True, seed_base: int = 2360) -> dict[str, Any]:
    started = time.perf_counter()
    if prepare:
        prep = prepare_heatmap_sessions(session=None)
        print(f"[V2.23.6] prepare status={prep.get('status')}")
    split = select_heatmap_split()
    if not split.train_refs or not split.validation_refs:
        report = {
            "schema_version": "2.23.6", "status": "insufficient_heatmap_sessions",
            "split_mode": split.mode, "notes": split.notes, "live_authority": False,
        }
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        _atomic_json(REPORT_ROOT / "latest.json", report)
        return report

    print(f"[V2.23.6 SPLIT] mode={split.mode} train={len(split.train_refs)} validation={len(split.validation_refs)} domain={len(split.domain_refs)}")
    print("[V2.23.6 BASELINE] engineering validation")
    val_baselines = evaluate_heatmap_baselines(split.validation_refs)
    baseline_name, baseline_best = _best_baseline(val_baselines)
    if baseline_best:
        print(
            f"[V2.23.6 BASELINE] best={baseline_name} top1@20={baseline_best.get('top1_at20',0):.3f} "
            f"top3@20={baseline_best.get('top3_at20',0):.3f} median={baseline_best.get('median_error_px')}"
        )

    trials = [
        {"kind": "linear_conv", "hidden": 0, "lr": 0.014, "stage1": 10 if quick else 24, "stage2": 6 if quick else 14, "seed": seed_base},
        {"kind": "spatial_conv", "hidden": 8, "lr": 0.0045, "stage1": 12 if quick else 30, "stage2": 7 if quick else 16, "seed": seed_base + 1},
        {"kind": "spatial_conv", "hidden": 16, "lr": 0.0035, "stage1": 14 if quick else 34, "stage2": 8 if quick else 18, "seed": seed_base + 2},
    ]
    entries: list[dict[str, Any]] = []
    selected_models: list[HeatmapModelV2236] = []
    for tidx, cfg in enumerate(trials, start=1):
        print(f"[V2.23.6 MODEL] trial {tidx}/{len(trials)} kind={cfg['kind']} hidden={cfg['hidden']} direct-heatmap")
        t0 = time.perf_counter()
        model = init_heatmap_model(cfg["kind"], hidden=max(2, cfg["hidden"]), seed=cfg["seed"])
        samples1 = build_training_samples(split.train_refs, seed=cfg["seed"])

        def progress(stage: str, ep: int, total: int, loss: float) -> None:
            print(f"[V2.23.6 MODEL] trial={tidx} {stage} epoch={ep}/{total} loss={loss:.5f}")

        hist1 = train_stage(
            model, samples1, epochs=cfg["stage1"], learning_rate=cfg["lr"], l2=0.0007,
            seed=cfg["seed"]+10, stage_name="stage1", progress=progress,
        )
        stage1_model = model.clone()
        stage1_val = evaluate_heatmap_model(stage1_model, split.validation_refs, label=f"trial{tidx}-stage1")
        print(
            f"[V2.23.6 MODEL] trial={tidx} stage1 top1@20={stage1_val.get('top1_at20',0):.3f} "
            f"top3@20={stage1_val.get('top3_at20',0):.3f} median={stage1_val.get('median_error_px')}"
        )

        mined, mine_stats = mine_heatmap_negatives(stage1_model, split.train_refs, per_shot=96)
        samples2 = build_training_samples(split.train_refs, seed=cfg["seed"]+1, mined=mined)
        stage2_model = stage1_model.clone()
        hist2 = train_stage(
            stage2_model, samples2, epochs=cfg["stage2"], learning_rate=cfg["lr"]*0.45, l2=0.0010,
            seed=cfg["seed"]+20, stage_name="hardmine", progress=progress,
        )
        stage2_val = evaluate_heatmap_model(stage2_model, split.validation_refs, label=f"trial{tidx}-hardmine")
        print(
            f"[V2.23.6 MODEL] trial={tidx} hardmine top1@20={stage2_val.get('top1_at20',0):.3f} "
            f"top3@20={stage2_val.get('top3_at20',0):.3f} median={stage2_val.get('median_error_px')}"
        )
        # Hard-negative mining is allowed to lose. Keep the checkpoint that is
        # actually better on engineering validation instead of blindly using the last epoch.
        if objective(stage2_val) > objective(stage1_val):
            chosen_stage = "hardmine"
            chosen_model = stage2_model
            chosen_val = stage2_val
        else:
            chosen_stage = "stage1"
            chosen_model = stage1_model
            chosen_val = stage1_val
        chosen_model.metadata = {
            "schema_version": "2.23.6-heatmap-model-1",
            "kind": chosen_model.kind,
            "hidden": cfg["hidden"],
            "chosen_stage": chosen_stage,
            "training_shots": len(split.train_refs),
            "gt_used_only_for_training_labels": True,
            "input_maps_gt_free": True,
            "global_candidate_ranking_required": False,
            "live_authority": False,
            "seed": cfg["seed"],
        }
        entry = {
            "trial": tidx, **cfg,
            "stage1_validation": stage1_val,
            "hardmine_validation": stage2_val,
            "chosen_stage": chosen_stage,
            "validation": chosen_val,
            "mine_stats": mine_stats,
            "histories": {"stage1": hist1, "hardmine": hist2},
            "seconds": time.perf_counter() - t0,
        }
        entries.append(entry)
        selected_models.append(chosen_model)
        print(
            f"[V2.23.6 MODEL] trial={tidx} chosen={chosen_stage} "
            f"top1@20={chosen_val.get('top1_at20',0):.3f} top3@20={chosen_val.get('top3_at20',0):.3f} "
            f"median={chosen_val.get('median_error_px')} time={entry['seconds']:.1f}s"
        )

    best_idx = max(range(len(entries)), key=lambda i: objective(entries[i]["validation"]))
    best_entry = entries[best_idx]
    best_model = selected_models[best_idx]
    val = best_entry["validation"]

    # Only after engineering selection do we touch the fresh domain session.
    domain_metrics = evaluate_heatmap_model(best_model, split.domain_refs, label="fresh-domain") if split.domain_refs else None
    domain_baselines = evaluate_heatmap_baselines(split.domain_refs) if split.domain_refs else {}
    domain_baseline_name, domain_baseline_best = _best_baseline(domain_baselines)

    # Choose the practical direct-localisation policy on engineering validation.
    # This is deliberately allowed to be a deterministic physical-map baseline:
    # if the registered maps already localise well, game-readiness should not be
    # blocked by insisting that a learned model beat a solution that already works.
    policy_metrics: dict[str, dict[str, Any]] = {"model": val}
    for name, metrics in val_baselines.items():
        policy_metrics[f"baseline:{name}"] = metrics
    selected_policy = max(policy_metrics, key=lambda k: objective(policy_metrics[k]))
    selected_validation = policy_metrics[selected_policy]
    baseline_top1 = float((baseline_best or {}).get("top1_at20", 0.0))
    learned_model_beats_baseline = bool(objective(val) > objective(baseline_best or {}))
    bootstrap_signal = bool(
        float(selected_validation.get("top1_at20", 0.0)) >= 0.12
        or float(selected_validation.get("top3_at20", 0.0)) >= 0.30
        or (selected_validation.get("median_error_px") is not None and float(selected_validation["median_error_px"]) <= 200.0)
    )
    direct_path_gate = bool(
        float(selected_validation.get("top1_at20", 0.0)) >= 0.25
        and float(selected_validation.get("top3_at20", 0.0)) >= 0.50
        and selected_validation.get("median_error_px") is not None and float(selected_validation["median_error_px"]) <= 100.0
    )
    bootstrap_gate = bool(
        direct_path_gate
        and learned_model_beats_baseline
        and float(val.get("top1_at20", 0.0)) >= max(0.25, baseline_top1)
    )
    domain_validated = bool(split.domain_refs)
    domain_policy_metrics: dict[str, dict[str, Any]] = {}
    if domain_validated and domain_metrics:
        domain_policy_metrics["model"] = domain_metrics
        for name, metrics in domain_baselines.items():
            domain_policy_metrics[f"baseline:{name}"] = metrics
    selected_domain = domain_policy_metrics.get(selected_policy) if domain_policy_metrics else None
    research_gate = bool(
        domain_validated and selected_domain
        and float(selected_domain.get("top1_at20", 0.0)) >= 0.35
        and float(selected_domain.get("top3_at20", 0.0)) >= 0.60
        and selected_domain.get("median_error_px") is not None and float(selected_domain["median_error_px"]) <= 80.0
    )

    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{split.mode}"
    stub = {
        "schema_version": "2.23.6", "run_id": run_id, "split_mode": split.mode,
        "bootstrap_signal": bootstrap_signal, "bootstrap_learnability_gate": bootstrap_gate,
        "direct_path_gate": direct_path_gate, "selected_policy": selected_policy,
        "domain_validated": domain_validated, "research_gate_passed": research_gate,
        "live_authority": False,
    }
    paths = _save_model(run_id, best_model, stub)
    report = {
        "schema_version": "2.23.6",
        "status": "ok",
        "run_id": run_id,
        "split": {
            "mode": split.mode,
            "train": len(split.train_refs), "validation": len(split.validation_refs), "fresh_domain": len(split.domain_refs),
            "train_sessions": split.train_sessions, "validation_sessions": split.validation_sessions,
            "domain_session": split.domain_session, "notes": split.notes,
        },
        "contract": {
            "registered_evidence_channels": 8,
            "heatmap_stride": 4,
            "kernel_size": 5,
            "direct_localisation": True,
            "global_dense_candidate_ranking": False,
            "dense_used_for_diagnostics_and_optional_snap_only": True,
            "gt_used_for_map_generation": False,
            "gt_used_for_training_labels": True,
            "hard_negative_checkpoint_may_be_rejected": True,
            "live_authority": False,
        },
        "validation_baselines": val_baselines,
        "best_validation_baseline": {"name": baseline_name, "metrics": baseline_best},
        "selected_policy": selected_policy,
        "selected_policy_validation": selected_validation,
        "learned_model_beats_baseline": learned_model_beats_baseline,
        "direct_path_gate": direct_path_gate,
        "trials": entries,
        "best_model": best_entry,
        "fresh_domain": domain_metrics,
        "fresh_domain_baselines": domain_baselines,
        "best_fresh_domain_baseline": {"name": domain_baseline_name, "metrics": domain_baseline_best},
        "selected_policy_fresh_domain": selected_domain,
        "bootstrap_signal": bootstrap_signal,
        "bootstrap_learnability_gate": bootstrap_gate,
        "domain_validated": domain_validated,
        "research_gate_passed": research_gate,
        "model_paths": paths,
        "elapsed_seconds": time.perf_counter() - started,
        "eligible_for_live_authority": False,
        "live_authority": False,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_json(REPORT_ROOT / f"{run_id}.json", report)
    _atomic_json(REPORT_ROOT / "latest.json", report)
    reg = _registry()
    reg["schema_version"] = "2.23.6"
    reg.setdefault("runs", []).append({
        "run_id": run_id, "model_dir": paths["model_dir"], "split_mode": split.mode,
        "bootstrap_signal": bootstrap_signal, "bootstrap_gate": bootstrap_gate,
        "direct_path_gate": direct_path_gate, "selected_policy": selected_policy,
        "research_gate": research_gate, "created_at": time.time(),
    })
    reg["runs"] = reg["runs"][-30:]
    if split.mode == "single_session_bootstrap" and direct_path_gate:
        reg["bootstrap_best"] = {"run_id": run_id, "selected_policy": selected_policy, "model_dir": paths["model_dir"], "metrics": selected_validation}
    if research_gate:
        reg["research_direct_policy"] = {"run_id": run_id, "selected_policy": selected_policy, "fresh_domain": selected_domain}
        if selected_policy == "model":
            reg["research_heatmap_champion"] = {"run_id": run_id, "model_dir": paths["model_dir"], "fresh_domain": domain_metrics}
    reg["live_authority"] = False
    _atomic_json(REGISTRY_PATH, reg)
    print(
        f"[V2.23.6 DONE] run={run_id} mode={split.mode} signal={bootstrap_signal} "
        f"direct_gate={direct_path_gate} learned_gate={bootstrap_gate} policy={selected_policy} "
        f"domain_validated={domain_validated} research_gate={research_gate} "
        f"elapsed={report['elapsed_seconds']:.1f}s"
    )
    return report


_CYCLE_LOCK = threading.Lock()
_CYCLE_ACTIVE = False


def schedule_cycle_v2236(*, session_id: str | None = None, quick: bool = True) -> bool:
    global _CYCLE_ACTIVE
    with _CYCLE_LOCK:
        if _CYCLE_ACTIVE:
            return False
        _CYCLE_ACTIVE = True

    def worker() -> None:
        global _CYCLE_ACTIVE
        try:
            from .proposal import expand_session
            from .trainer_v2233 import prepare_dense_sessions
            print(f"[V2.23.6 CYCLE] proposal session={session_id or 'latest'}")
            expand_session(session_id or "latest")
            print("[V2.23.6 CYCLE] rich/dense cache")
            prepare_dense_sessions(session=session_id or "latest")
            print("[V2.23.6 CYCLE] direct heatmap cache")
            prepare_heatmap_sessions(session=session_id or "latest")
            print("[V2.23.6 CYCLE] train direct heatmap")
            train_direct_heatmap_v2236(quick=quick, prepare=False)
        except Exception as exc:
            print(f"[V2.23.6 CYCLE] failed open: {type(exc).__name__}: {exc}")
        finally:
            with _CYCLE_LOCK:
                _CYCLE_ACTIVE = False

    thread = threading.Thread(target=worker, name="V2236HeatmapCycle", daemon=True)
    thread.start()
    return True
