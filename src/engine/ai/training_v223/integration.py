from __future__ import annotations

import math
import threading
import time
from typing import Any, Mapping

from .capture import TrainingCaptureV223
from .framepack import save_scene_framepack
from .registry import load_champion_model
from .trainer import schedule_quick_autotrain_v223

_INSTALLED = False


def _scene_background(scene: Any) -> str:
    try:
        return str(scene.MODE_NAMES[scene.bg_mode_index])
    except Exception:
        return "unknown"


def _frame_shape(scene: Any) -> tuple[int, int] | None:
    try:
        gray = scene.runtime.post_shot_gray
        if gray is None:
            gray = scene.runtime.pre_shot_gray
        if gray is not None:
            return int(gray.shape[0]), int(gray.shape[1])
    except Exception:
        pass
    try:
        from src.engine.camera import camera_manager
        frame = camera_manager.get_latest_frame()
        if frame is not None:
            return int(frame.shape[0]), int(frame.shape[1])
    except Exception:
        pass
    return None


def _tag_pool_candidate(candidate: Mapping[str, Any], pool_name: str) -> dict[str, Any]:
    item = dict(candidate)
    provenance = item.get("provenance", [])
    if isinstance(provenance, str):
        provenance = [provenance]
    elif not isinstance(provenance, (list, tuple)):
        provenance = []
    provenance = [str(x) for x in provenance]
    if pool_name not in provenance:
        provenance.append(pool_name)
    item["provenance"] = provenance
    item.setdefault("source_name", pool_name)
    return item


def _candidate_pool_stats(scene: Any) -> dict[str, int]:
    runtime = getattr(scene, "runtime", None)
    names = (
        "_v28_all_hypotheses",
        "_v28_hypothesis_pool",
        "_v28_core_pool",
        "_v28_recall_baseline_pool",
        "latest_candidates",
    )
    out: dict[str, int] = {}
    for name in names:
        try:
            out[name] = len(list(getattr(runtime, name, []) or []))
        except Exception:
            out[name] = 0
    try:
        from src.engine.camera import hit_scanner
        out["hit_scanner.last_candidates"] = len(list(getattr(hit_scanner, "last_candidates", []) or []))
    except Exception:
        out["hit_scanner.last_candidates"] = 0
    return out


def _candidate_union(scene: Any, *, include_v28_recall: bool = True) -> list[Mapping[str, Any]]:
    """Build a GT-free training pool.

    V2.23.0 only captured the already-truncated final list. V2.8 already keeps
    the wider micro-hypothesis/recall pools on AIRuntime after rank_with_funnel.
    V2.23.1 captures those pools too. No GT coordinate participates in retention.
    """
    runtime = getattr(scene, "runtime", None)
    named_pools: list[tuple[str, list[Any]]] = []
    if include_v28_recall and runtime is not None:
        for name in (
            "_v28_all_hypotheses",
            "_v28_hypothesis_pool",
            "_v28_core_pool",
            "_v28_recall_baseline_pool",
            "_v28_actual_pool",
        ):
            try:
                value = list(getattr(runtime, name, []) or [])
            except Exception:
                value = []
            if value:
                named_pools.append((name, value))
    for name, value in (
        ("runtime.latest_candidates", getattr(runtime, "latest_candidates", []) if runtime is not None else []),
        ("scene.ranked_candidates", getattr(scene, "ranked_candidates", [])),
    ):
        try:
            value = list(value or [])
        except Exception:
            value = []
        if value:
            named_pools.append((name, value))
    try:
        from src.engine.camera import hit_scanner
        value = list(getattr(hit_scanner, "last_candidates", []) or [])
        if value:
            named_pools.append(("hit_scanner.last_candidates", value))
    except Exception:
        pass

    out: list[dict[str, Any]] = []
    by_xy: dict[tuple[int, int], dict[str, Any]] = {}
    for pool_name, pool in named_pools:
        for candidate in pool:
            if not isinstance(candidate, Mapping):
                continue
            try:
                x = float(candidate.get("camera_x", candidate.get("x", 0.0)))
                y = float(candidate.get("camera_y", candidate.get("y", 0.0)))
                key = (int(round(x * 2.0)), int(round(y * 2.0)))  # 0.5 px, GT-independent
            except Exception:
                key = (id(candidate), 0)
            existing = by_xy.get(key)
            if existing is None:
                item = _tag_pool_candidate(candidate, pool_name)
                by_xy[key] = item
                out.append(item)
            else:
                prov = existing.get("provenance", [])
                if isinstance(prov, str):
                    prov = [prov]
                prov = list(prov) if isinstance(prov, (list, tuple)) else []
                if pool_name not in prov:
                    prov.append(pool_name)
                existing["provenance"] = prov
    return out


def _project_screen(screen_xy: tuple[float, float]) -> tuple[float, float] | None:
    try:
        from src.engine.ai.space_mapper import project_screen_point
        p = project_screen_point(float(screen_xy[0]), float(screen_xy[1]))
        return float(p.camera_x), float(p.camera_y)
    except Exception:
        return None


def _source_kind(scene: Any) -> str:
    if bool(getattr(scene, "auto_training_enabled", False)):
        return "f2_projected" if bool(getattr(scene, "auto_headless", False)) else "f1_projected"
    if bool(getattr(scene, "single_synth_round_active", False)):
        return "single_projected"
    return "physical_manual"


def _capture_known_gt(scene: Any, candidates: list[Mapping[str, Any]]) -> bool:
    target = getattr(scene, "auto_target_screen_xy", None)
    if target is None and bool(getattr(scene, "single_synth_round_active", False)):
        target = getattr(scene, "single_target_screen_xy", None)
    if target is None:
        return False
    camera = _project_screen((float(target[0]), float(target[1])))
    if camera is None:
        return False
    cap: TrainingCaptureV223 | None = getattr(scene, "_v223_capture", None)
    if cap is None:
        return False
    try:
        shot_id = getattr(getattr(scene, "runtime", None), "_last_completed_shot_id", None)
        if shot_id is None:
            shot_id = getattr(getattr(scene, "runtime", None), "_active_shot_id", None)
        if shot_id is None:
            shot_id = f"round_{getattr(scene, 'current_round_id', cap.shots_saved + 1)}"
        pool_counts = _candidate_pool_stats(scene)
        cap.save_from_candidates(
            shot_id=shot_id,
            candidates=candidates,
            gt_camera_xy=camera,
            gt_screen_xy=(float(target[0]), float(target[1])),
            timestamp=float(getattr(getattr(scene, "runtime", None), "shot_timestamp", time.time()) or time.time()),
            background=_scene_background(scene),
            sampling_mode=str(getattr(getattr(scene, "runtime", None), "sampling_mode", "unknown")),
            frame_shape=_frame_shape(scene),
            source_kind=_source_kind(scene),
            metadata={
                "capture_phase": "shot_detected_known_gt",
                "authority": "shadow_only",
                "pool_contract": "v2231_v28_recall_union",
                "pool_counts": pool_counts,
                "v2232_framepack_expected": True,
            },
        )
        framepack = save_scene_framepack(
            scene,
            session_id=cap.session_id,
            shot_id=shot_id,
            sequence=cap.shots_saved,
            gt_camera_xy=camera,
            gt_screen_xy=(float(target[0]), float(target[1])),
            current_candidates=candidates,
            source_kind=_source_kind(scene),
            background=_scene_background(scene),
            sampling_mode=str(getattr(getattr(scene, "runtime", None), "sampling_mode", "unknown")),
        )
        if framepack is not None:
            print(f"[V2.23.2 FRAMEPACK] saved {framepack}")
        nearest = min((math.hypot(float(c.get("camera_x", 0.0)) - camera[0], float(c.get("camera_y", 0.0)) - camera[1]) for c in candidates), default=float("inf"))
        print(
            "[V2.23.1 POOL] "
            f"shot={shot_id} union={len(candidates)} "
            f"v28_all={pool_counts.get('_v28_all_hypotheses', 0)} "
            f"v28_recall={pool_counts.get('_v28_hypothesis_pool', 0)} "
            f"nearest={nearest:.1f}px oracle20={nearest <= 20.0} oracle42={nearest <= 42.0}"
        )
        setattr(scene, "_v223_last_captured_token", (cap.shots_saved, tuple(target)))
        return True
    except Exception as exc:
        print(f"[V2.23 CAPTURE] failed open (known GT): {type(exc).__name__}: {exc}")
        return False


def _shadow_score(scene: Any, candidates: list[Mapping[str, Any]], gt_camera: tuple[float, float] | None) -> None:
    try:
        model = load_champion_model()
        if model is None or not candidates:
            return
        # Build a transient record through the same schema, but never mutate live order.
        from .schema import ShotTrainingRecord, candidate_rows_from_pool
        gx, gy = gt_camera if gt_camera is not None else (0.0, 0.0)
        rows = candidate_rows_from_pool(candidates, gt_camera_xy=(gx, gy), frame_shape=_frame_shape(scene))
        if not rows:
            return
        rec = ShotTrainingRecord(
            session_id="shadow", shot_id="shadow", source_kind="live_shadow", timestamp=time.time(),
            gt_camera_x=gx, gt_camera_y=gy, candidates=rows,
        )
        order = model.rank_indices(rec)
        if order.size == 0:
            return
        best = rows[int(order[0])]
        if gt_camera is not None:
            d = math.hypot(best.camera_x - gx, best.camera_y - gy)
            print(f"[V2.23 SHADOW] kind={model.kind} top1=({best.camera_x:.1f},{best.camera_y:.1f}) gt_dist={d:.1f}px candidates={len(rows)} authority=NO")
        else:
            print(f"[V2.23 SHADOW] kind={model.kind} top1=({best.camera_x:.1f},{best.camera_y:.1f}) candidates={len(rows)} authority=NO")
    except Exception as exc:
        print(f"[V2.23 SHADOW] failed open: {type(exc).__name__}: {exc}")


def install_v2230_training_pipeline() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from src.engine.scenes import ai_training as ai_training_module
        from src.engine.scenes.ai_training import AITrainingScene
    except Exception as exc:
        print(f"[V2.23] training pipeline unavailable; runtime unchanged: {exc}")
        return

    # V2.23.2: the historical function named center_bias was actually uniform.
    # Make it a real soft centre prior while retaining 25% uniform exploration.
    try:
        import random
        def _v2232_center_bias(vp, margin: int = 12):
            if random.random() < 0.25:
                x = random.randint(vp.left + margin, max(vp.left + margin, vp.right - margin))
                y = random.randint(vp.top + margin, max(vp.top + margin, vp.bottom - margin))
                return x, y
            cx = vp.centerx; cy = vp.centery
            sx = max(12.0, vp.w * 0.22); sy = max(12.0, vp.h * 0.22)
            lo_x, hi_x = vp.left + margin, max(vp.left + margin, vp.right - margin)
            lo_y, hi_y = vp.top + margin, max(vp.top + margin, vp.bottom - margin)
            for _ in range(12):
                x = int(round(random.gauss(cx, sx))); y = int(round(random.gauss(cy, sy)))
                if lo_x <= x <= hi_x and lo_y <= y <= hi_y:
                    return x, y
            return max(lo_x, min(hi_x, cx)), max(lo_y, min(hi_y, cy))
        ai_training_module.SAMPLING_MODES["center_bias"] = _v2232_center_bias
        ai_training_module.SAMPLING_MODES["center"] = _v2232_center_bias
    except Exception as exc:
        print(f"[V2.23.2] center-bias patch failed open: {type(exc).__name__}: {exc}")

    original_enter = AITrainingScene.on_enter
    original_exit = AITrainingScene.on_exit
    original_detected = AITrainingScene._on_shot_detected
    original_click = AITrainingScene._on_training_click
    original_toggle = AITrainingScene._toggle_auto_training
    original_report = AITrainingScene._build_auto_report

    def wrapped_enter(self):
        result = original_enter(self)
        try:
            cap = TrainingCaptureV223(
                source_kind="ai_training_scene",
                metadata={"background": _scene_background(self), "runtime_authority_changed": False},
            )
            self._v223_capture = cap
            self._v223_pending_candidates = None
            self._v223_f2_started = False
            print(f"[V2.23 CAPTURE] session={cap.session_id} ready")
        except Exception as exc:
            self._v223_capture = None
            print(f"[V2.23 CAPTURE] unavailable; legacy training continues: {type(exc).__name__}: {exc}")
        return result

    def wrapped_exit(self):
        try:
            cap = getattr(self, "_v223_capture", None)
            if cap is not None:
                cap.close()
        except Exception:
            pass
        return original_exit(self)

    def wrapped_toggle(self, headless: bool = False):
        was_running = bool(getattr(self, "auto_training_enabled", False))
        result = original_toggle(self, headless=headless)
        if not was_running and bool(getattr(self, "auto_training_enabled", False)):
            try:
                self._v223_f2_started = bool(headless)
                cap = getattr(self, "_v223_capture", None)
                if cap is not None:
                    cap.update_metadata(
                        mode="f2_headless" if headless else "f1_visual",
                        sampling_mode=str(getattr(getattr(self, "runtime", None), "sampling_mode", "unknown")),
                    )
            except Exception:
                pass
        return result

    def wrapped_detected(self):
        # Before legacy ranking, only keep the narrow current snapshot. V2.8's
        # recall attributes can still contain the previous shot at this point.
        pre_candidates = _candidate_union(self, include_v28_recall=False)
        result = original_detected(self)
        # rank_with_funnel has now populated the current shot's V2.8 pools.
        candidates = _candidate_union(self, include_v28_recall=True)
        if not candidates:
            candidates = pre_candidates
        # For F1/F2/right-click synthetic the GT already exists at shot time.
        captured = _capture_known_gt(self, candidates)
        if not captured:
            self._v223_pending_candidates = list(candidates)
            self._v223_pending_timestamp = time.time()
        target = getattr(self, "auto_target_screen_xy", None)
        gt = _project_screen(target) if target is not None else None
        _shadow_score(self, candidates, gt)
        return result

    def wrapped_click(self, screen_pos):
        # Manual physical click supplies GT. Save the *pre-click* candidate snapshot
        # before legacy SimpleAIMemory learns from the click.
        try:
            pending = getattr(self, "_v223_pending_candidates", None)
            cap = getattr(self, "_v223_capture", None)
            if cap is not None and pending is not None:
                camera = _project_screen((float(screen_pos[0]), float(screen_pos[1])))
                if camera is not None:
                    manual_shot_id = f"manual_{cap.shots_saved + 1}"
                    cap.save_from_candidates(
                        shot_id=manual_shot_id, candidates=pending,
                        gt_camera_xy=camera, gt_screen_xy=(float(screen_pos[0]), float(screen_pos[1])),
                        timestamp=float(getattr(self, "_v223_pending_timestamp", time.time())),
                        background=_scene_background(self),
                        sampling_mode=str(getattr(getattr(self, "runtime", None), "sampling_mode", "unknown")),
                        frame_shape=_frame_shape(self), source_kind="physical_manual",
                        metadata={
                            "capture_phase": "manual_gt_click",
                            "authority": "shadow_only",
                            "pool_contract": "v2231_v28_recall_union",
                            "pool_counts": _candidate_pool_stats(self),
                            "v2232_framepack_expected": True,
                        },
                    )
                    framepack = save_scene_framepack(
                        self,
                        session_id=cap.session_id,
                        shot_id=manual_shot_id,
                        sequence=cap.shots_saved,
                        gt_camera_xy=camera,
                        gt_screen_xy=(float(screen_pos[0]), float(screen_pos[1])),
                        current_candidates=pending,
                        source_kind="physical_manual",
                        background=_scene_background(self),
                        sampling_mode=str(getattr(getattr(self, "runtime", None), "sampling_mode", "unknown")),
                    )
                    print(f"[V2.23 CAPTURE] physical GT saved candidates={len(pending)} framepack={bool(framepack)}")
                self._v223_pending_candidates = None
        except Exception as exc:
            print(f"[V2.23 CAPTURE] failed open (manual GT): {type(exc).__name__}: {exc}")
        return original_click(self, screen_pos)

    def wrapped_report(self):
        result = original_report(self)
        try:
            cap = getattr(self, "_v223_capture", None)
            shots = int(getattr(cap, "shots_saved", 0) if cap is not None else 0)
            if bool(getattr(self, "_v223_f2_started", False)) and shots >= 10 and not bool(getattr(self, "_v223_autotrain_scheduled", False)):
                self._v223_autotrain_scheduled = True
                started = schedule_quick_autotrain_v223(trigger=f"f2:{getattr(cap, 'session_id', 'unknown')}")
                msg = f"V2.23: {shots} grupper sparade; challenger-träning {'startad' if started else 'redan aktiv'} (shadow)."
                try:
                    self.auto_report_lines.insert(-1, msg)
                except Exception:
                    pass
                print(f"[V2.23 AUTOTRAIN] F2 completed; captured={shots} scheduled={started}")
        except Exception as exc:
            print(f"[V2.23 AUTOTRAIN] scheduling failed open: {type(exc).__name__}: {exc}")
        return result

    AITrainingScene.on_enter = wrapped_enter
    AITrainingScene.on_exit = wrapped_exit
    AITrainingScene._toggle_auto_training = wrapped_toggle
    AITrainingScene._on_shot_detected = wrapped_detected
    AITrainingScene._on_training_click = wrapped_click
    AITrainingScene._build_auto_report = wrapped_report
    AITrainingScene._v223_pipeline_installed = True
    _INSTALLED = True
    print("[V2.23.2] proposal/data/domain training pipeline installed (framepacks + fresh-F2 domain gate; live authority unchanged)")
