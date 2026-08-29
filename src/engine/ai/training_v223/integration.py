from __future__ import annotations

import math
import threading
import time
from typing import Any, Mapping

from .capture import TrainingCaptureV223
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


def _candidate_union(scene: Any) -> list[Mapping[str, Any]]:
    pools: list[list[Any]] = []
    for value in (
        getattr(getattr(scene, "runtime", None), "latest_candidates", []),
        getattr(scene, "ranked_candidates", []),
    ):
        try:
            pools.append(list(value))
        except Exception:
            pass
    try:
        from src.engine.camera import hit_scanner
        pools.append(list(getattr(hit_scanner, "last_candidates", [])))
    except Exception:
        pass
    out: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    for pool in pools:
        for candidate in pool:
            if not isinstance(candidate, Mapping):
                continue
            ident = id(candidate)
            if ident in seen:
                continue
            seen.add(ident)
            out.append(candidate)
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
            metadata={"capture_phase": "shot_detected_known_gt", "authority": "shadow_only"},
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
        from src.engine.scenes.ai_training import AITrainingScene
    except Exception as exc:
        print(f"[V2.23] training pipeline unavailable; runtime unchanged: {exc}")
        return

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
        pre_candidates = _candidate_union(self)
        result = original_detected(self)
        candidates = _candidate_union(self)
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
                    cap.save_from_candidates(
                        shot_id=f"manual_{cap.shots_saved + 1}", candidates=pending,
                        gt_camera_xy=camera, gt_screen_xy=(float(screen_pos[0]), float(screen_pos[1])),
                        timestamp=float(getattr(self, "_v223_pending_timestamp", time.time())),
                        background=_scene_background(self),
                        sampling_mode=str(getattr(getattr(self, "runtime", None), "sampling_mode", "unknown")),
                        frame_shape=_frame_shape(self), source_kind="physical_manual",
                        metadata={"capture_phase": "manual_gt_click", "authority": "shadow_only"},
                    )
                    print(f"[V2.23 CAPTURE] physical GT saved candidates={len(pending)}")
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
    print("[V2.23.0] unified training/model pipeline installed (capture + shadow champion/challenger; live authority unchanged)")
