from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import pygame

from src.engine.events.event_bus import event_bus
from src.engine.scenes.ai_training import AITrainingScene


AUTOMATION_RESULT_SCHEMA_VERSION = "1.2"
BENCHMARK_CONTROL_PATH = Path("content/ai/benchmark_control.json")
V221_CAPTURE_CONTROL_PATH = Path("content/ai/v221_capture_control.json")


class AutomationAITrainingScene(AITrainingScene):
    """AITrainingScene with machine-readable automation lifecycle events.

    V2.16 adds *capture only* candidate packs to the F2 automation path.  Normal
    AITrainingScene/game scenes remain untouched.  Capture failure is never
    allowed to break or alter the underlying training/detector path.
    """

    EVENT_SOURCE = "AutomationAITrainingScene"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._automation_calibration_finished_emitted = False
        self._automation_waiting_emitted = False
        self._automation_completed_emitted = False
        self._automation_last_iteration = 0
        self._automation_scene_started_ts = 0.0
        self._automation_training_started_ts = 0.0
        self._automation_benchmark_seed = None
        self._v216_candidate_recorder = None
        self._v216_last_capture: dict[str, Any] | None = None
        # V2.17: snapshot existing session-local known holes as capture provenance.
        self._v217_known_holes_before_round: list[dict[str, Any]] = []
        # V2.21.1: one-shot automation control for short, capture-only F2 runs.
        # This is deliberately scoped to AutomationAITrainingScene and never
        # changes normal game hit authority.
        self._v221_capture_control: dict[str, Any] = {}
        self._v221_restore_benchmark_mode: bool | None = None

    def on_enter(self) -> None:
        self._automation_scene_started_ts = time.time()
        super().on_enter()
        self._apply_v221_capture_control()
        event_bus.emit(
            "aiTraining.started",
            {
                "schema_version": AUTOMATION_RESULT_SCHEMA_VERSION,
                "background": self.MODE_NAMES[self.bg_mode_index],
                "background_number": self.bg_mode_index + 1,
                "scene_started_ts": self._automation_scene_started_ts,
            },
            source=self.EVENT_SOURCE,
        )
        if self._auto_cal_phase is not None:
            event_bus.emit(
                "aiTraining.calibrationStarted",
                {
                    "schema_version": AUTOMATION_RESULT_SCHEMA_VERSION,
                    "phase": self._auto_cal_phase,
                    "background": self.MODE_NAMES[self.bg_mode_index],
                },
                source=self.EVENT_SOURCE,
            )
        else:
            self._emit_calibration_finished_and_waiting()

    def on_exit(self) -> None:
        capture = self._finalize_v216_capture()
        self._restore_v221_capture_mode()
        event_bus.emit(
            "aiTraining.exited",
            {
                "schema_version": AUTOMATION_RESULT_SCHEMA_VERSION,
                "background": self.MODE_NAMES[self.bg_mode_index],
                "training_running": bool(self.auto_training_enabled),
                "iteration": int(self.auto_iteration),
                "candidate_capture_v216": capture,
            },
            source=self.EVENT_SOURCE,
        )
        super().on_exit()

    def handle_event(self, event: pygame.event.Event):
        was_running = bool(self.auto_training_enabled)
        # Deterministic benchmark seeding happens immediately BEFORE the real
        # F2 path starts generating synthetic rounds. Normal/manual training is
        # unaffected when benchmark_control.json is disabled or absent.
        if (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_F2
            and not was_running
        ):
            self._apply_benchmark_seed_if_requested()

        result = super().handle_event(event)
        is_running = bool(self.auto_training_enabled)
        if (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_F2
            and not was_running
            and is_running
        ):
            self._automation_training_started_ts = time.time()
            capture = self._start_v216_capture_session()
            event_bus.emit(
                "aiTraining.trainingStarted",
                {
                    "schema_version": AUTOMATION_RESULT_SCHEMA_VERSION,
                    "mode": "headless",
                    "background": self.MODE_NAMES[self.bg_mode_index],
                    "background_number": self.bg_mode_index + 1,
                    "target_iterations": int(self.auto_target_iterations),
                    "sampling_mode": str(self.runtime.sampling_mode),
                    "training_started_ts": self._automation_training_started_ts,
                    "benchmark_seed": getattr(self, "_automation_benchmark_seed", None),
                    "candidate_capture_v216": capture,
                    "capture_control_v221": dict(self._v221_capture_control),
                },
                source=self.EVENT_SOURCE,
            )
        elif (
            event.type == pygame.KEYDOWN
            and event.key in (pygame.K_F1, pygame.K_F2)
            and was_running
            and not is_running
            and not self.auto_report_visible
        ):
            capture = self._finalize_v216_capture()
            self._restore_v221_capture_mode()
            event_bus.emit(
                "aiTraining.trainingStopped",
                {
                    "schema_version": AUTOMATION_RESULT_SCHEMA_VERSION,
                    "background": self.MODE_NAMES[self.bg_mode_index],
                    "iteration": int(self.auto_iteration),
                    "candidate_capture_v216": capture,
                    "capture_control_v221": dict(self._v221_capture_control),
                },
                source=self.EVENT_SOURCE,
            )
        return result

    def _apply_v221_capture_control(self) -> None:
        """Apply a one-shot, short F2 capture request written by automation.

        The control is consumed immediately when the automation training scene
        opens, so a stale file cannot silently affect a later manual run.
        ``freeze_learning`` temporarily enables the runtime benchmark flag; the
        previous value is restored at completion/stop/exit.
        """
        self._v221_capture_control = {}
        try:
            if not V221_CAPTURE_CONTROL_PATH.exists():
                return
            payload = json.loads(V221_CAPTURE_CONTROL_PATH.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not bool(payload.get("enabled", False)):
                return
            now = time.time()
            expires_at = float(payload.get("expires_at", 0.0) or 0.0)
            if expires_at > 0.0 and now > expires_at:
                payload["enabled"] = False
                payload["expired_at"] = now
                V221_CAPTURE_CONTROL_PATH.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print("[V2.21.1 CAPTURE] Ignoring expired capture control")
                return

            shots = max(1, min(100, int(payload.get("shots", 30))))
            freeze_learning = bool(payload.get("freeze_learning", True))
            self.auto_target_iterations = shots
            if freeze_learning:
                self._v221_restore_benchmark_mode = bool(
                    self.runtime.settings.get("benchmark_mode", False)
                )
                self.runtime.settings["benchmark_mode"] = True

            token = str(payload.get("token", ""))
            self._v221_capture_control = {
                "active": True,
                "token": token,
                "shots": shots,
                "freeze_learning": freeze_learning,
                "purpose": str(payload.get("purpose", "v221_fullframe_direct")),
            }

            # Consume once. A crash after this point must not poison a future
            # unrelated F2/manual session.
            payload["enabled"] = False
            payload["consumed_at"] = now
            V221_CAPTURE_CONTROL_PATH.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                "[V2.21.1 CAPTURE] armed "
                f"shots={shots} freeze_learning={freeze_learning} token={token}"
            )
        except Exception as exc:
            self._v221_capture_control = {"active": False, "error": str(exc)}
            print(f"[V2.21.1 CAPTURE] control ignored: {exc}")

    def _restore_v221_capture_mode(self) -> None:
        previous = self._v221_restore_benchmark_mode
        if previous is None:
            return
        try:
            self.runtime.settings["benchmark_mode"] = bool(previous)
        except Exception:
            pass
        self._v221_restore_benchmark_mode = None

    def _apply_benchmark_seed_if_requested(self) -> None:
        self._automation_benchmark_seed = None
        try:
            if not BENCHMARK_CONTROL_PATH.exists():
                return
            payload = json.loads(BENCHMARK_CONTROL_PATH.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not bool(payload.get("enabled", False)):
                return
            seed = int(payload.get("seed"))
            random.seed(seed)
            try:
                import numpy as np

                np.random.seed(seed & 0xFFFFFFFF)
            except Exception:
                pass
            # SyntheticHoleOverlay already uses a fixed deterministic RNG per
            # scene. If it has been constructed, re-seed it too so the complete
            # hole sequence is explicitly tied to this run seed.
            overlay = getattr(self, "synthetic_overlay", None)
            rng = getattr(overlay, "_rng", None)
            if rng is not None and hasattr(rng, "seed"):
                rng.seed(seed ^ 0x5A17C0DE)
            self._automation_benchmark_seed = seed
            print(f"[BENCHMARK] deterministic seed={seed}")
        except Exception as exc:
            print(f"[BENCHMARK] seed setup failed: {exc}")

    # ------------------------------------------------------------------
    # V2.16 automation-only candidate capture
    # ------------------------------------------------------------------
    def _start_v216_capture_session(self) -> dict[str, Any]:
        self._finalize_v216_capture()
        try:
            from src.engine.offline.candidate_pack_v216 import (
                CandidateCaptureConfigV216,
                CandidateShadowRecorderV216,
            )

            config = CandidateCaptureConfigV216.load()
            self._v216_candidate_recorder = CandidateShadowRecorderV216(
                config,
                background=self.MODE_NAMES[self.bg_mode_index],
                benchmark_seed=getattr(self, "_automation_benchmark_seed", None),
                sampling_mode=str(self.runtime.sampling_mode),
            )
            summary = self._v216_candidate_recorder.summary()
            if summary.get("enabled"):
                print(
                    "[V2.16 CAPTURE] session="
                    f"{summary.get('session_id')} patch={summary.get('patch_size')} "
                    f"post_frames={summary.get('max_post_frames')} cap={summary.get('max_candidates')} "
                    f"full_frames={summary.get('save_full_frames')} full_post={summary.get('full_frame_post_count')}"
                )
            return summary
        except Exception as exc:
            self._v216_candidate_recorder = None
            print(f"[V2.16 CAPTURE] unavailable: {exc}")
            return {"enabled": False, "error": str(exc), "shadow_only": True}

    def _finalize_v216_capture(self) -> dict[str, Any]:
        recorder = getattr(self, "_v216_candidate_recorder", None)
        if recorder is None:
            return {"enabled": False, "shadow_only": True}
        try:
            summary = dict(recorder.finalize())
        except Exception as exc:
            summary = {"enabled": True, "shadow_only": True, "error": str(exc)}
        return summary

    @staticmethod
    def _copy_v216_frame(frame):
        try:
            return None if frame is None else frame.copy()
        except Exception:
            return frame

    @staticmethod
    def _resolve_v217_recent_pre_frame(peak_ts: float):
        """Return a true camera frame shortly before the shot/audio peak.

        This is capture-only. The current live HitScanner keeps using its own
        legacy/reference pre-shot semantics. Future NEW-hole learning can use
        this recent camera frame on moving game/video backgrounds.
        """
        try:
            from src.engine.camera.hit_scanner import hit_scanner
            peak = float(peak_ts)
            if peak <= 0.0:
                return None, 0.0
            lead = max(0.06, float(getattr(hit_scanner, "association_lead_s", 0.08)))
            latest_allowed = peak - lead
            target = latest_allowed - 0.03
            best = None
            best_delta = float("inf")
            for item in list(getattr(hit_scanner, "frame_history", []) or []):
                ts = float(getattr(item, "timestamp", 0.0) or 0.0)
                if ts <= 0.0 or ts > latest_allowed:
                    continue
                delta = abs(ts - target)
                if delta < best_delta:
                    best = item
                    best_delta = delta
            if best is None:
                return None, 0.0
            gray = getattr(best, "gray", None)
            return (None if gray is None else gray.copy()), float(best.timestamp)
        except Exception:
            return None, 0.0

    def _start_auto_iteration(self, screen: pygame.Surface) -> None:
        """Capture existing HitScanner known holes before each synthetic shot.

        The registry is soft/session-local context only; this never changes live
        candidate scoring or authority.
        """
        try:
            from src.engine.camera.hit_scanner import hit_scanner
            self._v217_known_holes_before_round = [dict(h) for h in list(hit_scanner.known_holes)]
        except Exception:
            self._v217_known_holes_before_round = []
        super()._start_auto_iteration(screen)

    def _snapshot_v216_shot(self) -> dict[str, Any] | None:
        recorder = getattr(self, "_v216_candidate_recorder", None)
        if recorder is None or not bool(getattr(recorder, "enabled", False)):
            return None
        if not self.auto_training_enabled or self.auto_target_screen_xy is None:
            return None
        try:
            from src.engine.ai.space_mapper import project_screen_point
            from src.engine.camera.hit_scanner import hit_scanner

            raw = list(self.runtime.latest_candidates)
            if not raw:
                raw = list(hit_scanner.last_candidates)
            pre = self._copy_v216_frame(getattr(self.runtime, "pre_shot_gray", None))
            shot_ts = float(getattr(self.runtime, "_shot_ts", 0.0) or 0.0)
            if shot_ts <= 0.0:
                # Defensive fallback for capture timing: the F2 synthetic path
                # sets the scanner event before revealing the projected hole.
                # Prefer the newest pending/recent scanner peak if the legacy
                # runtime mirror has not synchronized yet.
                shot_ts = max(
                    (float(getattr(ev, "peak_ts", 0.0) or 0.0) for ev in list(getattr(hit_scanner, "audio_events", []) or [])),
                    default=0.0,
                )
            recent_pre, recent_pre_ts = self._resolve_v217_recent_pre_frame(shot_ts)
            post = self._copy_v216_frame(getattr(self.runtime, "post_shot_gray", None))
            post_frames = []
            for item in list(getattr(self.runtime, "_post_shot_frames", []) or []):
                if isinstance(item, (tuple, list)) and item:
                    frame = self._copy_v216_frame(item[0])
                    ts = float(item[1]) if len(item) > 1 else 0.0
                    post_frames.append((frame, ts))
            sx, sy = float(self.auto_target_screen_xy[0]), float(self.auto_target_screen_xy[1])
            projected = project_screen_point(sx, sy)
            return {
                "raw_candidates": raw,
                "pre_gray": pre,
                "recent_pre_gray": recent_pre,
                "recent_pre_timestamp": float(recent_pre_ts),
                "shot_timestamp": float(shot_ts),
                "post_gray": post,
                "post_frames": post_frames,
                "gt_camera_xy": (float(projected.camera_x), float(projected.camera_y)),
                "gt_screen_xy": (sx, sy),
                "match_radius_px": float(self.runtime.settings.get("click_match_radius_px", 42.0)),
                "known_holes_before_shot": [dict(h) for h in self._v217_known_holes_before_round],
            }
        except Exception as exc:
            print(f"[V2.16 CAPTURE] shot snapshot failed: {exc}")
            return None

    def _save_v216_shot(self, snapshot: dict[str, Any] | None, ranked_candidates: list[dict[str, Any]], round_id: int) -> None:
        if snapshot is None:
            return
        recorder = getattr(self, "_v216_candidate_recorder", None)
        if recorder is None:
            return
        try:
            result = recorder.capture_shot(
                round_id=int(round_id),
                raw_candidates=snapshot["raw_candidates"],
                ranked_candidates=list(ranked_candidates),
                pre_gray=snapshot["pre_gray"],
                recent_pre_gray=snapshot.get("recent_pre_gray"),
                recent_pre_timestamp=snapshot.get("recent_pre_timestamp"),
                post_gray=snapshot["post_gray"],
                post_frames=snapshot["post_frames"],
                gt_camera_xy=snapshot["gt_camera_xy"],
                gt_screen_xy=snapshot["gt_screen_xy"],
                match_radius_px=float(snapshot["match_radius_px"]),
                extra_metadata={
                    "auto_iteration_after_detection": int(self.auto_iteration),
                    "background_number": int(self.bg_mode_index + 1),
                    "source": "AutomationAITrainingScene/F2",
                    "known_holes_before_shot": snapshot.get("known_holes_before_shot", []),
                    "known_hole_registry_semantics": "session-local accepted HitScanner holes; incomplete for physical holes present before scene start",
                    "recent_pre_timestamp": snapshot.get("recent_pre_timestamp"),
                    "shot_timestamp": snapshot.get("shot_timestamp"),
                    "recent_pre_semantics": "true camera frame before audio peak for temporal/new-hole learning; capture-only, does not alter live HitScanner pre-shot/reference behaviour",
                },
            )
            self._v216_last_capture = dict(result)
            if not result.get("saved", False):
                print(f"[V2.16 CAPTURE] shot {round_id} not saved: {result}")
        except Exception as exc:
            print(f"[V2.16 CAPTURE] shot {round_id} failed: {exc}")

    def _on_shot_detected(self) -> None:
        """Wrap the unmodified detector/training path with read-only capture."""
        snapshot = self._snapshot_v216_shot()
        # If there are no raw candidates the base method can immediately finish
        # the final round and build the report. Capture first so the last miss is
        # included in the V2.16 session summary.
        no_candidates = snapshot is not None and not snapshot.get("raw_candidates")
        if no_candidates:
            self._save_v216_shot(snapshot, [], int(self.current_round_id) + 1)
            snapshot = None

        super()._on_shot_detected()

        if snapshot is not None:
            self._save_v216_shot(snapshot, list(self.ranked_candidates), int(self.current_round_id))

    def update(self, dt: float):
        previous_calibration_phase = self._auto_cal_phase
        result = super().update(dt)
        if (
            previous_calibration_phase is not None
            and self._auto_cal_phase is None
            and not self._automation_calibration_finished_emitted
        ):
            self._emit_calibration_finished_and_waiting()
        current_iteration = int(self.auto_iteration)
        if current_iteration > self._automation_last_iteration:
            self._emit_missing_iteration_events(current_iteration)
        return result

    def _build_auto_report(self) -> None:
        super()._build_auto_report()
        if self._automation_completed_emitted:
            return
        # A run can build its report in the same update that increments the
        # final iteration. Emit any missing progress event before completed.
        self._emit_missing_iteration_events(int(self.auto_iteration))

        capture_summary = self._finalize_v216_capture()
        if bool(capture_summary.get("enabled")):
            try:
                self.auto_report_lines += [
                    "",
                    "--- V2.16 candidate shadow capture ---",
                    f"Session: {capture_summary.get('session_id')}",
                    f"Saved shot packs: {capture_summary.get('shots_saved', 0)}",
                    f"Capture errors: {capture_summary.get('capture_errors', 0)}",
                    "Shadow only: yes",
                ]
            except Exception:
                pass

        self._automation_completed_emitted = True
        completed_ts = time.time()
        records = list(self.round_records)
        total = len(records)
        found = sum(1 for record in records if record.found)
        top1 = sum(1 for record in records if record.top1_correct)
        top3 = sum(1 for record in records if record.top3_correct)
        ai_correct = sum(1 for record in records if record.ai_guess_correct)
        missed = total - found
        nearest_distances = [
            float(record.nearest_dist)
            for record in records
            if float(record.nearest_dist) < 9000.0
        ]
        ai_distances = [
            float(record.ai_guess_dist_to_gt)
            for record in records
            if record.candidate_count_ranked > 0
            and float(record.ai_guess_dist_to_gt) < 9000.0
        ]
        raw_counts = [int(record.candidate_count_raw) for record in records]
        ranked_counts = [int(record.candidate_count_ranked) for record in records]
        funnel_summary = dict(self.runtime.funnel.summary())
        funnel_count = len(self.runtime.funnel.shots)

        match_radius_px = 0.0
        if records:
            match_radius_px = float(records[0].match_radius_px)
        else:
            try:
                match_radius_px = float(
                    self.runtime.settings.get("click_match_radius_px", 42.0)
                )
            except Exception:
                match_radius_px = 42.0
        training_duration = None
        if self._automation_training_started_ts > 0.0:
            training_duration = round(
                completed_ts - self._automation_training_started_ts,
                3,
            )

        scene_duration = None
        if self._automation_scene_started_ts > 0.0:
            scene_duration = round(
                completed_ts - self._automation_scene_started_ts,
                3,
            )
        metrics = {
            "iterations": total,
            "target_iterations": int(self.auto_target_iterations),
            "found": found,
            "found_pct": self._pct(found, total),
            "missed": missed,
            "missed_pct": self._pct(missed, total),
            "top1": top1,
            "top1_pct": self._pct(top1, total),
            "top3": top3,
            "top3_pct": self._pct(top3, total),
            "ai_guess_correct": ai_correct,
            "ai_guess_correct_pct": self._pct(ai_correct, total),
            "nearest_distance_px": self._distance_stats(nearest_distances),
            "ai_guess_distance_px": self._distance_stats(ai_distances),
            "candidates_raw": self._count_stats(raw_counts),
            "candidates_ranked": self._count_stats(ranked_counts),
        }
        detector_diagnostic_flush = {
            "finalized": 0,
            "missing_evaluation": 0,
        }
        try:
            from src.engine.camera.hit_scanner import HitScanner, hit_scanner

            detector_engine = getattr(HitScanner, "_candidate_generator_v2_engine", None)
            force_finalize = getattr(
                detector_engine,
                "force_finalize_benchmark_diagnostics",
                None,
            )
            if callable(force_finalize):
                detector_diagnostic_flush = dict(force_finalize(hit_scanner))
        except Exception as exc:
            detector_diagnostic_flush = {
                "finalized": 0,
                "missing_evaluation": 0,
                "error": str(exc),
            }
        # The capture-only run is finished; restore the prior runtime mode
        # before publishing completion so later work cannot inherit it.
        self._restore_v221_capture_mode()

        consistency = {
            "current_round_id": int(self.current_round_id),
            "round_record_count": total,
            "funnel_shot_count": funnel_count,
            "counts_match": bool(
                int(self.current_round_id) == total == funnel_count
            ),
            "detector_diagnostic_flush": detector_diagnostic_flush,
            "candidate_capture_v216": capture_summary,
        }
        event_bus.emit(
            "aiTraining.completed",
            {
                "schema_version": AUTOMATION_RESULT_SCHEMA_VERSION,
                "background": self.MODE_NAMES[self.bg_mode_index],
                "background_number": self.bg_mode_index + 1,
                "sampling_mode": str(self.runtime.sampling_mode),
                "match_radius_px": match_radius_px,
                "mode": "headless" if self.auto_headless else "visual",
                "completed_ts": completed_ts,
                "benchmark_seed": getattr(self, "_automation_benchmark_seed", None),
                "training_duration_seconds": training_duration,
                "scene_duration_seconds": scene_duration,
                # Compatibility fields kept at top level.
                "iterations": total,
                "target_iterations": int(self.auto_target_iterations),
                "found": found,
                "top1": top1,
                "top3": top3,
                "ai_guess_correct": ai_correct,
                # Preferred machine-readable data.
                "metrics": metrics,
                "funnel": funnel_summary,
                "consistency": consistency,
                "candidate_capture_v216": capture_summary,
                "capture_control_v221": dict(self._v221_capture_control),
                "round_records": [record.to_csv_dict() for record in records],
                # Human-readable report remains useful for people but must not
                # be parsed by automation when structured fields exist.
                "report": list(self.auto_report_lines),
            },
            source=self.EVENT_SOURCE,
        )

    @staticmethod
    def _pct(value: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return round(100.0 * float(value) / float(total), 3)

    @staticmethod
    def _distance_stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {
                "count": 0,
                "avg": None,
                "median": None,
                "min": None,
                "max": None,
            }
        return {
            "count": len(values),
            "avg": round(sum(values) / len(values), 3),
            "median": round(float(statistics.median(values)), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }

    @staticmethod
    def _count_stats(values: list[int]) -> dict[str, Any]:
        if not values:
            return {
                "count": 0,
                "avg": 0.0,
                "min": 0,
                "max": 0,
                "zero_count": 0,
                "over_50_count": 0,
                "over_100_count": 0,
                "over_200_count": 0,
            }
        return {
            "count": len(values),
            "avg": round(sum(values) / len(values), 3),
            "min": min(values),
            "max": max(values),
            "zero_count": sum(1 for value in values if value == 0),
            "over_50_count": sum(1 for value in values if value > 50),
            "over_100_count": sum(1 for value in values if value > 100),
            "over_200_count": sum(1 for value in values if value > 200),
        }

    def _emit_missing_iteration_events(self, current_iteration: int) -> None:
        if current_iteration <= self._automation_last_iteration:
            return
        for iteration in range(
            self._automation_last_iteration + 1,
            current_iteration + 1,
        ):
            event_bus.emit(
                "aiTraining.iterationCompleted",
                {
                    "schema_version": AUTOMATION_RESULT_SCHEMA_VERSION,
                    "iteration": iteration,
                    "target_iterations": int(self.auto_target_iterations),
                    "background": self.MODE_NAMES[self.bg_mode_index],
                },
                source=self.EVENT_SOURCE,
            )
        self._automation_last_iteration = current_iteration

    def _emit_calibration_finished_and_waiting(self) -> None:
        if self._automation_calibration_finished_emitted:
            return

        self._automation_calibration_finished_emitted = True
        result_text = str(self._auto_cal_result or "")
        lower_result = result_text.lower()
        failed = (
            "misslyck" in lower_result
            or "failed" in lower_result
            or "avbruten" in lower_result
        )
        if failed:
            event_bus.emit(
                "aiTraining.calibrationFailed",
                {
                    "schema_version": AUTOMATION_RESULT_SCHEMA_VERSION,
                    "result": result_text,
                    "attempts": int(self._auto_cal_attempts),
                    "background": self.MODE_NAMES[self.bg_mode_index],
                },
                source=self.EVENT_SOURCE,
            )
            return
        event_bus.emit(
            "aiTraining.calibrationDone",
            {
                "schema_version": AUTOMATION_RESULT_SCHEMA_VERSION,
                "result": result_text,
                "attempts": int(self._auto_cal_attempts),
                "background": self.MODE_NAMES[self.bg_mode_index],
            },
            source=self.EVENT_SOURCE,
        )
        if not self._automation_waiting_emitted:
            self._automation_waiting_emitted = True
            event_bus.emit(
                "aiTraining.waitingForFirstShot",
                {
                    "schema_version": AUTOMATION_RESULT_SCHEMA_VERSION,
                    "background": self.MODE_NAMES[self.bg_mode_index],
                    "message": "AI training scene is ready for a shot or F2.",
                },
                source=self.EVENT_SOURCE,
            )
