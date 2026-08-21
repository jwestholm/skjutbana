from __future__ import annotations

import statistics
import time
from typing import Any

import pygame

from src.engine.events.event_bus import event_bus
from src.engine.scenes.ai_training import AITrainingScene


AUTOMATION_RESULT_SCHEMA_VERSION = "1.0"


class AutomationAITrainingScene(AITrainingScene):
    """
    AITrainingScene with automation lifecycle events.

    The normal AITrainingScene remains untouched. This subclass is only used
    when an external automation command starts a training session.

    Events are deliberately machine-readable. ``aiTraining.completed`` contains
    both compact aggregate metrics and the complete RoundRecord dataset so a
    future AI/automation process never has to parse the human report text.
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

    def on_enter(self) -> None:
        self._automation_scene_started_ts = time.time()
        super().on_enter()

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
        event_bus.emit(
            "aiTraining.exited",
            {
                "schema_version": AUTOMATION_RESULT_SCHEMA_VERSION,
                "background": self.MODE_NAMES[self.bg_mode_index],
                "training_running": bool(self.auto_training_enabled),
                "iteration": int(self.auto_iteration),
            },
            source=self.EVENT_SOURCE,
        )
        super().on_exit()

    def handle_event(self, event: pygame.event.Event):
        was_running = bool(self.auto_training_enabled)
        result = super().handle_event(event)
        is_running = bool(self.auto_training_enabled)

        if (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_F2
            and not was_running
            and is_running
        ):
            self._automation_training_started_ts = time.time()
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
            event_bus.emit(
                "aiTraining.trainingStopped",
                {
                    "schema_version": AUTOMATION_RESULT_SCHEMA_VERSION,
                    "background": self.MODE_NAMES[self.bg_mode_index],
                    "iteration": int(self.auto_iteration),
                },
                source=self.EVENT_SOURCE,
            )

        return result

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

        consistency = {
            "current_round_id": int(self.current_round_id),
            "round_record_count": total,
            "funnel_shot_count": funnel_count,
            "counts_match": bool(
                int(self.current_round_id) == total == funnel_count
            ),
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
