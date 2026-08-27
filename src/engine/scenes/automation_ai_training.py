from __future__ import annotations

import pygame

from src.engine.events.event_bus import event_bus
from src.engine.scenes.ai_training import AITrainingScene


class AutomationAITrainingScene(AITrainingScene):
    """
    AITrainingScene with automation lifecycle events.

    The normal AITrainingScene remains untouched. This subclass is only used
    when an external automation command starts a training session.
    """

    EVENT_SOURCE = "AutomationAITrainingScene"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._automation_calibration_finished_emitted = False
        self._automation_waiting_emitted = False
        self._automation_completed_emitted = False
        self._automation_last_iteration = 0

    def on_enter(self) -> None:
        super().on_enter()

        event_bus.emit(
            "aiTraining.started",
            {
                "background": self.MODE_NAMES[self.bg_mode_index],
                "background_number": self.bg_mode_index + 1,
                "target_iterations": int(self.auto_target_iterations),
            },
            source=self.EVENT_SOURCE,
        )

        if self._auto_cal_phase is not None:
            event_bus.emit(
                "aiTraining.calibrationStarted",
                {
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
                "background": self.MODE_NAMES[self.bg_mode_index],
                "training_running": bool(self.auto_training_enabled),
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
            event_bus.emit(
                "aiTraining.trainingStarted",
                {
                    "mode": "headless",
                    "background": self.MODE_NAMES[self.bg_mode_index],
                    "target_iterations": int(self.auto_target_iterations),
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
                    "background": self.MODE_NAMES[self.bg_mode_index],
                    "iteration": int(self.auto_iteration),
                },
                source=self.EVENT_SOURCE,
            )

        return result

    def update(self, dt: float):
        previous_calibration_phase = self._auto_cal_phase
        previous_iteration = int(self.auto_iteration)

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
        records = list(self.round_records)
        total = len(records)
        found = sum(1 for record in records if record.found)
        top1 = sum(1 for record in records if record.top1_correct)
        top3 = sum(1 for record in records if record.top3_correct)
        ai_correct = sum(1 for record in records if record.ai_guess_correct)

        event_bus.emit(
            "aiTraining.completed",
            {
                "background": self.MODE_NAMES[self.bg_mode_index],
                "background_number": self.bg_mode_index + 1,
                "iterations": total,
                "target_iterations": int(self.auto_target_iterations),
                "found": found,
                "top1": top1,
                "top3": top3,
                "ai_guess_correct": ai_correct,
                "report": list(self.auto_report_lines),
            },
            source=self.EVENT_SOURCE,
        )

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
        failed = "misslyck" in lower_result or "failed" in lower_result or "avbruten" in lower_result

        if failed:
            event_bus.emit(
                "aiTraining.calibrationFailed",
                {
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
                    "background": self.MODE_NAMES[self.bg_mode_index],
                    "message": "AI training scene is ready for a shot or F2.",
                },
                source=self.EVENT_SOURCE,
            )
