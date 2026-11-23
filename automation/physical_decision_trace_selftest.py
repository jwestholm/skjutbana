"""Focused synthetic checks for opt-in physical decision trace v1."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.engine.camera.hit_scanner import AudioShotEvent, HitScanner
from src.engine.audio.audio_peak_detector import AudioPeakDetector
from src.engine.physical_trace import PhysicalTraceRecorder, _build_decision_trace_v1
from src.engine.shot_track_v2226 import update_tracks_frame_unique_v2226
from src.engine.track_audit import record_selection, record_selector_path
from src.engine.input.hit_input import hit_input


def candidate(x: float, *, sid: int = 7, **extra):
    return {
        "camera_x": float(x), "camera_y": 41.25, "score": 12.0,
        "timestamp": 100.125, "detector_v2": 1,
        "v2224_producer_shot_id": sid, "area": None, **extra,
    }


# Synthetic captures use controlled free space; live disk policy is unchanged.
from automation.trace_test_support import setUpModule, tearDownModule


class DecisionTraceTests(unittest.TestCase):
    def scanner(self, *, trace: bool):
        scanner = HitScanner()
        scanner.physical_trace_capture_enabled = trace
        scanner.physical_decision_trace_v1_enabled = trace
        scanner.last_trace_pipeline_shot_id = 7
        event = AudioShotEvent(7, 100.0, 100.0)
        scanner.audio_events.append(event)
        update_tracks_frame_unique_v2226(scanner, [candidate(123.5)], 100.125)
        track = scanner._best_track_for_event(event)
        return scanner, event, track

    def test_toggle_is_observational_and_identity_xy_stay_exact(self):
        results = []
        for enabled in (False, True):
            scanner, event, track = self.scanner(trace=enabled)
            if enabled:
                record_selection(scanner, event, track, "BASE")
                record_selector_path(scanner, event, owner="V252", branch="FAIL_OPEN_NO_FRESH_TRACK",
                                     reason="authority_fail_open_elapsed", fallback_to="base_physical_selector")
                record_selector_path(scanner, event, owner="V253", branch="DELEGATE_AFTER_FULL_RESCUE",
                                     reason="full_rescue_consumed", fallback_to="base_physical_selector")
                self.assertEqual(scanner._audit_selections[7]["selector_path_hops"][0]["owner"], "V252")
                self.assertEqual(scanner._audit_selections[7]["selector_path_hops"][1]["owner"], "V253")
            results.append((track.track_id, track.camera_x, track.camera_y, track.best_score))
        self.assertEqual(results[0], results[1])

    def test_opt_in_persists_referenced_candidate_ai_and_actual_emission(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PhysicalTraceRecorder(directory, enabled=True)
            recorder.decision_trace_enabled = True
            scanner, event, track = self.scanner(trace=True)
            record_selection(scanner, event, track, "V253")
            record_selector_path(scanner, event, owner="V252", branch="FAIL_OPEN_NO_FRESH_TRACK",
                                 reason="authority_fail_open_elapsed", fallback_to="base_physical_selector")
            record_selector_path(scanner, event, owner="V253", branch="DELEGATE_AFTER_FULL_RESCUE",
                                 reason="full_rescue_consumed", fallback_to="base_physical_selector")
            with patch.object(PhysicalTraceRecorder, "_rescue_state", return_value={
                    "state": "AVAILABLE", "requested": True, "consumed": True}):
                recorder.start_from_scanner(scanner, event, {})
                recorder.capture_decision(scanner, track, event, {})
                recorder.record_ai_decision(scanner, event, mode="advisory", decision={
                    "apply": False, "reason": "advisory_only", "camera_x": 900.0,
                    "camera_y": 901.0, "confidence": 0.77,
                })
                event.state = "matched"
                event.emitted = True
                event.matched_track_id = track.track_id
                scanner.last_best_candidate = {
                    "camera_x": 123.5, "camera_y": 41.25, "event": 7.0, "score": 12.0,
                }
                event.audio_trigger.setdefault("timing", {})[
                    "physical_decision_trace_push_camera_xy"] = [123.5, 41.25]
                recorder.record_emission_decision(scanner, event, track)
                recorder.finish(7, scanner, event)
            self.assertTrue(recorder.flush())

            trace = json.loads((Path(directory) / "shots/shot_00000007/trace.json").read_text())
            v1 = trace["decision_trace_v1"]
            self.assertEqual(v1["schema"], "physical-decision-trace-1")
            self.assertEqual(v1["ai"]["mode"], "advisory")
            self.assertFalse(v1["ai"]["apply"])
            self.assertEqual(v1["emission"]["camera_xy"]["value"], [123.5, 41.25])
            row = v1["candidate_ledger"]["tracks"][0]
            self.assertEqual(row["track_camera_xy"]["value"], [123.5, 41.25])
            self.assertEqual(row["candidate_camera_xy"]["value"], [123.5, 41.25])
            self.assertEqual(row["track_id"]["value"], track.track_id)
            self.assertEqual(row["candidate_identity"]["value"], "7:1:0")
            self.assertEqual(v1["selector"]["selected_track_id"]["value"], track.track_id)
            self.assertEqual(row["source_frame_ids"]["state"], "UNRECORDED")
            self.assertEqual(row["evidence_field_states"]["area"]["state"], "MISSING")
            self.assertEqual(row["evidence_values_ref"]["state"], "AVAILABLE")
            hops = v1["selector"]["path_hops"]["value"]
            self.assertEqual([hop["owner"] for hop in hops], ["V252", "V253"])
            self.assertEqual(v1["fallback_rescue"]["consumed"], True)
            self.assertIn(v1["selector"]["owner"]["state"], ("AVAILABLE", "UNRECORDED"))
            recorder.shutdown()

    def test_emission_trace_uses_push_arguments_not_mutable_debug_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PhysicalTraceRecorder(directory, enabled=True)
            recorder.decision_trace_enabled = True
            scanner, event, track = self.scanner(trace=True)
            scanner.known_holes = [dict(hole_id=9, camera_x=100.0, camera_y=100.0,
                                        score=10.0, first_seen_ts=1.0,
                                        last_seen_ts=1.0, hit_count=1)]
            track.camera_x, track.camera_y = 106.0, 108.0
            track.best_score = 20.0
            with patch.object(PhysicalTraceRecorder, "_rescue_state", return_value={"state": "UNAVAILABLE"}):
                recorder.start_from_scanner(scanner, event, {})
                with patch.object(hit_input, "push_camera_hit") as push:
                    self.assertTrue(scanner._emit_track_result(track, event))
                    push.assert_called_once_with(100.0, 100.0)
                # A later event/debug update must not rewrite this event's XY.
                scanner.last_best_candidate = {"camera_x": 999.0, "camera_y": 998.0, "event": 8.0}
                recorder.record_emission_decision(scanner, event, track)
                recorder.finish(7, scanner, event)
                self.assertTrue(recorder.flush())
            trace = json.loads((Path(directory) / "shots/shot_00000007/trace.json").read_text())
            emission = trace["decision_trace_v1"]["emission"]
            self.assertEqual(emission["camera_xy"]["value"], [100.0, 100.0])
            self.assertEqual(emission["matched_hole"]["state"], "AVAILABLE")
            self.assertEqual(emission["matched_hole"]["value"], {"hole_id": 9, "camera_xy": [100.0, 100.0]})
            self.assertEqual(emission["candidate_camera_xy"]["value"], [106.0, 108.0])
            self.assertEqual(trace["outcome"]["final_camera_xy"], {"camera_x": 100.0, "camera_y": 100.0})
            self.assertEqual(trace["outcome"]["final_camera_xy_semantics"], "HitInput.push_camera_hit arguments")
            recorder.shutdown()

    def test_failed_push_is_not_reported_as_emitted_and_keeps_attempt_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PhysicalTraceRecorder(directory, enabled=True)
            recorder.decision_trace_enabled = True
            scanner, event, track = self.scanner(trace=True)
            recorder.start_from_scanner(scanner, event, {})
            with patch.object(hit_input, "push_camera_hit", side_effect=RuntimeError("synthetic push failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic push failure"):
                    scanner._emit_track_result(track, event)
            recorder.finish(7, scanner, event)
            self.assertTrue(recorder.flush())
            trace = json.loads((Path(directory) / "shots/shot_00000007/trace.json").read_text())
            emission = trace["decision_trace_v1"]["emission"]
            self.assertFalse(emission["emitted"])
            self.assertEqual(emission["camera_xy"]["state"], "NOT_APPLICABLE")
            self.assertEqual(emission["push_attempt"]["state"], "AVAILABLE")
            self.assertEqual(emission["push_attempt"]["camera_xy"], [123.5, 41.25])
            self.assertEqual(emission["push_attempt"]["call_state"], "RAISED")
            self.assertEqual(emission["push_attempt"]["exception_type"], "RuntimeError")
            self.assertFalse(trace["outcome"]["emitted"])
            recorder.shutdown()

    def test_bootstrap_emission_wrapper_records_real_push_boundary(self):
        from src.engine.ai import bootstrap
        from src.engine.ai import runtime as runtime_module
        from src.engine import physical_trace as trace_module

        original_emit = HitScanner._emit_track_result
        original_update = HitScanner.update
        runtime = SimpleNamespace(settings={"mode": "train_only"},
                                  observe_scanner=lambda *args, **kwargs: None,
                                  choose_for_emission=lambda x, y, **kwargs: {"apply": False},
                                  mark_shot_finished=lambda *args, **kwargs: None)
        with tempfile.TemporaryDirectory() as directory:
            recorder = PhysicalTraceRecorder(directory, enabled=True)
            recorder.decision_trace_enabled = True
            scanner, event, track = self.scanner(trace=True)
            scanner.known_holes = [dict(hole_id=9, camera_x=100.0, camera_y=100.0,
                                        score=10.0, first_seen_ts=1.0,
                                        last_seen_ts=1.0, hit_count=1)]
            track.camera_x, track.camera_y, track.best_score = 106.0, 108.0, 20.0
            recorder.start_from_scanner(scanner, event, {})
            with patch.object(trace_module, "get_physical_trace_recorder", return_value=recorder), \
                 patch.object(runtime_module, "get_ai_runtime", return_value=runtime):
                bootstrap._patch_hit_scanner()
                try:
                    with patch.object(hit_input, "push_camera_hit") as push:
                        self.assertTrue(scanner._emit_track_result(track, event))
                        push.assert_called_once_with(100.0, 100.0)
                finally:
                    HitScanner._emit_track_result = original_emit
                    HitScanner.update = original_update
            recorder.finish(7, scanner, event)
            self.assertTrue(recorder.flush())
            trace = json.loads((Path(directory) / "shots/shot_00000007/trace.json").read_text())
            self.assertEqual(trace["decision_trace_v1"]["emission"]["camera_xy"]["value"], [100.0, 100.0])
            recorder.shutdown()

    def test_ai_override_trace_keeps_selected_baseline_ai_proposal_and_push_separate(self):
        from src.engine.ai import bootstrap
        from src.engine.ai import runtime as runtime_module
        from src.engine import physical_trace as trace_module

        original_emit = HitScanner._emit_track_result
        original_update = HitScanner.update
        runtime = SimpleNamespace(settings={"mode": "shadow"},
                                  observe_scanner=lambda *args, **kwargs: None,
                                  choose_for_emission=lambda x, y, **kwargs: {
                                      "apply": True, "camera_x": 333.25, "camera_y": 44.75,
                                      "confidence": 0.9,
                                  },
                                  mark_shot_finished=lambda *args, **kwargs: None)
        with tempfile.TemporaryDirectory() as directory:
            recorder = PhysicalTraceRecorder(directory, enabled=True)
            recorder.decision_trace_enabled = True
            scanner, event, track = self.scanner(trace=True)
            recorder.start_from_scanner(scanner, event, {})
            with patch.object(trace_module, "get_physical_trace_recorder", return_value=recorder), \
                 patch.object(runtime_module, "get_ai_runtime", return_value=runtime):
                bootstrap._patch_hit_scanner()
                try:
                    with patch.object(hit_input, "push_camera_hit") as push:
                        self.assertTrue(scanner._emit_track_result(track, event))
                        push.assert_called_once_with(333.25, 44.75)
                finally:
                    HitScanner._emit_track_result = original_emit
                    HitScanner.update = original_update
            emission = recorder._active[7]["trace"]["decision_trace_runtime_v1"]["emission"]
            self.assertEqual(emission["candidate_camera_xy"]["value"], [123.5, 41.25])
            self.assertEqual(emission["camera_xy"]["value"], [333.25, 44.75])
            self.assertEqual(emission["push_attempt"]["camera_xy"], [333.25, 44.75])
            recorder.finish(7, scanner, event)
            self.assertTrue(recorder.flush())
            recorder.shutdown()

    def test_bootstrap_wrapper_records_failed_push_before_reraising(self):
        from src.engine.ai import bootstrap
        from src.engine.ai import runtime as runtime_module
        from src.engine import physical_trace as trace_module

        original_emit = HitScanner._emit_track_result
        original_update = HitScanner.update
        runtime = SimpleNamespace(settings={"mode": "train_only"},
                                  observe_scanner=lambda *args, **kwargs: None,
                                  choose_for_emission=lambda x, y, **kwargs: {"apply": False},
                                  mark_shot_finished=lambda *args, **kwargs: None)
        with tempfile.TemporaryDirectory() as directory:
            recorder = PhysicalTraceRecorder(directory, enabled=True)
            recorder.decision_trace_enabled = True
            scanner, event, track = self.scanner(trace=True)
            recorder.start_from_scanner(scanner, event, {})
            with patch.object(trace_module, "get_physical_trace_recorder", return_value=recorder), \
                 patch.object(runtime_module, "get_ai_runtime", return_value=runtime):
                bootstrap._patch_hit_scanner()
                try:
                    with patch.object(hit_input, "push_camera_hit", side_effect=RuntimeError("synthetic push failure")):
                        with self.assertRaisesRegex(RuntimeError, "synthetic push failure"):
                            scanner._emit_track_result(track, event)
                finally:
                    HitScanner._emit_track_result = original_emit
                    HitScanner.update = original_update
            emission = recorder._active[7]["trace"]["decision_trace_runtime_v1"]["emission"]
            self.assertFalse(emission["emitted"])
            self.assertEqual(emission["push_attempt"]["call_state"], "RAISED")
            self.assertEqual(emission["push_attempt"]["camera_xy"], [123.5, 41.25])
            recorder.finish(7, scanner, event)
            self.assertTrue(recorder.flush())
            trace = json.loads((Path(directory) / "shots/shot_00000007/trace.json").read_text())
            self.assertEqual(trace["decision_trace_v1"]["emission"]["push_attempt"]["call_state"], "RAISED")
            recorder.shutdown()

    def test_bootstrap_diagnostic_finalizer_failure_does_not_mask_push_failure(self):
        from src.engine.ai import bootstrap
        from src.engine.ai import runtime as runtime_module
        from src.engine import physical_trace as trace_module

        original_emit = HitScanner._emit_track_result
        original_update = HitScanner.update
        runtime = SimpleNamespace(settings={"mode": "train_only"},
                                  observe_scanner=lambda *args, **kwargs: None,
                                  choose_for_emission=lambda x, y, **kwargs: {"apply": False},
                                  mark_shot_finished=lambda *args, **kwargs: None)
        with tempfile.TemporaryDirectory() as directory:
            recorder = PhysicalTraceRecorder(directory, enabled=True)
            recorder.decision_trace_enabled = True
            scanner, event, track = self.scanner(trace=True)
            recorder.start_from_scanner(scanner, event, {})
            with patch.object(trace_module, "get_physical_trace_recorder", return_value=recorder), \
                 patch.object(runtime_module, "get_ai_runtime", return_value=runtime):
                bootstrap._patch_hit_scanner()
                try:
                    with patch.object(recorder, "record_emission_decision",
                                      side_effect=RuntimeError("synthetic trace failure")) as finalize, \
                         patch.object(hit_input, "push_camera_hit",
                                      side_effect=ValueError("synthetic push failure")):
                        with self.assertRaisesRegex(ValueError, "synthetic push failure"):
                            scanner._emit_track_result(track, event)
                        finalize.assert_called_once_with(scanner, event, track)
                finally:
                    HitScanner._emit_track_result = original_emit
                    HitScanner.update = original_update
            recorder.shutdown()

    def test_opt_in_audio_pipe_window_is_bounded_metadata_only_and_shot_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PhysicalTraceRecorder(directory, enabled=True)
            recorder.decision_trace_enabled = True
            scanner, event, track = self.scanner(trace=True)
            event.audio_trigger = {"timestamp": event.peak_ts, "diagnostics": {"acquisition": {
                "stream_id": "pcm-test", "sequence": 10, "sample_start_index": 2304,
                "sample_end_index_exclusive": 2560, "read_start_monotonic": 11.0,
                "read_complete_monotonic": 11.016, "receive_wall_ts": 21.0,
                "device_timestamp": None, "upstream_buffer_age_s": None,
            }}}

            def blocks(first, last, *, limit=129):
                if first <= 10 <= last:
                    return [{"stream_id": "pcm-test", "sequence": seq,
                             "sample_start_index": (seq-1)*256,
                             "read_complete_monotonic": 11.0+seq*.016}
                            for seq in range(first,last+1)]
                if first == 11 and last == 74:
                    return [{"stream_id": "pcm-test", "sequence": 11, "sample_start_index": 2560,
                             "read_complete_monotonic": 11.032}]
                self.fail(f"unexpected metadata window {first}..{last}, limit={limit}")

            with patch.object(AudioPeakDetector, "acquisition_metadata_window", side_effect=blocks):
                recorder.start_from_scanner(scanner, event, {})
                recorder.capture_decision(scanner, track, event, {})
                event.state = "missed"
                recorder.finish(7, scanner, event)
                self.assertTrue(recorder.flush())

            trace = json.loads((Path(directory) / "shots/shot_00000007/trace.json").read_text())
            timing = trace["audio_pipe_timing_v1"]
            self.assertEqual(timing["state"], "AVAILABLE")
            self.assertEqual(timing["trigger_sequence"], 10)
            self.assertEqual([row["sequence"] for row in timing["preceding_and_trigger_blocks"]], list(range(1,11)))
            self.assertTrue(timing["pre_window"]["complete"])
            self.assertEqual([row["sequence"] for row in timing["post_blocks"]], [11])
            self.assertEqual(timing["post_capture_state"], "PARTIAL")
            self.assertIsNone(timing["device_timestamp"])
            self.assertIsNone(timing["upstream_buffer_age_s"])
            self.assertFalse(any("pcm" in key or "samples" in key for row in
                                 timing["preceding_and_trigger_blocks"] + timing["post_blocks"]
                                 for key in row))
            recorder.shutdown()

    def test_opt_in_audio_timing_marks_missing_acquisition_unrecorded(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PhysicalTraceRecorder(directory, enabled=True)
            recorder.decision_trace_enabled = True
            scanner, event, _track = self.scanner(trace=True)
            recorder.start_from_scanner(scanner, event, {})
            timing = recorder._active[7]["trace"]["audio_pipe_timing_v1"]
            self.assertEqual(timing["state"], "UNRECORDED")
            self.assertIsNone(timing["device_timestamp"])
            recorder.shutdown()

    def test_audio_pipe_window_is_not_snapshotted_when_decision_trace_off(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PhysicalTraceRecorder(directory, enabled=True)
            recorder.decision_trace_enabled = False
            scanner, event, _track = self.scanner(trace=False)
            with patch("src.engine.physical_trace._audio_pipe_timing_snapshot",
                       side_effect=AssertionError("opt-in snapshot must not run")):
                recorder.start_from_scanner(scanner, event, {})
            self.assertNotIn("audio_pipe_timing_v1", recorder._active[7]["trace"])
            recorder.shutdown()

    def test_selector_path_and_rescue_state_are_shot_scoped(self):
        scanner, event, track = self.scanner(trace=True)
        other = AudioShotEvent(8, 101.0, 101.0)
        update_tracks_frame_unique_v2226(scanner, [candidate(300.0, sid=8)], 101.125)
        scanner.audio_events.append(other)
        first = scanner._best_track_for_event(event)
        second = scanner._best_track_for_event(other)
        record_selection(scanner, event, first, "V253")
        record_selection(scanner, other, second, "V253")
        record_selector_path(scanner, event, owner="V253", branch="REQUEST_FULL_RESCUE",
                             reason="no_shared_registered_frame")
        self.assertEqual(len(scanner._audit_selections[7]["selector_path_hops"]), 1)
        self.assertNotIn("selector_path_hops", scanner._audit_selections[8])

        from src.engine.shot_fast_v2225 import rescue_router_v2225
        with patch.object(rescue_router_v2225, "requested", return_value=True), \
             patch.object(rescue_router_v2225, "was_consumed", return_value=False):
            rescue = PhysicalTraceRecorder._rescue_state(7)
        self.assertEqual(rescue["state"], "AVAILABLE")
        self.assertTrue(rescue["requested"])
        self.assertFalse(rescue["consumed"])

    def test_branch_ledger_is_bounded_and_marks_truncation(self):
        scanner, event, track = self.scanner(trace=True)
        record_selection(scanner, event, track, "V252")
        for index in range(70):
            record_selector_path(scanner, event, owner="V252", branch=f"WAIT_{index}")
        audit = scanner._audit_selections[7]
        self.assertEqual(len(audit["selector_path_hops"]), 64)
        self.assertTrue(audit["selector_path_hops_truncated"])

    def test_source_frame_identity_requires_exact_saved_timestamp_join(self):
        trace = {
            "shot_id": 7, "peak_ts": 100.0,
            "frames": [{"timestamp": 100.125, "acquisition": {"stream_id": "cam-a", "sequence": 22}}],
            "decision_input": {"complete_track_audit": {"complete": True, "tracks": [{
                "track_id": 1, "observation_id": "7:1:0", "camera_x": 123.5, "camera_y": 41.25,
                "source": "V2", "last_candidate": {"timestamp": 100.125},
            }]}},
        }
        row = _build_decision_trace_v1(trace)["candidate_ledger"]["tracks"][0]
        self.assertEqual(row["source_frame_ids"]["state"], "AVAILABLE")
        self.assertEqual(row["source_frame_ids"]["value"], {
            "stream_id": "cam-a", "sequence": 22, "timestamp": 100.125,
        })
        trace["decision_input"]["complete_track_audit"]["tracks"][0]["last_candidate"]["timestamp"] = 100.126
        row = _build_decision_trace_v1(trace)["candidate_ledger"]["tracks"][0]
        self.assertEqual(row["source_frame_ids"]["state"], "UNRECORDED")

    def test_duplicate_source_timestamps_do_not_claim_arbitrary_frame_identity(self):
        trace = {
            "shot_id": 7, "peak_ts": 100.0,
            "frames": [
                {"timestamp": 100.125, "acquisition": {"stream_id": "cam-a", "sequence": 22}},
                {"timestamp": 100.125, "acquisition": {"stream_id": "cam-b", "sequence": 91}},
            ],
            "decision_input": {"complete_track_audit": {"complete": True, "tracks": [{
                "track_id": 1, "observation_id": "7:1:0", "camera_x": 123.5, "camera_y": 41.25,
                "source": "V2", "last_candidate": {"timestamp": 100.125},
            }]}},
        }
        row = _build_decision_trace_v1(trace)["candidate_ledger"]["tracks"][0]
        self.assertEqual(row["source_frame_ids"]["state"], "AMBIGUOUS")
        self.assertIsNone(row["source_frame_ids"]["value"])
        self.assertIn("multiple saved frame identities", row["source_frame_ids"]["reason"])

    def test_null_candidate_timestamp_falls_back_to_source_frame_timestamp(self):
        trace = {
            "shot_id": 7, "peak_ts": 100.0,
            "frames": [{"timestamp": 100.125, "acquisition": {"stream_id": "cam-a", "sequence": 22}}],
            "decision_input": {"complete_track_audit": {"complete": True, "tracks": [{
                "track_id": 1, "observation_id": "7:1:0", "camera_x": 123.5, "camera_y": 41.25,
                "source": "V2", "last_candidate": {"timestamp": None, "source_frame_ts": 100.125},
            }]}},
        }
        row = _build_decision_trace_v1(trace)["candidate_ledger"]["tracks"][0]
        self.assertEqual(row["source_frame_ids"]["state"], "AVAILABLE")
        self.assertEqual(row["source_frame_ids"]["value"], {
            "stream_id": "cam-a", "sequence": 22, "timestamp": 100.125,
        })

    def test_source_frame_fallback_does_not_replace_raw_observation_timestamp(self):
        trace = {
            "shot_id": 7, "peak_ts": 100.0,
            "frames": [{"timestamp": 100.125, "acquisition": {"stream_id": "cam-a", "sequence": 22}}],
            "decision_input": {"complete_track_audit": {"complete": True, "tracks": [{
                "track_id": 1, "observation_id": "7:1:0", "camera_x": 123.5, "camera_y": 41.25,
                "source": "V2", "last_candidate": {"timestamp": None, "source_frame_ts": 100.125},
            }]}},
        }
        row = _build_decision_trace_v1(trace)["candidate_ledger"]["tracks"][0]
        self.assertEqual(row["source_frame_ids"]["state"], "AVAILABLE")
        self.assertEqual(row["source_frame_ids"]["value"]["timestamp"], 100.125)
        self.assertEqual(row["source_observation_timestamp"]["state"], "MISSING")
        self.assertIsNone(row["source_observation_timestamp"]["value"])
        self.assertIn("raw candidate timestamp", row["source_observation_timestamp"]["semantics"])

    def test_conflicting_candidate_timestamps_join_on_observation_timestamp(self):
        trace = {
            "shot_id": 7, "peak_ts": 100.0,
            "frames": [
                {"timestamp": 100.125, "acquisition": {"stream_id": "cam-a", "sequence": 22}},
                {"timestamp": 100.250, "acquisition": {"stream_id": "cam-b", "sequence": 23}},
            ],
            "decision_input": {"complete_track_audit": {"complete": True, "tracks": [{
                "track_id": 1, "observation_id": "7:1:0", "camera_x": 123.5, "camera_y": 41.25,
                "source": "V2", "last_candidate": {
                    "timestamp": 100.125, "source_frame_ts": 100.250,
                },
            }]}},
        }
        row = _build_decision_trace_v1(trace)["candidate_ledger"]["tracks"][0]
        self.assertEqual(row["source_frame_ids"]["state"], "AVAILABLE")
        self.assertEqual(row["source_frame_ids"]["value"], {
            "stream_id": "cam-a", "sequence": 22, "timestamp": 100.125,
        })
        self.assertEqual(row["source_observation_timestamp"]["value"], 100.125)
        self.assertEqual(row["source_frame_timestamp"]["value"], 100.250)
        self.assertEqual(row["source_frame_timestamp"]["state"], "AVAILABLE")

    def test_terminal_audit_fallback_references_the_actual_source(self):
        trace = {
            "shot_id": 7, "peak_ts": 100.0,
            "terminal_track_audit_v1": {"complete": True, "tracks": [{
                "track_id": 2, "observation_id": "7:2:0", "camera_x": 8.0, "camera_y": 9.0,
                "source": "V2", "last_candidate": {"camera_x": 8.0, "camera_y": 9.0},
            }]},
            "decision_input": {},
        }
        projection = _build_decision_trace_v1(trace)
        row = projection["candidate_ledger"]["tracks"][0]
        self.assertEqual(row["evidence_values_ref"]["path"],
                         "/terminal_track_audit_v1/tracks/0/last_candidate")
        self.assertEqual(projection["candidate_ledger"]["source"], "terminal_track_audit_v1")

    def test_capture_opt_in_is_separate_and_disabled_by_default(self):
        recorder = PhysicalTraceRecorder(enabled=False)
        recorder.configure({"physical_trace_capture_enabled": False,
                            "physical_decision_trace_enabled": True})
        self.assertFalse(recorder.enabled)
        self.assertFalse(recorder.decision_trace_enabled)
        with patch.dict("os.environ", {"SKJUTBANA_PHYSICAL_DECISION_TRACE": "1"}):
            recorder.configure({"physical_trace_capture_enabled": True})
        self.assertTrue(recorder.enabled)
        self.assertTrue(recorder.decision_trace_enabled)
        recorder.shutdown()

    def test_existing_trace_has_no_v1_projection_without_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PhysicalTraceRecorder(directory, enabled=True)
            scanner, event, track = self.scanner(trace=True)
            scanner.physical_decision_trace_v1_enabled = False
            recorder.capture_decision(scanner, track, event, {})
            event.state = "missed"
            recorder.finish(7, scanner, event)
            self.assertTrue(recorder.flush())
            trace = json.loads((Path(directory) / "shots/shot_00000007/trace.json").read_text())
            self.assertNotIn("decision_trace_v1", trace)
            recorder.shutdown()


if __name__ == "__main__":
    unittest.main()
