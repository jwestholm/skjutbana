"""Deterministic stale-frame status check; no camera or game process."""
from __future__ import annotations

import os
import importlib
import time
from unittest.mock import patch

import numpy as np

module = importlib.import_module("src.engine.camera.hit_scanner")
HitScanner = module.HitScanner
ScanportFrame = module.ScanportFrame


def main() -> None:
    scanner = HitScanner()
    scanner.enabled = True
    scanner.state = scanner.STATE_ACTIVE
    scanner._last_frame_ts = 10.0
    scanner.frame_history.clear()
    scanner.frame_history.append(ScanportFrame(10.0, np.zeros((12, 12), dtype=np.uint8)))
    event_now = time.time()
    scanner.audio_events.append(module.AudioShotEvent(shot_id=1, peak_ts=event_now, created_at=event_now))
    frame = np.zeros((12, 12, 3), dtype=np.uint8)

    with patch.object(module.camera_manager, "get_new_frames_since_last_pickup", return_value=[]), \
         patch.object(module.camera_manager, "get_latest_frame", return_value=frame), \
         patch.object(module.camera_manager, "get_latest_timestamp", return_value=10.0), \
         patch.object(scanner, "_detect_frame_candidates", side_effect=AssertionError("stale frame detected")), \
         patch.object(scanner, "_resolve_audio_events", return_value=0), \
         patch.object(scanner, "_prune_finished_events"), \
         patch.object(scanner, "_prune_old_holes"), \
         patch.object(scanner, "_refresh_debug_views"), \
         patch.dict(os.environ, {"SKJUTBANA_REPORT_NO_FRESH_PENDING_FRAME": "1"}):
        scanner.update(0.016)

    assert scanner.last_status == "pending_shot_no_fresh_camera_frame", scanner.last_status
    assert scanner.audio_events[0].state == "pending"
    print("PASS pending shot with unchanged camera timestamp is reported as lacking a fresh frame")
    print("PASS stale frame is not redetected and event semantics remain unchanged")

    with patch.object(module.camera_manager, "get_new_frames_since_last_pickup", return_value=[]), \
         patch.object(module.camera_manager, "get_latest_frame", return_value=frame), \
         patch.object(module.camera_manager, "get_latest_timestamp", return_value=10.0), \
         patch.object(scanner, "_detect_frame_candidates", side_effect=AssertionError("stale frame detected")), \
         patch.object(scanner, "_resolve_audio_events", return_value=0), \
         patch.object(scanner, "_prune_finished_events"), \
         patch.object(scanner, "_prune_old_holes"), \
         patch.object(scanner, "_refresh_debug_views"), \
         patch.dict(os.environ, {}, clear=True):
        scanner.update(0.016)
    assert scanner.last_status.startswith("active peaks="), scanner.last_status
    print("PASS diagnostic remains off by default")

    with patch.object(module.camera_manager, "get_new_frames_since_last_pickup", return_value=[]), \
         patch.object(module.camera_manager, "get_latest_frame", return_value=None), \
         patch.dict(os.environ, {"SKJUTBANA_REPORT_NO_FRESH_PENDING_FRAME": "1"}):
        scanner.update(0.016)
    assert scanner.last_status == "pending_shot_no_fresh_camera_frame", scanner.last_status
    print("PASS missing current frame reports pending-shot freshness fault when opted in")

    overdue = HitScanner()
    overdue.enabled = True
    overdue.state = overdue.STATE_ACTIVE
    overdue_event = module.AudioShotEvent(
        shot_id=2, peak_ts=time.time() - overdue.event_timeout_s - 1.0,
        created_at=time.time() - overdue.event_timeout_s - 1.0)
    overdue.audio_events.append(overdue_event)
    with patch.object(module.camera_manager, "get_new_frames_since_last_pickup", return_value=[]), \
         patch.object(module.camera_manager, "get_latest_frame", return_value=None), \
         patch.dict(os.environ, {"SKJUTBANA_RESOLVE_WITHOUT_CAMERA": "1"}, clear=True):
        overdue.update(0.016)
    assert overdue_event.state == "missed", overdue_event.state
    assert overdue_event.lifecycle_state == "EXPIRED"
    print("PASS overdue shot expires when no current camera frame exists")

    recent = HitScanner()
    recent.enabled = True
    recent.state = recent.STATE_ACTIVE
    recent.audio_events.append(module.AudioShotEvent(
        shot_id=3, peak_ts=time.time(), created_at=time.time()))
    with patch.object(module.camera_manager, "get_new_frames_since_last_pickup", return_value=[]), \
         patch.object(module.camera_manager, "get_latest_frame", return_value=None), \
         patch.dict(os.environ, {"SKJUTBANA_RESOLVE_WITHOUT_CAMERA": "1"}, clear=True):
        recent.update(0.016)
    assert recent.audio_events[0].state == "pending"
    print("PASS non-overdue shot remains pending without a camera frame")

    default_path = HitScanner()
    default_path.enabled = True
    default_path.state = default_path.STATE_ACTIVE
    default_event = module.AudioShotEvent(
        shot_id=4, peak_ts=time.time() - default_path.event_timeout_s - 1.0,
        created_at=time.time() - default_path.event_timeout_s - 1.0)
    default_path.audio_events.append(default_event)
    with patch.object(module.camera_manager, "get_new_frames_since_last_pickup", return_value=[]), \
         patch.object(module.camera_manager, "get_latest_frame", return_value=None), \
         patch.dict(os.environ, {}, clear=True):
        default_path.update(0.016)
    assert default_event.state == "pending"
    print("PASS no-camera lifecycle behavior remains unchanged by default")


if __name__ == "__main__":
    main()
