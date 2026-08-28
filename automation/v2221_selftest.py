"""V2.22.1 perspective ROI + integration selftest.

Run from repository root:
    python3 -m automation.v2221_selftest

No camera/projector is required. This validates geometry/coordinate invariants and
static bootstrap integration only; real latency/candidate behaviour must be
verified on the shooting computer afterwards.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np

from src.engine.camera.analysis_geometry_v2221 import build_perspective_geometry_v2221


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def _perspective_fixture():
    screen = np.asarray(
        [[0.0, 0.0], [1920.0, 0.0], [1920.0, 1080.0], [0.0, 1080.0]],
        dtype=np.float32,
    )
    # Deliberately angled/trapezoidal camera view inside a 4K frame.
    camera = np.asarray(
        [[620.0, 285.0], [3290.0, 405.0], [3140.0, 1900.0], [785.0, 1790.0]],
        dtype=np.float32,
    )
    H_screen_to_camera = cv2.getPerspectiveTransform(screen, camera)
    H_camera_to_screen = cv2.getPerspectiveTransform(camera, screen)
    return screen, camera, H_screen_to_camera, H_camera_to_screen


def test_perspective_crop_and_roundtrip() -> None:
    _screen, camera, H_inv, H = _perspective_fixture()
    geom = build_perspective_geometry_v2221(
        (2160, 3840),
        H_inv,
        (0.0, 0.0, 1920.0, 1080.0),
        guard_screen_px=12.0,
        crop_padding_camera_px=16,
    )
    _check(geom.mode == "homography", "homography geometry is selected")
    _check(0.20 < geom.crop_fraction < 0.95, "camera analysis is a bounded crop, not the whole frame")
    _check(geom.safe_pixels < geom.crop_pixels, "safe playfield mask removes the guarded edge band")

    # A point detected in crop-local coordinates must become the exact full
    # camera point before HitInput performs its existing homography.
    camera_point = np.asarray([[[1920.0, 1080.0]]], dtype=np.float32)
    local_x = float(camera_point[0, 0, 0]) - geom.crop_x0
    local_y = float(camera_point[0, 0, 1]) - geom.crop_y0
    global_x, global_y = geom.local_to_camera(local_x, local_y)
    _check(abs(global_x - 1920.0) < 1e-6 and abs(global_y - 1080.0) < 1e-6,
           "crop-local candidate restores exact full-camera XY")

    projected = cv2.perspectiveTransform(
        np.asarray([[[global_x, global_y]]], dtype=np.float32), H
    )[0, 0]
    expected = cv2.perspectiveTransform(camera_point, H)[0, 0]
    _check(float(np.linalg.norm(projected - expected)) < 1e-4,
           "restored camera XY produces identical screen/game homography result")

    # The transformed physical edge lies outside the safe inner polygon; the
    # centre remains searchable.
    edge_cam = camera[0]
    ex = int(round(float(edge_cam[0]) - geom.crop_x0))
    ey = int(round(float(edge_cam[1]) - geom.crop_y0))
    edge_value = 0
    if 0 <= ey < geom.safe_mask_local.shape[0] and 0 <= ex < geom.safe_mask_local.shape[1]:
        edge_value = int(geom.safe_mask_local[ey, ex])
    centre_screen = np.asarray([[[960.0, 540.0]]], dtype=np.float32)
    centre_cam = cv2.perspectiveTransform(centre_screen, H_inv)[0, 0]
    cx = int(round(float(centre_cam[0]) - geom.crop_x0))
    cy = int(round(float(centre_cam[1]) - geom.crop_y0))
    _check(edge_value == 0, "physical playfield corner is excluded by perspective-aware edge guard")
    _check(int(geom.safe_mask_local[cy, cx]) == 255, "playfield centre remains inside searchable mask")


def test_bootstrap_contains_v2221_and_ai_results() -> None:
    text = Path("src/engine/ai/bootstrap.py").read_text(encoding="utf-8")
    _check("_install_v2221_hit_scanner()" in text, "bootstrap installs V2.22.1 HitScanner patch")
    _check('item.type == "ai_results"' in text, "bootstrap restores ai_results scene mapping")
    order_ok = text.index("_install_v2221_hit_scanner()") < text.index("_patch_hit_scanner()")
    _check(order_ok, "ROI patch installs before AI wraps HitScanner")


def test_runtime_defaults_present() -> None:
    text = Path("src/engine/ai/runtime_v222.py").read_text(encoding="utf-8")
    for key in (
        "analysis_roi_crop_v2221_enabled",
        "analysis_playfield_edge_guard_screen_px",
        "analysis_crop_padding_camera_px",
        "analysis_v2221_log",
    ):
        _check(key in text, f"runtime default exists: {key}")


def main() -> None:
    print("V2.22.1 ROI / AI RESULTS SELFTEST")
    print("=================================")
    test_perspective_crop_and_roundtrip()
    test_bootstrap_contains_v2221_and_ai_results()
    test_runtime_defaults_present()
    print("\nAll V2.22.1 selftests passed.")


if __name__ == "__main__":
    main()
