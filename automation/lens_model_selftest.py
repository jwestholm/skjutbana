"""Deterministic checks for the optional radial lens model (no hardware)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine import lens_model


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def _synthetic(k1=0.9, k2=-2.0):
    H_true = np.array([[0.44, -0.05, -600.0], [0.004, 0.65, -650.0], [-1.3e-6, -1.5e-4, 1.0]])
    truth = dict(cx=1950.0, cy=1090.0, n=3840.0, k1=k1, k2=k2)
    screen = np.array([[x, y] for x in np.linspace(57, 741, 6) for y in np.linspace(38, 560, 4)])
    undist = cv2.perspectiveTransform(screen.reshape(-1, 1, 2), np.linalg.inv(H_true)).reshape(-1, 2)
    camera = lens_model.distort_points(undist, truth)
    return camera, screen, truth


def main():
    camera, screen, truth = _synthetic()
    back = lens_model.undistort_points(lens_model.distort_points(camera, truth), truth)
    check(np.abs(lens_model.distort_points(lens_model.undistort_points(camera, truth), truth) - camera).max() < 1e-6,
          "distort is the inverse of undistort")
    check(lens_model.parse({}) is None and lens_model.parse({"camera_distortion": {"model": "other"}}) is None,
          "absent or unknown lens field means plain homography")
    model, H, rms = lens_model.fit(camera, screen, center_px=(1920.0, 1080.0))
    check(rms < 0.5, f"fit recovers a known distortion (rms {rms:.3f} screen px)")
    plain = lens_model.leave_one_out(camera, screen, with_distortion=False)
    lens = lens_model.leave_one_out(camera, screen, with_distortion=True)
    check(np.median(lens) < 0.5 * np.median(plain), "held-out error improves when the lens is modelled")
    parsed = lens_model.parse({"camera_distortion": model})
    sx, sy = screen[7]
    cx, cy = lens_model.screen_to_camera(parsed, sx, sy)
    rx, ry = lens_model.camera_to_screen(parsed, cx, cy)
    check(abs(rx - sx) < 1e-6 and abs(ry - sy) < 1e-6, "lens-aware screen->camera->screen round trip")
    check(np.hypot(cx - camera[7][0], cy - camera[7][1]) < 1.0, "lens-aware mapping lands on the true raw camera pixel")

    from src.engine.input.hit_input import HitInput
    hi = HitInput.__new__(HitInput)
    hi.calibration_data = {"method": "aruco_viewport_board", "camera_distortion": model,
                           "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    hi.calibration_method = "aruco_viewport_board"
    hi.homography = np.eye(3, dtype=np.float32)
    hi.inverse = np.eye(3, dtype=np.float32)
    hi.lens = parsed
    check(np.allclose(hi.camera_to_screen(cx, cy), (sx, sy), atol=1e-4), "hit_input camera_to_screen uses the lens model")
    check(np.allclose(hi.screen_to_camera(sx, sy), (cx, cy), atol=1e-4), "hit_input screen_to_camera uses the lens model")
    frozen = hi.frozen_camera_to_screen("homography")
    check(np.allclose(frozen(cx, cy), (sx, sy), atol=1e-4), "frozen asynchronous mapper keeps the lens model")
    hi.lens = None
    check(np.allclose(hi.camera_to_screen(10.0, 20.0), (10.0, 20.0)), "without lens field hit_input is unchanged")
    print("ALL LENS MODEL SELFTESTS PASSED")


if __name__ == "__main__":
    main()
