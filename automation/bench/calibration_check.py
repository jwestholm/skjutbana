"""Projector/camera calibration check (and optional recalibration).

  python3 -m automation.bench calibration-check            # validate only, never writes
  python3 -m automation.bench calibration-check --save     # recalibrate, back up settings first

Projects the existing ArUco viewport board in a window at the calibrated desktop
position (default from the saved calibration setup), validates the SAVED mapping
against the observed markers, fits a fresh homography and reports how far the
saved calibration is off. ``--save`` writes the fresh calibration only if it
fits better, after copying content/settings.json to a timestamped backup.
Run it while the game is NOT running (one process owns the camera).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np


def _window_position() -> tuple[int, int]:
    from src.engine.settings import load_camera_calibration
    setup = (load_camera_calibration() or {}).get("setup_at_calibration") or {}
    pos = (setup.get("window") or {}).get("sdl_desktop_position")
    return (int(pos[0]), int(pos[1])) if pos else (0, 0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m automation.bench calibration-check")
    parser.add_argument("--save", action="store_true", help="write the fresh calibration (with backup)")
    parser.add_argument("--window-pos", default=None, help="X,Y desktop position (default: saved setup)")
    parser.add_argument("--settle-s", type=float, default=1.5)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--distortion", action="store_true",
                        help="also fit the optional radial lens model; saved (with --save) only if it beats "
                             "the plain homography on held-out markers")
    args = parser.parse_args(argv)

    x, y = (int(v) for v in args.window_pos.split(",")) if args.window_pos else _window_position()
    import cv2
    import pygame
    from config import SCREEN_HEIGHT, SCREEN_WIDTH
    from src.engine.camera.aruco_calibrator import ArucoCalibrator
    from src.engine.camera.camera_manager import camera_manager
    from src.engine.input.hit_input import hit_input

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    # Same placement primitive as the app's setWindowPos (client-area position),
    # so the projected board lands exactly where the game renders.
    from pygame._sdl2.video import Window
    Window.from_display_module().position = (x, y)
    pygame.display.set_caption("Skjutbana calibration check")
    calibrator = ArucoCalibrator()
    if not calibrator.available:
        print("OpenCV ArUco unavailable")
        return 2
    if not camera_manager.start():
        print(f"Camera unavailable: {camera_manager.last_error}")
        return 2
    try:
        deadline = time.monotonic() + args.settle_s
        while time.monotonic() < deadline:
            pygame.event.pump()
            calibrator.render_markers(screen)
            pygame.display.flip()
            time.sleep(0.03)
        validations, fits = [], []
        for _ in range(args.frames):
            pygame.event.pump()
            calibrator.render_markers(screen)
            pygame.display.flip()
            time.sleep(0.12)
            camera_manager.update()
            frame = camera_manager.get_latest_frame()
            if frame is None:
                continue
            validations.append(calibrator.validate_existing(frame, hit_input))
            fits.append(calibrator.detect_and_calibrate(frame))
    finally:
        camera_manager.stop()
        pygame.quit()

    good = [f for f in fits if f.success]
    report = dict(window_position=[x, y], frames=len(fits), fresh_fit_successes=len(good))
    if validations:
        v = validations[-1]
        rows = v.get("points") or []
        errors = [float(r.get("error_camera_px")) for r in rows if isinstance(r, dict) and r.get("error_camera_px") is not None]
        report["saved_calibration_validation"] = {k: v[k] for k in v if k != "points"}
        if rows:
            report["saved_calibration_mean_offset_camera_px"] = [float(np.mean([r["dx"] for r in rows])),
                                                                  float(np.mean([r["dy"] for r in rows]))]
        if errors:
            report["saved_calibration_marker_error_px"] = dict(median=float(np.median(errors)), max=float(max(errors)),
                                                               count=len(errors))
    if good:
        best = min(good, key=lambda f: f.reprojection_error_px)
        report["fresh_fit"] = dict(markers=best.marker_count, inliers=best.inliers,
                                   reprojection_error_px=float(best.reprojection_error_px))
        old_h = hit_input.calibration_data.get("homography") if isinstance(hit_input.calibration_data, dict) else None
        if old_h is not None:
            # Displacement of the viewport centre between saved and fresh mapping, in camera px.
            from src.engine.settings import load_viewport_rect
            vp = load_viewport_rect()
            centre = np.float64([[[vp.x + vp.w / 2, vp.y + vp.h / 2]]])
            old_cam = cv2.perspectiveTransform(centre, np.linalg.inv(np.array(old_h, dtype=np.float64)))[0, 0]
            new_cam = cv2.perspectiveTransform(centre, np.linalg.inv(np.array(best.calibration_dict["homography"], dtype=np.float64)))[0, 0]
            report["viewport_centre_shift_camera_px"] = [float(new_cam[0] - old_cam[0]), float(new_cam[1] - old_cam[1])]
        if args.distortion or args.save:
            # --save always evaluates the lens model too: a new plain fit must not
            # silently drop a better lens-aware calibration.
            from src.engine import lens_model
            model, fit_report = lens_model.evaluate_and_build(best.calibration_dict)
            report["lens_model"] = fit_report
            if model is not None:
                best.calibration_dict["camera_distortion"] = model
        if args.save:
            import config
            src = Path(getattr(config, "SETTINGS_PATH", "content/settings.json"))
            backup_dir = Path("/data/skjutbana/backups")   # outside the repo: never committed
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / f"settings.backup_{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}.json"
            shutil.copy2(src, backup)
            calibrator.save_and_apply(best, refit_lens=False)   # lens model decided above
            report["saved"] = True
            report["backup"] = str(backup)
    print(json.dumps(report, indent=2, default=str))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
