"""Optional radial lens model for camera<->screen mapping (calibration field).

Calibration dict field (absent = plain homography, today's behaviour)::

    "camera_distortion": {
        "model": "radial2_v1",
        "convention": "distorted_to_undistorted",
        "center_px": [cx, cy],          # raw (distorted) camera pixels
        "normalization_px": 3840.0,      # r is measured in units of this length
        "k1": .., "k2": ..,
        "homography_undistorted": 3x3,   # UNDISTORTED camera px -> screen px
        "inverse_homography_undistorted": 3x3,
        "fit": {...}                     # provenance / validation, informational
    }

With d = (p - C) / N and r2 = |d|^2, undistort(p) = C + N * d * (1 + k1*r2 + k2*r2^2).
The top-level calibration "homography"/"inverse_homography" stay the plain
raw-pixel fit (approximate users such as crop/ROI geometry keep today's
behaviour); only lens-aware mappers use the undistorted-plane matrices stored
inside this field. Canonical camera XY everywhere else (traces, GT, candidates, hits) stays in raw
distorted camera pixels; correction is applied only inside the mapping.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

MODEL = "radial2_v1"


def parse(calibration: Mapping[str, Any] | None) -> dict[str, float] | None:
    raw = (calibration or {}).get("camera_distortion")
    if not isinstance(raw, Mapping) or raw.get("model") != MODEL:
        return None
    if raw.get("convention", "distorted_to_undistorted") != "distorted_to_undistorted":
        return None
    try:
        cx, cy = (float(v) for v in raw["center_px"])
        model = dict(cx=cx, cy=cy, n=float(raw.get("normalization_px", 3840.0)),
                     k1=float(raw["k1"]), k2=float(raw["k2"]))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in model.values()) or model["n"] <= 0:
        return None
    try:
        H = np.array(raw["homography_undistorted"], dtype=np.float64)
        Hinv = np.array(raw.get("inverse_homography_undistorted") or np.linalg.inv(H), dtype=np.float64)
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return None
    if H.shape != (3, 3) or Hinv.shape != (3, 3) or not (np.isfinite(H).all() and np.isfinite(Hinv).all()):
        return None
    model["H"] = H
    model["Hinv"] = Hinv
    return model


def camera_to_screen(m: Mapping[str, Any], x: float, y: float) -> tuple[float, float]:
    import cv2
    u = undistort_points(np.array([[x, y]]), m)
    r = cv2.perspectiveTransform(u.reshape(-1, 1, 2), m["H"])[0, 0]
    return float(r[0]), float(r[1])


def screen_to_camera(m: Mapping[str, Any], x: float, y: float) -> tuple[float, float]:
    import cv2
    u = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float64), m["Hinv"])[0]
    p = distort_points(u, m)[0]
    return float(p[0]), float(p[1])


def undistort_points(points: np.ndarray, m: Mapping[str, float]) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    c = np.array([m["cx"], m["cy"]])
    d = (p - c) / m["n"]
    r2 = (d * d).sum(axis=1, keepdims=True)
    return c + m["n"] * d * (1.0 + m["k1"] * r2 + m["k2"] * r2 * r2)


def distort_points(points: np.ndarray, m: Mapping[str, float], iterations: int = 30) -> np.ndarray:
    """Inverse of undistort_points by fixed-point iteration (converges for mild lenses)."""
    u = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    c = np.array([m["cx"], m["cy"]])
    du = (u - c) / m["n"]
    d = du.copy()
    for _ in range(iterations):
        r2 = (d * d).sum(axis=1, keepdims=True)
        d = du / (1.0 + m["k1"] * r2 + m["k2"] * r2 * r2)
    return c + m["n"] * d


def fit(camera_points: Sequence[Sequence[float]], screen_points: Sequence[Sequence[float]],
        *, normalization_px: float = 3840.0, center_px: Sequence[float] = (1920.0, 1080.0)):
    """Fit k1, k2 (and centre) plus a homography in the undistorted plane.

    Returns (model_dict, H_undistorted_camera_to_screen, rms_screen_px). Pure
    numpy/OpenCV coarse grid + coordinate-descent refinement (no SciPy).
    """
    import cv2
    cam = np.asarray(camera_points, dtype=np.float64).reshape(-1, 2)
    scr = np.asarray(screen_points, dtype=np.float64).reshape(-1, 2)

    def sse(k1, k2, cx, cy):
        m = dict(cx=cx, cy=cy, n=normalization_px, k1=k1, k2=k2)
        u = undistort_points(cam, m)
        H = cv2.findHomography(u, scr, 0)[0]
        if H is None:
            return math.inf, None
        pred = cv2.perspectiveTransform(u.reshape(-1, 1, 2), H).reshape(-1, 2)
        return float(((pred - scr) ** 2).sum()), H

    cx, cy = float(center_px[0]), float(center_px[1])
    best = (math.inf, 0.0, 0.0)
    for k1 in np.linspace(-2.0, 2.0, 41):
        for k2 in np.linspace(-4.0, 4.0, 41):
            e, _ = sse(k1, k2, cx, cy)
            if e < best[0]:
                best = (e, float(k1), float(k2))
    e, k1, k2 = best
    x = np.array([k1, k2, cx, cy])
    step = np.array([0.05, 0.1, 40.0, 40.0])
    for _ in range(80):
        improved = False
        for i in range(4):
            for s in (-1.0, 1.0):
                y = x.copy()
                y[i] += s * step[i]
                ee, _ = sse(*y)
                if ee < e:
                    e, x, improved = ee, y, True
        if not improved:
            step /= 2.0
            if step[0] < 1e-4:
                break
    e, H = sse(*x)
    model = dict(model=MODEL, convention="distorted_to_undistorted", center_px=[float(x[2]), float(x[3])],
                 normalization_px=float(normalization_px), k1=float(x[0]), k2=float(x[1]),
                 homography_undistorted=H.tolist(), inverse_homography_undistorted=np.linalg.inv(H).tolist())
    return model, H, math.sqrt(e / max(1, len(cam)))


def leave_one_out(camera_points, screen_points, *, with_distortion: bool) -> list[float]:
    """Held-out marker screen error; distortion AND H are refit per fold."""
    import cv2
    cam = np.asarray(camera_points, dtype=np.float64).reshape(-1, 2)
    scr = np.asarray(screen_points, dtype=np.float64).reshape(-1, 2)
    errors = []
    for i in range(len(cam)):
        keep = np.ones(len(cam), bool)
        keep[i] = False
        if with_distortion:
            model, H, _ = fit(cam[keep], scr[keep])
            p = undistort_points(cam[i:i + 1], parse({"camera_distortion": model}))
        else:
            H = cv2.findHomography(cam[keep], scr[keep], 0)[0]
            p = cam[i:i + 1]
        pred = cv2.perspectiveTransform(p.reshape(-1, 1, 2), H).reshape(-1, 2)
        errors.append(float(np.linalg.norm(pred[0] - scr[i])))
    return errors


def evaluate_and_build(calibration: Mapping[str, Any]) -> tuple[dict | None, dict]:
    """Fit the lens model for a calibration and decide acceptance (held-out markers).

    Accepted only when the leave-one-out median error drops below 0.8x the plain
    homography's and the max error does not grow.  Returns (model or None, report).
    Takes several seconds (24 markers ~8 s): never call it on the render thread.
    """
    cam_pts, scr_pts = calibration.get("camera_points"), calibration.get("viewport_points")
    if not cam_pts or not scr_pts or len(cam_pts) < 8 or len(cam_pts) != len(scr_pts):
        return None, dict(accepted=False, reason="too few correspondences")
    import time as _time
    plain = np.array(leave_one_out(cam_pts, scr_pts, with_distortion=False))
    lens = np.array(leave_one_out(cam_pts, scr_pts, with_distortion=True))
    model, _H, rms = fit(cam_pts, scr_pts)
    accept = bool(np.median(lens) < 0.8 * np.median(plain) and lens.max() < plain.max())
    model["fit"] = dict(markers=len(cam_pts), rms_screen_px=rms,
                        loo_plain_median_screen_px=float(np.median(plain)), loo_plain_max_screen_px=float(plain.max()),
                        loo_lens_median_screen_px=float(np.median(lens)), loo_lens_max_screen_px=float(lens.max()),
                        accepted=accept, fitted_at=_time.strftime("%Y-%m-%dT%H:%M:%S"))
    return (model if accept else None), model["fit"]


def calibration_fingerprint(calibration: Mapping[str, Any]) -> str:
    """Hash of the inputs a lens fit depends on (correspondences + plain homography)."""
    import hashlib
    import json as _json
    key = {k: calibration.get(k) for k in ("camera_points", "viewport_points", "homography", "camera_frame_size",
                                           "viewport_rect", "calibrated_at")}
    return hashlib.sha256(_json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


_PENDING: list = []          # accepted (fingerprint, model) waiting for the main thread


def refit_saved_calibration_async(calibration: Mapping[str, Any], log=print):
    """Background: FIT ONLY.  Applying happens on the main thread (apply_pending).

    The plain calibration is already saved and live.  The worker fits the lens
    model (~8 s) on a copy and queues it if it beats the plain homography on
    held-out markers.  ``apply_pending()`` (called once per frame by the live
    loop) saves it only if the saved calibration still has the same fingerprint.
    """
    import copy
    import threading
    snapshot = copy.deepcopy(dict(calibration))
    fingerprint = calibration_fingerprint(snapshot)

    def work():
        try:
            model, report = evaluate_and_build(snapshot)
            log(f"[LENS-MODEL] background fit: accepted={report.get('accepted')} "
                f"loo median {report.get('loo_plain_median_screen_px', float('nan')):.2f} -> "
                f"{report.get('loo_lens_median_screen_px', float('nan')):.2f} screen px")
            if model is not None:
                _PENDING.append((fingerprint, model))
        except Exception as exc:   # never break calibration
            log(f"[LENS-MODEL] background fit failed: {exc}")

    thread = threading.Thread(target=work, name="lens-model-fit", daemon=True)
    thread.start()
    return thread


def apply_pending(log=print) -> bool:
    """Main thread: save a queued lens model if its calibration is still the saved one."""
    if not _PENDING:
        return False
    fingerprint, model = _PENDING.pop(0)
    from src.engine.settings import load_camera_calibration, save_camera_calibration
    current = load_camera_calibration() or {}
    if current.get("camera_distortion") or calibration_fingerprint(current) != fingerprint:
        log("[LENS-MODEL] calibration changed meanwhile; lens model not saved")
        return False
    current["camera_distortion"] = model
    save_camera_calibration(current)
    from src.engine.input.hit_input import hit_input
    hit_input.reload_calibration()
    log("[LENS-MODEL] lens model added to the saved calibration")
    return True
