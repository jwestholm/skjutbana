"""V2.22.2 fast-path cleanup selftest.

Run from repository root:
    python3 -m automation.v2222_selftest

No camera/projector is required.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.engine.camera.analysis_filters_v2222 import (
    apply_novelty_cleanup_v2222,
    project_camera_candidates_to_screen_v2222,
    suppress_horizontal_ridges_v2222,
)
from src.engine.camera.hit_scanner_v2222 import _select_backlog_frames


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def test_old_hole_novelty() -> None:
    known = [{"camera_x": 100.0, "camera_y": 100.0}]
    candidates = [
        {"camera_x": 101.0, "camera_y": 101.0, "score": 30.0, "pre_shot_change": 0.4},
        {"camera_x": 102.0, "camera_y": 100.0, "score": 22.0, "pre_shot_change": 11.0},
        {"camera_x": 400.0, "camera_y": 300.0, "score": 18.0, "pre_shot_change": 9.0},
        # Ensure PRE-shot evidence is globally informative.
        {"camera_x": 450.0, "camera_y": 320.0, "score": 9.0, "pre_shot_change": 8.0},
    ]
    cleaned, stats = apply_novelty_cleanup_v2222(
        candidates, known, duplicate_radius_px=18.0, fresh_rehit_min=5.0
    )
    coords = {(round(c["camera_x"]), round(c["camera_y"])) for c in cleaned}
    _check((101, 101) not in coords, "stale registered old-hole candidate is removed")
    _check((102, 100) in coords, "fresh PRE->POST evidence preserves a genuine re-hit")
    _check(stats["stale_known_removed"] == 1, "old-hole cleanup reports its removal")


def test_static_unknown_is_demoted_not_hard_dropped() -> None:
    candidates = [
        {"camera_x": 10.0, "camera_y": 10.0, "score": 20.0, "pre_shot_change": 0.5},
        {"camera_x": 50.0, "camera_y": 50.0, "score": 10.0, "pre_shot_change": 10.0},
        {"camera_x": 80.0, "camera_y": 80.0, "score": 9.0, "pre_shot_change": 8.0},
    ]
    cleaned, stats = apply_novelty_cleanup_v2222(candidates, [])
    old = next(c for c in cleaned if c["camera_x"] == 10.0)
    _check(old["score"] < 3.0, "static unregistered hole appearance is strongly demoted")
    _check(len(cleaned) == len(candidates), "unknown low-novelty candidate is demoted rather than blindly deleted")
    _check(stats["novelty_demoted"] >= 1, "novelty demotion is reported")


def test_perspective_horizontal_ridge() -> None:
    screen = np.asarray(
        [[0.0, 0.0], [1920.0, 0.0], [1920.0, 1080.0], [0.0, 1080.0]],
        dtype=np.float32,
    )
    camera = np.asarray(
        [[620.0, 285.0], [3290.0, 405.0], [3140.0, 1900.0], [785.0, 1790.0]],
        dtype=np.float32,
    )
    H_screen_to_camera = cv2.getPerspectiveTransform(screen, camera)
    H_camera_to_screen = cv2.getPerspectiveTransform(camera, screen)

    ridge_screen = np.asarray([[150.0 + i * 150.0, 420.0] for i in range(11)], dtype=np.float32)
    ridge_cam = cv2.perspectiveTransform(ridge_screen.reshape(-1, 1, 2), H_screen_to_camera).reshape(-1, 2)
    candidates = [
        {
            "camera_x": float(x), "camera_y": float(y), "score": 10.0 + i * 0.1,
            "pre_shot_change": 1.0,
        }
        for i, (x, y) in enumerate(ridge_cam)
    ]
    # One genuinely fresh hit lies on the motion ridge and must survive.
    candidates[5]["pre_shot_change"] = 12.0
    candidates[5]["score"] = 25.0
    # Independent candidate away from the ridge.
    off_screen = np.asarray([[[960.0, 700.0]]], dtype=np.float32)
    off_cam = cv2.perspectiveTransform(off_screen, H_screen_to_camera)[0, 0]
    candidates.append({"camera_x": float(off_cam[0]), "camera_y": float(off_cam[1]), "score": 15.0, "pre_shot_change": 9.0})

    projected = project_camera_candidates_to_screen_v2222(candidates, H_camera_to_screen)
    cleaned, stats = suppress_horizontal_ridges_v2222(
        candidates,
        projected,
        screen_rect_xywh=(0.0, 0.0, 1920.0, 1080.0),
        band_px=7.0,
        min_count=9,
        min_span_fraction=0.35,
        fresh_preserve_min=6.0,
        max_preserve_per_ridge=3,
    )
    _check(stats["ridge_groups"] == 1, "physical horizontal band is found after perspective projection")
    _check(stats["ridge_removed"] >= 8, "long horizontal motion ridge is heavily reduced")
    _check(any(c.get("pre_shot_change") == 12.0 for c in cleaned), "strong fresh hit on a ridge is preserved")
    _check(any(abs(c.get("pre_shot_change", 0.0) - 9.0) < 1e-6 for c in cleaned), "off-ridge candidate survives")


def test_backlog_thinning() -> None:
    frames = list(range(31))
    selected = _select_backlog_frames(frames, 3)
    _check(len(selected) == 3, "open-shot frame backlog is bounded")
    _check(selected[-1] == 30, "newest camera frame is always retained")
    _check(selected[0] == 0 and 10 <= selected[1] <= 20, "bounded backlog keeps temporal coverage, not only adjacent frames")


def test_bootstrap_and_defaults() -> None:
    bootstrap = Path("src/engine/ai/bootstrap.py").read_text(encoding="utf-8")
    _check("_install_v2222_hit_scanner()" in bootstrap, "bootstrap installs V2.22.2 fast-path patch")
    _check(
        bootstrap.index("_install_v2221_hit_scanner()") < bootstrap.index("_install_v2222_hit_scanner()") < bootstrap.index("_patch_hit_scanner()"),
        "V2.22.2 installs after perspective ROI and before AIRuntime wraps HitScanner",
    )
    runtime = Path("src/engine/ai/runtime_v222.py").read_text(encoding="utf-8")
    for key in (
        "analysis_cursor_guard_v2222_enabled",
        "analysis_novelty_cleanup_v2222_enabled",
        "analysis_horizontal_ridge_filter_v2222_enabled",
        "analysis_ingest_max_frames_v2222",
        "analysis_v2222_log",
    ):
        _check(key in runtime, f"runtime default exists: {key}")



def test_real_hit_scanner_install_path() -> None:
    """Catch the package-attribute/submodule ambiguity seen in the real game.

    src.engine.camera.__init__ exports a singleton called ``hit_scanner``.
    The installers must explicitly resolve the *submodule*, then patch its
    HitScanner class.  This test intentionally uses the real repository module.
    """
    from importlib import import_module

    hs_module = import_module("src.engine.camera.hit_scanner")
    _check(hasattr(hs_module, "HitScanner"), "real hit_scanner submodule exposes HitScanner class")

    from src.engine.camera.hit_scanner_v2221 import install_v2221_hit_scanner_patch
    from src.engine.camera.hit_scanner_v2222 import install_v2222_hit_scanner_patch

    install_v2221_hit_scanner_patch()
    install_v2222_hit_scanner_patch()
    _check(bool(getattr(hs_module.HitScanner, "_v2221_roi_patch", False)), "V2.22.1 patch is attached to the real HitScanner class")
    _check(bool(getattr(hs_module.HitScanner, "_v2222_fast_cleanup_patch", False)), "V2.22.2 patch is attached to the real HitScanner class")


def main() -> None:
    print("V2.22.2 FAST-PATH CLEANUP SELFTEST")
    print("=================================")
    test_old_hole_novelty()
    test_static_unknown_is_demoted_not_hard_dropped()
    test_perspective_horizontal_ridge()
    test_backlog_thinning()
    test_bootstrap_and_defaults()
    test_real_hit_scanner_install_path()
    print("\nAll V2.22.2 selftests passed.")


if __name__ == "__main__":
    main()
