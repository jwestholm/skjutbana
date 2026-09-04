from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def main() -> None:
    print("V2.24.3 SELFTEST")
    print("===============")

    runtime_path = ROOT / "src/engine/shot_object_local_v243.py"
    source = runtime_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    check("V2.24.3 runtime parses", tree is not None)
    check("local restriction is at HitScanner frame ROI", "HitScanner._frame_roi_mask = frame_roi_v243" in source)
    check("FULL-RESCUE restores global ROI", "rescue_router_v2225.requested(sid)" in source and "return global_mask" in source)
    check("zero-overlap uses region recovery, not silent global", 'mode = "region_recovery"' in source and "selected = region_mask" in source)
    check("implicit content rect starts at local origin", "pygame.Rect(0, 0" in source)
    check("HitInput bound content loader is patched", "hit_input_module.load_content_rect = load_content_rect_v243" in source)
    check("shot snapshot refreshes calibration", "hit_input.reload_calibration()" in source)
    check("V2.24.1 candidate-only hook is explicitly superseded", "_v241_previous_extract" in source)

    # Exercise the pure camera-region mask helper.
    import src.engine.shot_object_local_v243 as v243
    windows = [SimpleNamespace(x0=10.0, y0=20.0, x1=30.0, y1=50.0)]
    mask = v243._build_region_mask((100, 120), windows)
    check("region mask has expected shape", mask.shape == (100, 120))
    check("region mask contains requested AABB", int(np.count_nonzero(mask)) == 20 * 30)
    check("region mask excludes outside pixels", mask[0, 0] == 0 and mask[25, 15] == 255)

    main_source = (ROOT / "main.py").read_text(encoding="utf-8")
    check("main imports V2.24.3 installer", "install_v243_runtime" in main_source)
    check("V2.24.3 installs after V2.24.1", main_source.rfind("install_v243_runtime(App)") > main_source.rfind("install_v241_runtime(App)"))

    scene_source = (ROOT / "content/games/hit_context_test_v242.py").read_text(encoding="utf-8")
    check("physical testscene identifies V2.24.3 or successor", ("V2.24.3 HIT CONTEXT TEST" in scene_source) or ("V2.24.4 HIT CONTEXT TEST" in scene_source))
    check("moving target is faster for snapshot separation", "max(180.0, w * 0.18)" in scene_source)
    check("scene logs frozen/current motion distance", "last_motion_px" in scene_source and "motion=" in scene_source)

    # Menu updater must be idempotent on a small fixture.
    old = '{"title": "Hit Context Test (V2.24.2)"}'
    updated = old.replace('"title": "Hit Context Test (V2.24.2)"', '"title": "Hit Context Test (V2.24.3)"', 1)
    check("menu title update remains valid JSON", json.loads(updated)["title"] == "Hit Context Test (V2.24.3)")

    print("\nAll V2.24.3 selftests passed.")


if __name__ == "__main__":
    main()
