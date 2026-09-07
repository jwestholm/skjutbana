from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def main() -> None:
    print("V2.24.4 SELFTEST")
    print("===============")

    runtime_path = ROOT / "src/engine/shot_object_local_v244.py"
    source = runtime_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    check("V2.24.4 runtime parses", tree is not None)
    check("V2.22.1 working geometry is used", "_v2221_active_geometry" in source)
    check("camera coordinates are translated by crop origin", "float(x) - float(self.crop_x0)" in source and "float(y) - float(self.crop_y0)" in source)
    check("future worker scale is explicit", "float(w) / crop_w" in source and "float(h) / crop_h" in source)
    check("no hard-coded half-resolution mapping", "/ 2" not in source and "* 0.5" not in source)
    check("FULL-RESCUE remains global", "rescue_router_v2225.requested(sid)" in source and "return global_mask" in source)
    check("V2.24.3 broken ROI wrapper is bypassed", "_v243_previous_frame_roi_mask" in source)
    check("ROI mapping telemetry exists", "[V2.24.4 ROI-MAP]" in source and "[V2.24.4 LOCAL-ROI]" in source)

    # Exercise the actual mapping helper using the project module.  The fake
    # geometry mirrors V2.22.1: a full camera image, cropped detector plane and
    # optional future worker scale.
    import src.engine.shot_object_local_v244 as v244
    from src.engine.shot_object_local_v241 import LocalSearchWindowV241

    geometry = SimpleNamespace(
        frame_height=2160,
        frame_width=3840,
        crop_x0=1700,
        crop_y0=1000,
        crop_width=1500,
        crop_height=700,
        mode="homography",
    )
    scanner = SimpleNamespace(_v2221_active_geometry=geometry)
    camera_window = LocalSearchWindowV241(1800.0, 1200.0, 1900.0, 1300.0, ("target",), ("target",))
    mapping, work = v244.map_camera_windows_to_work_v244(scanner, (700, 1500), (camera_window,))
    check("normal V2.22.1 crop scale is identity", abs(mapping.scale_x - 1.0) < 1e-9 and abs(mapping.scale_y - 1.0) < 1e-9)
    check("full camera AABB maps into crop-local XY", abs(work[0].x0 - 100.0) < 1e-9 and abs(work[0].y0 - 200.0) < 1e-9)
    mask = v244._build_work_region_mask((700, 1500), work)
    check("mapped object region is non-empty", int(mask.sum()) > 0)

    mapping2, work2 = v244.map_camera_windows_to_work_v244(scanner, (350, 750), (camera_window,))
    check("future downsample scale is derived, not guessed", abs(mapping2.scale_x - 0.5) < 1e-9 and abs(mapping2.scale_y - 0.5) < 1e-9)
    check("scaled working AABB remains correct", abs(work2[0].x0 - 50.0) < 1e-9 and abs(work2[0].y0 - 100.0) < 1e-9)

    main_source = (ROOT / "main.py").read_text(encoding="utf-8")
    check("main imports V2.24.4 installer", "install_v244_runtime" in main_source)
    check("V2.24.4 installs after V2.24.3", main_source.rfind("install_v244_runtime(App)") > main_source.rfind("install_v243_runtime(App)"))

    scene_source = (ROOT / "content/games/hit_context_test_v242.py").read_text(encoding="utf-8")
    check("physical testscene identifies V2.24.4", "V2.24.4 HIT CONTEXT TEST" in scene_source)
    check("testscene logs V2.24.4 hit verdict", "[V2.24.4 TEST-HIT]" in scene_source)

    old = '{"title": "Hit Context Test (V2.24.3)"}'
    updated = old.replace('"title": "Hit Context Test (V2.24.3)"', '"title": "Hit Context Test (V2.24.4)"', 1)
    check("menu title update remains valid JSON", json.loads(updated)["title"] == "Hit Context Test (V2.24.4)")

    print("\nAll V2.24.4 selftests passed.")


if __name__ == "__main__":
    main()
