from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import time

from src.engine.input.object_hit_v2223 import ObjectHitRegistryV2223, viewport_center_prior
from src.engine.shot_critical_v2223 import ShotCriticalControllerV2223, select_recent_pre_frame_v2223


@dataclass
class Frame:
    timestamp: float
    frame_bgr: object = None


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"[PASS] {name}")


def main() -> None:
    print("V2.22.3 SHOT-CRITICAL / OBJECT-HIT SELFTEST")
    print("===========================================")

    peak = 100.0
    ring = [Frame(99.10), Frame(99.40), Frame(99.64), Frame(99.80), Frame(99.95), Frame(100.02)]
    best = select_recent_pre_frame_v2223(ring, peak, target_offset_s=0.35, latest_safe_offset_s=0.08)
    check("recent PRE selects a safe timestamped frame before the shot", best is not None and abs(best.timestamp - 99.64) < 1e-6)
    check("recent PRE never leaks a post/too-late frame", best.timestamp <= peak - 0.08)

    centre, edge = viewport_center_prior(500, 500, (0, 0, 1000, 1000))
    outer_centre, outer_edge = viewport_center_prior(990, 500, (0, 0, 1000, 1000))
    check("viewport centre prior is stronger in the middle", centre > outer_centre)
    check("edge-distance prior is weaker near the edge", edge > outer_edge)

    reg = ObjectHitRegistryV2223()
    reg.register_rect("target", (100, 100, 200, 100), metadata={"kind": "enemy"})
    snap = reg.snapshot(7, 123.4)
    check("object hit regions are frozen per shot", len(snap.regions) == 1 and snap.regions[0].object_id == "target")

    candidates = [
        {"camera_x": 20.0, "camera_y": 20.0, "score": 9.0, "pre_shot_change": 8.0},
        {"camera_x": 150.0, "camera_y": 140.0, "score": 7.0, "pre_shot_change": 6.0, "local_contrast": 4.0},
    ]
    results = reg.evaluate_candidates(
        7,
        candidates,
        camera_to_screen=lambda x, y: (x, y),
        viewport_rect_xywh=(0, 0, 1000, 1000),
    )
    check("object shadow selects a candidate inside the frozen object", len(results) == 1 and results[0].hit and results[0].candidate_rank == 2)
    check("objects can ask was_hit() without owning OpenCV", reg.was_hit("target", 7))
    check("object result keeps local hit coordinates", 0.0 <= results[0].local_x <= 1.0 and 0.0 <= results[0].local_y <= 1.0)

    class FakeAudio:
        last_peak_ts = 10.0

    ctl = ShotCriticalControllerV2223()
    ctl.latency_cursor_enabled = False
    check("audio pending is detected before ordinary engine work", ctl.pending_audio(FakeAudio()))
    ctl.last_seen_peak_ts = 10.0
    check("same audio event is not re-dispatched", not ctl.pending_audio(FakeAudio()))

    root = Path(__file__).resolve().parents[1]
    main_py = (root / "main.py").read_text(encoding="utf-8")
    installer_pos = main_py.find("install_v2223_runtime(App)")
    run_pos = main_py.find("App().run()")
    check("main.py installs shot-critical policy before App().run", installer_pos >= 0 and run_pos > installer_pos)

    # Static source guard: V2.22.3 must not regress into per-frame capability probing.
    source = (root / "src/engine/shot_critical_v2223.py").read_text(encoding="utf-8")
    check("camera fast update does not consume HitScanner pickup cursor", "self._last_pickup_count = self._read_count" not in source)
    check("camera capability probe is moved to explicit slow path", "refresh_capabilities_v2223" in source)
    run_src = source[source.find("def run_v2223"):source.find("AppClass.run = run_v2223")]
    check("game loop does not auto-refresh camera capabilities", "camera_manager.refresh_capabilities_v2223(" not in run_src)
    check("audio dispatch appears before camera update in shot-critical loop", run_src.find("audio_peak_detector.update()") < run_src.find("camera_manager.update()"))
    check("HitScanner is serviced before ordinary scene update", run_src.find("hit_scanner.update(dt)") < run_src.find("self.scene.update(dt)"))
    check("object engine is shadow-only in V2.22.3", "SHADOW-only" in (root / "src/engine/input/object_hit_v2223.py").read_text(encoding="utf-8"))

    print("\nAll V2.22.3 selftests passed.")


if __name__ == "__main__":
    main()
