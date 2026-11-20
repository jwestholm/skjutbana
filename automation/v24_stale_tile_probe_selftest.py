"""Regression test: V24 tile probes from an earlier FULL extraction must not leak into later FAST frames.

In FAST mode `_extract_candidates` (the only place that refreshes `_v24_last_tile_probes`) is bypassed, but the
V24 `generate` wrapper still merges whatever probes it last stored.  This test drives the real V24 wrappers on a
minimal fake generator:
  * default (flag unset): documents the current behaviour - the stale probe IS merged into a later shot's frame;
  * SKJUTBANA_V24_FRESH_TILE_PROBES=1: probes are merged only on the frame they were measured on.
Run: python3 automation/v24_stale_tile_probe_selftest.py
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("v24_stale_selftest", ROOT / "src/engine/camera/detector_v24_extension.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Result(SimpleNamespace):
    pass


def _make(mod):
    class FakeGen:
        def __init__(self):
            self.next = []

        def _extract_candidates(self, *a, **k):
            return []

        def generate(self, *a, **k):
            return _Result(candidates=list(self.next), telemetry=dict(shot_id=k.get("shot_id", 0)))

        def reset_runtime_state(self):
            pass

        def flush_resolved_shots(self, *a, **k):
            return {}

        def record_funnel_evaluation(self, *a, **k):
            pass

        def record_empty_evaluation(self, *a, **k):
            pass

    mod.patch_candidate_generator_class(FakeGen)
    return FakeGen


def _run(flag: bool) -> int:
    if flag:
        os.environ["SKJUTBANA_V24_FRESH_TILE_PROBES"] = "1"
    else:
        os.environ.pop("SKJUTBANA_V24_FRESH_TILE_PROBES", None)
    mod = _load()
    Gen = _make(mod)
    g = Gen()
    g._v24_config = SimpleNamespace(snapshot=lambda: dict(enabled=True, tile_probe_reserved_slots=100,
                                                          tile_probe_nms_radius_px=3.5, accumulator_enabled=False))
    # a FULL extraction on shot 1 at t=100.0 left one probe far from everything
    g._v24_last_tile_probes = [dict(camera_x=500.0, camera_y=500.0, score=9.0, v24_patch_prior=0.9, timestamp=100.0)]
    g._v24_last_tile_probes_ts = 100.0
    # later FAST frame of shot 2 at t=105.0: generator returns one current candidate elsewhere
    g.next = [dict(camera_x=100.0, camera_y=100.0, score=5.0, timestamp=105.0)]
    out = g.generate(scanner=None, frame_ts=105.0, shot_id=2)
    return sum(1 for c in out.candidates if abs(float(c["camera_x"]) - 500.0) < 1e-6)


def test_default_documents_leak() -> None:
    assert _run(False) == 1, "expected current default behaviour to merge the stale probe"


def test_flag_blocks_stale_probe() -> None:
    assert _run(True) == 0, "stale tile probe leaked into a later frame with the fix enabled"


def test_flag_keeps_same_frame_probe() -> None:
    os.environ["SKJUTBANA_V24_FRESH_TILE_PROBES"] = "1"
    mod = _load()
    g = _make(mod)()
    g._v24_config = SimpleNamespace(snapshot=lambda: dict(enabled=True, tile_probe_reserved_slots=100,
                                                          tile_probe_nms_radius_px=3.5, accumulator_enabled=False))
    g._v24_last_tile_probes = [dict(camera_x=500.0, camera_y=500.0, score=9.0, v24_patch_prior=0.9)]
    g._v24_last_tile_probes_ts = 105.0
    g.next = [dict(camera_x=100.0, camera_y=100.0, score=5.0)]
    out = g.generate(scanner=None, frame_ts=105.0, shot_id=2)
    assert any(abs(float(c["camera_x"]) - 500.0) < 1e-6 for c in out.candidates), "same-frame probe was dropped"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    os.environ.pop("SKJUTBANA_V24_FRESH_TILE_PROBES", None)
