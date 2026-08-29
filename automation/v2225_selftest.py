from __future__ import annotations

import inspect
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.shot_fast_v2225 import (
    FastConfigV2225,
    SCHEMA_VERSION,
    fast_extract_candidates_v2225,
    local_confirm_candidates_v2225,
    rescue_router_v2225,
)


def check(cond: bool, label: str) -> None:
    if not cond:
        raise AssertionError(label)
    print(f"[PASS] {label}")


class FakeGenerator:
    @staticmethod
    def _refine_peak(**kwargs):
        return int(kwargs['px']), int(kwargs['py']), 0.0

    @staticmethod
    def _candidate_features(**kwargs):
        px = int(kwargs['px']); py = int(kwargs['py'])
        absdiff = kwargs['absdiff']; sal = kwargs['saliency']; z = kwargs['zscore']; dog = kwargs['dog']; dark = kwargs['darkening']
        return {
            'area': 4.0,
            'radius': 2.0,
            'circularity': 0.8,
            'score': float(max(3.6, sal[py, px] * 0.3)),
            'center_change': float(absdiff[py, px]),
            'local_contrast': float(absdiff[py, px]),
            'zscore': float(z[py, px]),
            'absdiff': float(absdiff[py, px]),
            'darkening': float(dark[py, px]),
            'dog_value': float(dog[py, px]),
        }

    @staticmethod
    def _apply_known_hole_penalty(scanner, candidate):
        return None


class FakeScanner:
    def __init__(self):
        self.last_window_debug = {}
        self.audio_events = []


def test_fast_extractor() -> None:
    h, w = 540, 960
    sal = np.zeros((h, w), np.float32)
    absdiff = np.zeros_like(sal)
    dark = np.zeros_like(sal)
    dog = np.zeros_like(sal)
    z = np.zeros_like(sal)
    valid = np.ones((h, w), bool)
    # Real compact peak + many weaker isolated artifacts.
    y, x = 231, 517
    sal[y, x] = 60.0; absdiff[y, x] = 18.0; dark[y, x] = 14.0; dog[y, x] = 5.0; z[y, x] = 8.0
    for i in range(120):
        yy = 20 + (i * 37) % (h - 40)
        xx = 20 + (i * 61) % (w - 40)
        sal[yy, xx] = max(sal[yy, xx], 12.0 + (i % 5))
        absdiff[yy, xx] = max(absdiff[yy, xx], 4.5)
        z[yy, xx] = max(z[yy, xx], 2.0)
    cfg = {
        'min_temporal_change': 1.8,
        'min_zscore': 1.5,
        'strong_temporal_change': 4.0,
        'local_max_kernel': 3,
        'nms_radius_px': 3.5,
        'tile_columns': 8,
        'tile_rows': 6,
        'per_tile_candidates': 7,
        'global_extra_candidates': 100,
        'max_v2_candidates': 220,
        'peak_refine_enabled': False,
    }
    fake = FakeGenerator(); scanner = FakeScanner()
    original_cc = cv2.connectedComponentsWithStats
    cv2.connectedComponentsWithStats = lambda *a, **k: (_ for _ in ()).throw(AssertionError('megapixel CC used'))
    try:
        t0 = time.perf_counter()
        out = fast_extract_candidates_v2225(
            fake, scanner=scanner, saliency=sal, absdiff=absdiff,
            darkening=dark, dog=dog, zscore=z, valid=valid,
            bbox=(100, 50, 100+w, 50+h), frame_ts=123.0,
            threshold=10.0, cfg=cfg,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
    finally:
        cv2.connectedComponentsWithStats = original_cc
    check(bool(out), 'FAST extractor emits candidates')
    nearest = min(out, key=lambda c: (c['camera_x']-(x+100))**2 + (c['camera_y']-(y+50))**2)
    check(abs(nearest['camera_x']-(x+100)) <= 1 and abs(nearest['camera_y']-(y+50)) <= 1, 'FAST extractor preserves strong peak XY')
    check(nearest.get('v2225_fast_extract') == 1.0, 'FAST candidate provenance is marked')
    check('v2225_fast_extract_ms' in scanner.last_window_debug, 'FAST extractor telemetry is recorded')
    # Generous smoke ceiling: catches accidental 1-second sleeps/whole-image Python loops,
    # but is not a hardware benchmark gate.
    check(elapsed < 800.0, 'FAST extractor smoke test stays bounded')


def test_local_confirm() -> None:
    h, w = 300, 500
    pre = np.full((h, w), 150, np.uint8)
    current = pre.copy()
    cx, cy = 244, 133
    cv2.circle(current, (cx, cy), 3, 118, -1)
    candidates = [
        {'camera_x': cx, 'camera_y': cy, 'score': 20.0, 'pre_shot_change': 12.0},
        {'camera_x': 80, 'camera_y': 80, 'score': 19.0, 'pre_shot_change': 10.0},
    ]
    cfg = FastConfigV2225(local_confirm_max_candidates=20)
    confirmed, diag = local_confirm_candidates_v2225(pre, current, candidates, frame_ts=5.0, config=cfg)
    check(len(confirmed) == 1, 'local confirm accepts new compact change and rejects unchanged old candidate')
    check(abs(confirmed[0]['camera_x']-cx) < 0.1 and abs(confirmed[0]['camera_y']-cy) < 0.1, 'local confirm never drags authoritative XY')
    check(confirmed[0].get('v2225_local_confirm') == 1.0, 'local confirmation provenance is marked')
    check(diag['tested'] == 2.0 and diag['confirmed'] == 1.0, 'local confirmation telemetry counts candidates')

    # Re-hit/hole-in-hole remains valid: pre may already contain an old dark hole,
    # but a new additional darkening at the same coordinate must still confirm.
    pre2 = np.full((h, w), 150, np.uint8)
    cv2.circle(pre2, (cx, cy), 3, 126, -1)
    post2 = pre2.copy()
    cv2.circle(post2, (cx, cy), 3, 104, -1)
    rehit, _ = local_confirm_candidates_v2225(pre2, post2, [candidates[0]], frame_ts=6.0, config=cfg)
    check(len(rehit) == 1, 're-hit/hole-in-hole with fresh temporal change remains legal')


def test_rescue_router() -> None:
    rescue_router_v2225.reset()
    check(rescue_router_v2225.request(7), 'full rescue can be requested once')
    check(not rescue_router_v2225.request(7), 'duplicate full rescue request is rejected')
    check(rescue_router_v2225.consume(7), 'worker can consume full rescue request')
    check(not rescue_router_v2225.consume(7), 'full rescue is not consumed twice')
    check(rescue_router_v2225.was_consumed(7), 'full rescue history is retained until shot cleanup')
    rescue_router_v2225.clear(7)


def test_files_and_entrypoint() -> None:
    main_text = (ROOT / 'main.py').read_text(encoding='utf-8')
    check('install_v2225_runtime' in main_text, 'main.py installs V2.22.5')
    check(main_text.index('install_v2224_runtime(App)') < main_text.index('install_v2225_runtime(App)'), 'V2.22.5 installs after V2.22.4')
    module_text = (ROOT / 'src/engine/shot_fast_v2225.py').read_text(encoding='utf-8')
    check('fast proposal + local confirmation installed' in module_text, 'runtime install banner exists')
    check('FULL-RESCUE' in module_text, 'high-recall rescue path remains explicit')
    check('local_confirm_candidates_v2225' in module_text, 'local confirmation API exists')


def main() -> None:
    print('V2.22.5 FAST PROPOSAL + LOCAL CONFIRMATION SELFTEST')
    print('===================================================')
    check(SCHEMA_VERSION == '2.22.5', 'schema is 2.22.5')
    test_fast_extractor()
    test_local_confirm()
    test_rescue_router()
    test_files_and_entrypoint()
    print('\nAll V2.22.5 selftests passed.')


if __name__ == '__main__':
    main()
