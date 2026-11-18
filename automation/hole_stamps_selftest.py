"""Selftest for src/engine/offline/hole_stamps.py and the stamp hole model.

  python3 -m automation.hole_stamps_selftest
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.engine.offline.hole_stamps import StampSet, clean_one


def _ratio_with(hole_depth=0.35, line=True, other_hole=True, noise=0.005, seed=0):
    rng = np.random.default_rng(seed)
    r = np.ones((41, 41), np.float32)
    yy, xx = np.mgrid[0:41, 0:41]
    r -= hole_depth * np.exp(-((yy - 20) ** 2 + (xx - 20) ** 2) / (2 * 1.8 ** 2))
    if line:                                   # projected edge passing through the hole
        r[20, :] -= 0.15
    if other_hole:                             # unrelated dark structure at the edge
        r -= 0.4 * np.exp(-((yy - 5) ** 2 + (xx - 35) ** 2) / (2 * 1.5 ** 2))
    return r + rng.normal(0, noise, r.shape).astype(np.float32)


class CleanTests(unittest.TestCase):
    def test_only_the_hole_remains(self):
        d, info = clean_one(_ratio_with(), 0.005)
        self.assertEqual(info["reason"], "OK")
        self.assertGreater(d[20, 20], 0.2)
        self.assertEqual(float(d[5, 35]), 0.0)          # other hole removed
        self.assertEqual(float(d[20, 2]), 0.0)          # line far from the hole removed
        self.assertEqual(float(d[20, 38]), 0.0)
        self.assertEqual(float(d[:8].max()), 0.0)       # background exactly zero
        self.assertGreaterEqual(float(d.min()), 0.0)

    def test_weak_or_pre_existing_hole_rejected(self):
        d, info = clean_one(_ratio_with(hole_depth=0.03, line=False, other_hole=False), 0.005)
        self.assertIsNone(d)
        self.assertTrue(info["reason"].startswith("TOO_WEAK"))


class RenderTests(unittest.TestCase):
    def test_stamp_patch_scaled_and_multiplicative(self):
        d, _ = clean_one(_ratio_with(line=False, other_hole=False), 0.005)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.npz"
            np.savez(path, stamps=np.stack([d]), id=np.array(["x"]))
            stamps = StampSet(path)
            full, _ = stamps.patch(np.random.default_rng(1), 1.0)
            half, _ = stamps.patch(np.random.default_rng(1), 0.5)
            self.assertLess(half.shape[0], full.shape[0])
            self.assertLess(float(half.max()), 0.96)
            # multiplicative: the same hole darkens a dark board less in grey levels
            bright = 200.0 * (1 - full.max())
            dark = 100.0 * (1 - full.max())
            self.assertAlmostEqual((200.0 - bright) / (100.0 - dark), 2.0, places=4)

    def test_frame_renderer_uses_stamps(self):
        from src.engine.offline.ar15_scenarios import FrameRenderer, PhysicsAssumptions, Scenario, Shot
        d, _ = clean_one(_ratio_with(line=False, other_hole=False), 0.005)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.npz"
            np.savez(path, stamps=np.stack([d]), id=np.array(["x"]))
            phys = PhysicsAssumptions(hole_model="stamps", hole_stamp_path=str(path), sensor_noise_sigma=0.0,
                                      jpeg_quality=None, hole_stamp_scale=(0.5, 0.5))

            class _Geo:
                def px_per_mm(self, *_):
                    return 1.3
            scen = Scenario.__new__(Scenario)
            scen.shots = [Shot(fire_t=0.0, impact_t=0.1, camera_x=50.3, camera_y=40.7, diameter_mm=2.5)]
            r = FrameRenderer(np.full((100, 100), 150, np.uint8), scen, _Geo(), phys, seed=1)
            before = r.render(0.05)[:, :, 0].astype(float)
            after = r.render(1.0)[:, :, 0].astype(float)
            self.assertTrue(np.all(before == 150))
            ys, xs = np.unravel_index(np.argmin(after), after.shape)
            self.assertLess(after.min(), 150 * 0.8)
            self.assertLessEqual(abs(xs - 50.3), 1.5)
            self.assertLessEqual(abs(ys - 40.7), 1.5)
            self.assertEqual(r.stamp_ids, ["x"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
