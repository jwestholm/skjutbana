"""Headless selftest for the AI-training scene test content.

  SDL_VIDEODRIVER=dummy python3 -m automation.ai_training_content_selftest

Covers: real_stamp holes darken multiplicatively (bright and dark background),
graceful fallback without stamps, every new background mode renders, and
"freeze at shot: off" keeps animated backgrounds moving after the shot.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import time
import unittest

import numpy as np
import pygame

from src.engine.synthetic import synthetic_hole_overlay as sho
from src.engine.synthetic.synthetic_hole_overlay import (
    REAL_STAMP_KIND,
    STAMP_NOMINAL_RADIUS_CAM_PX,
    SyntheticHoleOverlay,
    real_stamps_available,
)

NEW_MODES = ("images", "dark_scene", "scroll_landscape", "moving_targets")


def _composite(bg_value: int, radius_px: float, seed: int = 3):
    ov = SyntheticHoleOverlay(64, 64, rng_seed=seed)
    ov.add_hole(32, 32, kind=REAL_STAMP_KIND, radius_px=radius_px)
    frame = np.full((64, 64, 3), bg_value, np.uint8)
    out = ov.composite_on(frame).astype(np.float64)
    bgra = ov.render_overlay_bgra()
    return out, bgra


@unittest.skipUnless(real_stamps_available(), "real-hole stamp library not available")
class RealStampTests(unittest.TestCase):
    def test_multiplicative_on_bright_and_dark(self):
        for bg in (230, 40):
            out, bgra = _composite(bg, STAMP_NOMINAL_RADIUS_CAM_PX)  # scale 1 sprite
            self.assertEqual(int(bgra[..., :3].max()), 0, "real_stamp sprite must be black")
            d = bgra[..., 3].astype(np.float64) / 255.0
            mask = d > 0.05
            self.assertGreater(int(mask.sum()), 5)
            ratio = out[..., 0][mask] / bg
            expected = 1.0 - d[mask]
            tol = 1.0 / bg + 0.01  # uint8 quantisation
            self.assertLessEqual(float(np.abs(ratio - expected).max()), tol, f"bg={bg}")
            self.assertTrue(np.all(out[..., 0][~(d > 0)] == bg), "no change outside the hole")
            # measured hole model: darkened, never black
            self.assertGreater(float(ratio.min()), 0.35)
            self.assertLess(float(ratio.min()), 0.95)

    def test_co2_scale_still_darkens(self):
        out, bgra = _composite(200, STAMP_NOMINAL_RADIUS_CAM_PX * 0.2 / 2.0)
        self.assertGreater(int(bgra[..., 3].max()), 0)
        self.assertLess(float(out[..., 0].min()), 200.0)

    def test_fallback_without_stamps(self):
        key = str(sho.STAMP_LIBRARY_DIR)
        saved = sho._STAMP_CACHE.get(key, "missing")
        sho._STAMP_CACHE[key] = None
        try:
            ov = SyntheticHoleOverlay(64, 64, rng_seed=1)
            hid = ov.add_hole(32, 32, kind=REAL_STAMP_KIND, radius_px=2.0)
            bgra = ov.render_overlay_bgra()
            self.assertGreater(int(bgra[..., 3].max()), 0)
            self.assertEqual(ov._holes[hid].kind, "clean_hole")
            from src.engine.scenes.ai_training_content import AITrainingContent
            c = AITrainingContent()
            self.assertEqual(c.effective_hole_profile(), "legacy")
            self.assertNotEqual(c.choose_hole()[0], REAL_STAMP_KIND)
        finally:
            if saved == "missing":
                sho._STAMP_CACHE.pop(key, None)
            else:
                sho._STAMP_CACHE[key] = saved


class SceneContentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.screen = pygame.display.set_mode((800, 600))
        from src.engine.scenes.ai_training import AITrainingScene
        s = AITrainingScene()
        s.viewport = pygame.Rect(20, 10, 760, 580)
        s.font = pygame.font.Font(None, 28)
        s.small = pygame.font.Font(None, 22)
        s.tiny = pygame.font.Font(None, 18)
        cls.scene = s

    def _frame(self) -> np.ndarray:
        self.scene.render(self.screen)
        vp = self.scene.viewport
        return pygame.surfarray.array3d(self.screen)[vp.left + 5:vp.right - 5, vp.top + 60:vp.bottom - 60].copy()

    def test_defaults(self):
        c = self.scene._content
        self.assertEqual(c.effects_level, "off")
        self.assertTrue(c.freeze_at_shot)
        self.assertIn(c.hole_profile, ("co2", "airgun", "legacy"))
        self.assertEqual(self.scene.MODE_NAMES[:8], ["white", "white_grid", "coord_grid", "gray", "black",
                                                      "checker", "checker_anim", "bubbles"])

    def test_f2_is_blocked_without_verified_calibration(self):
        s = self.scene
        s.auto_training_enabled = False
        s._auto_cal_result = "VARNING: sparad kalibrering klarade inte markörkontrollen"
        s._toggle_auto_training(headless=True)
        self.assertFalse(s.auto_training_enabled)
        self.assertIn("Autoträning spärrad", s.status_message)

    def test_new_modes_render(self):
        s = self.scene
        for mode in NEW_MODES:
            s.bg_mode_index = s.MODE_NAMES.index(mode)
            t0 = time.perf_counter()
            frames = [self._frame() for _ in range(5)]
            ms = (time.perf_counter() - t0) / 5 * 1000.0
            self.assertGreater(float(frames[-1].std()), 3.0, f"{mode} renders content")
            print(f"  {mode}: {ms:.1f} ms/frame (incl. HUD + overlay composite)")
        s.bg_mode_index = s.MODE_NAMES.index("dark_scene")
        mean = float(self._frame().mean())
        self.assertTrue(30.0 <= mean <= 90.0, f"dark_scene mean {mean:.1f}")

    def _moves_after_shot(self, mode: str, freeze_at_shot: bool) -> bool:
        s = self.scene
        s.bg_mode_index = s.MODE_NAMES.index(mode)
        s._content.freeze_at_shot = freeze_at_shot
        s._animation_frozen = False
        self._frame()
        time.sleep(0.03)
        self._frame()
        s._animation_frozen = True  # the shot
        self._frame()
        a = self._frame()
        time.sleep(0.12)
        b = self._frame()
        s._animation_frozen = False
        s._content.freeze_at_shot = True
        return bool(np.any(a != b))

    def test_freeze_toggle(self):
        for mode in ("scroll_landscape", "moving_targets", "checker_anim"):
            self.assertFalse(self._moves_after_shot(mode, True), f"{mode} frozen with freeze on")
            self.assertTrue(self._moves_after_shot(mode, False), f"{mode} moves with freeze off")

    def test_effects_cycle_and_reveal(self):
        c = self.scene._content
        levels = [c.cycle_effects() for _ in range(4)]
        self.assertEqual(levels, ["marker", "marker+score", "all", "off"])
        c.effects_level = "all"
        score0 = c.score
        c.on_hole_revealed(400, 300)
        self.assertGreater(c.score, score0)
        self.assertEqual(len(c._markers), 1)
        s = self.scene
        s.bg_mode_index = s.MODE_NAMES.index("gray")
        self._frame()
        c.effects_level = "off"
        c._markers.clear()

    def test_hole_spec_profiles(self):
        c = self.scene._content
        k = c.camera_px_per_screen_px()
        self.assertTrue(0.25 <= k <= 8.0)
        if real_stamps_available():
            c.hole_profile = "co2_shrunk"
            kind, r = c.choose_hole()
            self.assertEqual(kind, REAL_STAMP_KIND)
            # projected footprint (calibrated, see PROJECTED_PROFILE) and optical gain
            self.assertTrue(STAMP_NOMINAL_RADIUS_CAM_PX * 0.55 / k - 1e-9 <= r <= STAMP_NOMINAL_RADIUS_CAM_PX * 0.65 / k + 1e-9)
            self.assertTrue(2.5 <= c.hole_strength() <= 4.0)
            c.hole_profile = "co2"
            kind, r = c.choose_hole()
            if sho.real_stamps_available(sho.AR15_STAMP_LIBRARY_DIR):
                # real AR15 stamps, true size, own library
                self.assertEqual(kind, sho.REAL_STAMP_AR15_KIND)
                self.assertTrue(STAMP_NOMINAL_RADIUS_CAM_PX * 0.95 / k - 1e-9 <= r <= STAMP_NOMINAL_RADIUS_CAM_PX / k + 1e-9)
            else:
                self.assertEqual(c.effective_hole_profile(), "co2_shrunk")
            c.hole_profile = "airgun"
            self.assertTrue(STAMP_NOMINAL_RADIUS_CAM_PX * 0.95 / k - 1e-9 <= c.choose_hole()[1] <= STAMP_NOMINAL_RADIUS_CAM_PX / k + 1e-9)
            self.assertTrue(2.5 <= c.hole_strength() <= 3.5)
        c.hole_profile = "legacy"
        self.assertNotEqual(c.choose_hole()[0], REAL_STAMP_KIND)
        c.hole_profile = "co2"
        self.assertIn("Frys vid skott", c.status_text())


if __name__ == "__main__":
    unittest.main(verbosity=1)
