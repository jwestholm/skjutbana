"""Headless selftest for the AI-training scene test content.

  SDL_VIDEODRIVER=dummy python3 -m automation.ai_training_content_selftest

Covers: real_stamp holes darken multiplicatively (bright and dark background),
graceful fallback without stamps, every new background mode renders, and
"freeze at shot: off" keeps animated backgrounds moving after the shot.
"""
from __future__ import annotations

import os
import random

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

NEW_MODES = ("images", "dark_scene", "scroll_landscape", "moving_targets", "pastel_images")


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

    def test_ar15_exclusion_keeps_other_runtime_stamps(self):
        key = str(sho.AR15_STAMP_LIBRARY_DIR)
        saved_cache = sho._STAMP_CACHE.pop(key, "missing")
        old_env = os.environ.get("SKJUTBANA_EXCLUDE_AR15_STAMPS")
        try:
            os.environ["SKJUTBANA_EXCLUDE_AR15_STAMPS"] = "aim8a,aim10"
            stamps = sho.load_real_hole_stamps(sho.AR15_STAMP_LIBRARY_DIR)
            self.assertEqual(len(stamps or []), 6)
        finally:
            if old_env is None:
                os.environ.pop("SKJUTBANA_EXCLUDE_AR15_STAMPS", None)
            else:
                os.environ["SKJUTBANA_EXCLUDE_AR15_STAMPS"] = old_env
            sho._STAMP_CACHE.pop(key, None)
            if saved_cache != "missing":
                sho._STAMP_CACHE[key] = saved_cache


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

    def test_black_f2_enables_flash_only_while_headless_training(self):
        s = self.scene
        old = (s.auto_training_enabled, s.auto_headless, s.bg_mode_index)
        try:
            s.auto_training_enabled = True
            s.auto_headless = True
            s.bg_mode_index = s.MODE_NAMES.index("black")
            self.assertEqual(s.precision_flash, "on")
            s.bg_mode_index = s.MODE_NAMES.index("white")
            self.assertEqual(s.precision_flash, "auto")
            s.auto_training_enabled = False
            s.bg_mode_index = s.MODE_NAMES.index("black")
            self.assertEqual(s.precision_flash, "auto")
        finally:
            s.auto_training_enabled, s.auto_headless, s.bg_mode_index = old

    def test_flash_wrapper_redraws_black_f2_hole(self):
        from src.engine.scenes.overlay_scene import OverlayScene
        from src.engine.precision_flash import resolve_policy
        s = self.scene
        old = (s.auto_training_enabled, s.auto_headless, s.bg_mode_index,
               getattr(s, "synthetic_overlay", None))
        try:
            s.auto_training_enabled = True
            s.auto_headless = True
            s.bg_mode_index = s.MODE_NAMES.index("black")
            self.assertTrue(resolve_policy(OverlayScene(s))[0])
            overlay = SyntheticHoleOverlay(self.screen.get_width(), self.screen.get_height())
            overlay.add_hole(400, 300, kind="clean_hole", radius_px=12,
                             strength=1.0, opacity=1.0, rotation_deg=0.0)
            s.synthetic_overlay = overlay
            self.screen.fill((248, 248, 248))  # precision flash white frame
            OverlayScene(s).render_after_precision_flash(self.screen)
            pixels = pygame.surfarray.array3d(self.screen)
            self.assertLess(int(pixels[400, 300].mean()), 248)
        finally:
            (s.auto_training_enabled, s.auto_headless, s.bg_mode_index,
             s.synthetic_overlay) = old

    def test_f2_evidence_uses_camera_frames_after_projector_settle(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from src.engine.camera.camera_manager import camera_manager
        from src.engine.scenes.automation_ai_training import AutomationAITrainingScene
        frames = [
            SimpleNamespace(timestamp=10.20, frame_bgr=np.full((16, 16, 3), 100, np.uint8)),
            SimpleNamespace(timestamp=10.36, frame_bgr=np.full((16, 16, 3), 110, np.uint8)),
            SimpleNamespace(timestamp=10.40, frame_bgr=np.full((16, 16, 3), 120, np.uint8)),
        ]
        with patch.object(camera_manager, "get_ring_snapshot", return_value=frames):
            posts, source = AutomationAITrainingScene._camera_post_frames_after_reveal(10.0)
        self.assertEqual(source, "camera_ring_after_projector_settle")
        self.assertEqual([ts for _, ts in posts], [10.36, 10.40])
        self.assertEqual(posts[-1][0].shape, (16, 16))

    def test_black_f2_evidence_captures_frames_inside_flash_window(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from src.engine.camera.camera_manager import camera_manager
        from src.engine import precision_flash
        from src.engine.scenes.automation_ai_training import AutomationAITrainingScene
        frames = [
            SimpleNamespace(timestamp=10.35, frame_bgr=np.full((16, 16, 3), 240, np.uint8)),
            SimpleNamespace(timestamp=10.40, frame_bgr=np.full((16, 16, 3), 230, np.uint8)),
            SimpleNamespace(timestamp=10.45, frame_bgr=np.full((16, 16, 3), 220, np.uint8)),
            SimpleNamespace(timestamp=10.50, frame_bgr=np.full((16, 16, 3), 210, np.uint8)),
            SimpleNamespace(timestamp=10.70, frame_bgr=np.zeros((16, 16, 3), np.uint8)),
        ]
        fake_controller = SimpleNamespace(flash_windows=[(10.30, 10.50)],
                                          cfg=SimpleNamespace(receive_latency_s=0.06))
        with patch.object(camera_manager, "get_ring_snapshot", return_value=frames), \
             patch.object(precision_flash, "get_controller", return_value=fake_controller):
            posts, source = AutomationAITrainingScene._camera_post_frames_after_reveal(10.0)
        self.assertEqual(source, "precision_flash_window")
        self.assertEqual([ts for _, ts in posts], [10.40, 10.45, 10.50])

    def test_f2_capture_control_keeps_130_round_matrix(self):
        from src.engine.scenes.automation_ai_training import _capture_control_shot_count
        self.assertEqual(_capture_control_shot_count(130), 130)
        self.assertEqual(_capture_control_shot_count(10001), 10000)

    def test_f2_stays_overlay_free_after_stop(self):
        from unittest.mock import patch
        s = self.scene
        old = (s.auto_training_enabled, s.auto_headless, s.auto_report_visible)
        try:
            s.auto_training_enabled = False
            s.auto_headless = True
            s.auto_report_visible = True
            with patch.object(s, "_render_candidates") as draw_candidates:
                self._frame()
                draw_candidates.assert_not_called()
        finally:
            s.auto_training_enabled, s.auto_headless, s.auto_report_visible = old

    def test_headless_f2_is_single_shot_and_paced_for_camera(self):
        from unittest.mock import patch
        from src.engine.scenes.ai_training import (
            AUTO_HEADLESS_ITERATION_GAP_S,
            SYNTHETIC_MIN_SETTLE_SECONDS,
        )
        s = self.scene
        old = (s.auto_training_enabled, s.auto_headless, s.synthetic_trigger_batch_mode,
               getattr(s, "_live", None), s.synthetic_reveal_ts,
               s.synthetic_camera_frames_after_reveal)
        try:
            s.auto_training_enabled = True
            s.auto_headless = True
            s.synthetic_trigger_batch_mode = True
            s._live = dict(rounds=[{}], hits=[], burst_queue=[], subscribed=False)
            s._schedule_burst(time.time())
            self.assertEqual(s._live["burst_queue"], [])
            self.assertEqual(AUTO_HEADLESS_ITERATION_GAP_S, 3.0)
            self.assertEqual(s.synthetic_min_camera_frames_after_reveal, 12)
            self.assertEqual(SYNTHETIC_MIN_SETTLE_SECONDS, 1.0)
            s.synthetic_reveal_ts = 10.0
            s.synthetic_camera_frames_after_reveal = 12
            with patch.object(s, "_update_synthetic_camera_progress"):
                self.assertFalse(s._synthetic_camera_window_ready(10.99))
                self.assertTrue(s._synthetic_camera_window_ready(11.0))
        finally:
            (s.auto_training_enabled, s.auto_headless, s.synthetic_trigger_batch_mode, s._live,
             s.synthetic_reveal_ts, s.synthetic_camera_frames_after_reveal) = old

    def test_mixed_profile_uses_both_measured_libraries(self):
        from src.engine.scenes.ai_training_content import AITrainingContent
        c = AITrainingContent(hole_profile="mixed_co2_airgun")
        counts = {"co2": 0, "airgun": 0}
        rng = random.Random(20261124)
        for _ in range(1000):
            kind, _ = c.choose_hole(rng)
            counts[c.last_selected_profile] += 1
            if c.last_selected_profile == "co2":
                self.assertEqual(kind, "real_stamp_ar15")
            else:
                self.assertEqual(kind, "real_stamp")
        self.assertGreater(counts["co2"], 400)
        self.assertGreater(counts["airgun"], 400)

    def test_new_modes_render(self):
        s = self.scene
        for mode in NEW_MODES:
            s.bg_mode_index = s.MODE_NAMES.index(mode)
            t0 = time.perf_counter()
            frames = [self._frame() for _ in range(5)]
            ms = (time.perf_counter() - t0) / 5 * 1000.0
            self.assertGreater(float(frames[-1].std()), 3.0, f"{mode} renders content")
            print(f"  {mode}: {ms:.1f} ms/frame (incl. HUD + overlay composite)")
        s.auto_iteration = 0
        s.bg_mode_index = s.MODE_NAMES.index("pastel_images")
        pastel_mean = float(self._frame().mean())
        self.assertGreater(pastel_mean, 155.0, f"pastel image mean {pastel_mean:.1f}")
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
