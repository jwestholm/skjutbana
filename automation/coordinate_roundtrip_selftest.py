"""Exercise existing coordinate APIs with explicit fixtures, no capture calibration claims."""
from contextlib import ExitStack
from importlib import import_module
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pygame

from src.engine.camera.analysis_geometry_v2221 import build_perspective_geometry_v2221
from src.engine.camera.camera_manager import CameraManager
from src.engine.input.object_hit_v2223 import (GameHitRegionV240, transform_game_rect_to_camera_aabb,
                                              transform_camera_point_to_game)
from src.engine.shot_object_local_v244 import map_camera_windows_to_work_v244
from src.engine.shot_object_local_v241 import LocalSearchWindowV241

hi = import_module('src.engine.input.hit_input')


class Tests(unittest.TestCase):
    def mapper(self, viewport, content, matrix, *, homography=True):
        stack = ExitStack(); self.addCleanup(stack.close)
        calibration = dict(method='aruco_viewport_board', homography=matrix.tolist()) if homography else {}
        stack.enter_context(patch.object(hi, 'load_camera_calibration', return_value=calibration))
        stack.enter_context(patch.object(hi, 'load_viewport_rect', return_value=viewport))
        stack.enter_context(patch.object(hi, 'load_content_rect', return_value=content))
        stack.enter_context(patch.object(hi, 'load_scanport_rect', return_value=pygame.Rect(1500, 900, 1400, 800)))
        return hi.HitInput()

    def test_camera_game_camera_perspective_and_letterboxing(self):
        # Two explicitly calibrated render sizes, including a projector-sized
        # fixture. This tests geometry, not a nonexistent fullscreen UI mode.
        for width, height in ((1024, 768), (1920, 1080)):
            with self.subTest(size=(width, height)):
                viewport = pygame.Rect(70, 45, width-140, height-90)
                content = pygame.Rect(40, 30, viewport.w-80, viewport.h-60)
                camera = np.float32([[1510, 970], [2900, 1030], [2850, 1700], [1570, 1750]])
                screen = np.float32([[viewport.x, viewport.y], [viewport.right, viewport.y],
                                     [viewport.right, viewport.bottom], [viewport.x, viewport.bottom]])
                h = cv2.getPerspectiveTransform(camera, screen)
                mapper = self.mapper(viewport, content, h)
                for xy in ((1800.25, 1200.75), (2400.125, 1550.5), (2100., 1300.)):
                    event = mapper._build_event_from_camera(source='camera', camera_x=xy[0], camera_y=xy[1])
                    back = mapper._canonical_screen_to_camera(event.game_x+viewport.x, event.game_y+viewport.y)
                    np.testing.assert_allclose(back, xy, atol=.002, rtol=0)
                    self.assertAlmostEqual(event.content_x, event.game_x-content.x)
                    self.assertAlmostEqual(event.content_norm_y, (event.game_y-content.y)/content.h)
                    game = transform_camera_point_to_game(*xy, tuple(viewport), mapper._canonical_camera_to_screen)
                    np.testing.assert_allclose(game, (event.game_x, event.game_y), atol=.002, rtol=0)

    def test_explicit_rectangular_board_fixture_normalized_round_trips(self):
        # The existing scanport normalization can represent a supplied rectangular
        # board fixture; it is not evidence that a real board boundary was saved.
        vp = pygame.Rect(70, 40, 1000, 600)
        mapper = self.mapper(vp, pygame.Rect(0, 0, vp.w, vp.h), np.eye(3), homography=False)
        for bx, by in ((0., 0.), (.17, .83), (.5, .5), (1., 1.)):
            camera = (1500+1400*bx, 900+800*by)
            _, _, u, v = mapper._camera_to_scanport(*camera)
            np.testing.assert_allclose((u, v), (bx, by), atol=1e-12)
            screen = mapper._canonical_camera_to_screen(*camera)
            back = mapper._canonical_screen_to_camera(*screen)
            np.testing.assert_allclose(back, camera, atol=1e-9)
            event = mapper._build_event_from_camera(source='camera', camera_x=camera[0], camera_y=camera[1])
            np.testing.assert_allclose((event.game_x/vp.w, event.game_y/vp.h), (bx, by), atol=1e-12)

    def test_crop_roundtrip_and_scaled_working_region(self):
        inverse = np.array([[1.2, .07, 1300], [.03, .9, 700], [.00001, .00002, 1]], np.float32)
        geom = build_perspective_geometry_v2221((2160, 3840), inverse, (80, 50, 1100, 700))
        for point in ((1600.25, 1100.75), (2400., 1500.)):
            np.testing.assert_allclose(geom.local_to_camera(*geom.camera_to_local(*point)), point, atol=1e-10)
        region = LocalSearchWindowV241(1700, 1200, 1800, 1300, ('front', 'rear'), ('target',))
        mapping, work = map_camera_windows_to_work_v244(SimpleNamespace(_v2221_active_geometry=geom),
                                                       (geom.crop_height//2, geom.crop_width//2), (region,))
        recovered = (work[0].x0/mapping.scale_x+mapping.crop_x0, work[0].y0/mapping.scale_y+mapping.crop_y0)
        np.testing.assert_allclose(recovered, (1700, 1200), atol=1e-9)
        self.assertGreater(np.count_nonzero(geom.safe_mask_local), 0)

    def test_four_corner_hitregion_projection(self):
        vp = pygame.Rect(80, 50, 1000, 600)
        h = np.array([[.7, .08, -700], [.02, .7, -500], [.00003, .00002, 1]], np.float32)
        mapper = self.mapper(vp, pygame.Rect(0, 0, 1000, 600), h)
        region = GameHitRegionV240('object', 100, 200, 40, 30)
        aabb = transform_game_rect_to_camera_aabb(region, tuple(vp), mapper._canonical_screen_to_camera)
        for x, y in ((100, 200), (140, 200), (140, 230), (100, 230)):
            cx, cy = mapper._canonical_screen_to_camera(x+vp.x, y+vp.y)
            self.assertTrue(aabb.x-.002 <= cx <= aabb.x+aabb.width+.002)
            self.assertTrue(aabb.y-.002 <= cy <= aabb.y+aabb.height+.002)

    def test_camera_orientation_inverse_for_all_supported_rotations_and_mirrors(self):
        raw = np.arange(7*11*3, dtype=np.uint8).reshape(7, 11, 3)
        camera = CameraManager.__new__(CameraManager)
        for rotation in (0, 90, 180, 270):
            for horizontal, vertical in ((False, False), (True, False), (False, True), (True, True)):
                camera.rotation, camera.mirror_horizontal, camera.mirror_vertical = rotation, horizontal, vertical
                oriented = camera._apply_frame_transform(raw)
                inverse = oriented
                if horizontal:
                    inverse = cv2.flip(inverse, 1)
                if vertical:
                    inverse = cv2.flip(inverse, 0)
                if rotation:
                    code = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}[rotation]
                    inverse = cv2.rotate(inverse, code)
                np.testing.assert_array_equal(inverse, raw)


if __name__ == '__main__':
    unittest.main()
