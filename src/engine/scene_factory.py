from __future__ import annotations

from src.engine.content_loader import MenuItem
from src.engine.scenes.overlay_scene import OverlayScene


def _wrap(scene):
    return OverlayScene(scene)


def build_scene_from_item(item: MenuItem):
    if item.type == "video":
        from src.engine.scenes.video import VideoScene
        return _wrap(VideoScene(item.path, fit=item.fit, bg_color=item.bg_color))

    if item.type == "image":
        from src.engine.scenes.image import ImageScene
        return _wrap(ImageScene(item.path, fit=item.fit, bg_color=item.bg_color))

    if item.type == "game":
        from src.engine.scenes.game import GameScene
        return _wrap(GameScene(game_root=item.path, script_path=item.script))

    if item.type == "transform_debug":
        from src.engine.scenes.transform_debug import TransformDebugScene
        return _wrap(TransformDebugScene(bg_color=item.bg_color))

    if item.type == "settings":
        from src.engine.scenes.calibrate import CalibrateViewportScene
        return CalibrateViewportScene()

    if item.type == "camera_scanport":
        from src.engine.scenes.camera_test import CameraTestScene
        return CameraTestScene(bg_color=item.bg_color)

    if item.type == "visual_hits_settings":
        from src.engine.scenes.visual_hits_settings import VisualHitsSettingsScene
        return VisualHitsSettingsScene(bg_color=item.bg_color)

    if item.type == "scanner_debug_settings":
        from src.engine.scenes.scanner_debug_settings import ScannerDebugSettingsScene
        return ScannerDebugSettingsScene(bg_color=item.bg_color)

    raise ValueError(f"Unknown item type: {item.type}")