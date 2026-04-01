from __future__ import annotations

from src.engine.content_loader import MenuItem
from src.engine.scenes.overlay_scene import OverlayScene


def _attach_led_config(scene, item: MenuItem):
    setattr(scene, "scene_led_enabled", bool(item.led_enabled))
    setattr(scene, "scene_led_color", tuple(item.led_color))
    return scene


def _attach_return_menu_state(scene, return_menu_state):
    setattr(scene, "return_menu_state", return_menu_state)
    inner = getattr(scene, "inner", None)
    if inner is not None:
        setattr(inner, "return_menu_state", return_menu_state)
    return scene


def _wrap(scene, item: MenuItem, return_menu_state=None):
    wrapped = OverlayScene(scene)
    wrapped = _attach_led_config(wrapped, item)
    wrapped = _attach_return_menu_state(wrapped, return_menu_state)
    return wrapped


def build_scene_from_item(item: MenuItem, return_menu_state: dict | None = None):
    if item.type == "video":
        from src.engine.scenes.video import VideoScene

        return _wrap(
            VideoScene(item.path, fit=item.fit, bg_color=item.bg_color),
            item,
            return_menu_state=return_menu_state,
        )

    if item.type == "image":
        from src.engine.scenes.image import ImageScene

        return _wrap(
            ImageScene(item.path, fit=item.fit, bg_color=item.bg_color),
            item,
            return_menu_state=return_menu_state,
        )

    if item.type == "game":
        from src.engine.scenes.game import GameScene

        return _wrap(
            GameScene(game_root=item.path, script_path=item.script),
            item,
            return_menu_state=return_menu_state,
        )

    if item.type == "transform_debug":
        from src.engine.scenes.transform_debug import TransformDebugScene

        return _wrap(
            TransformDebugScene(bg_color=item.bg_color),
            item,
            return_menu_state=return_menu_state,
        )

    if item.type == "settings":
        from src.engine.scenes.calibrate import CalibrateViewportScene

        return _wrap(
            CalibrateViewportScene(),
            item,
            return_menu_state=return_menu_state,
        )

    if item.type == "camera_scanport":
        from src.engine.scenes.camera_test import CameraTestScene

        return _wrap(
            CameraTestScene(bg_color=item.bg_color),
            item,
            return_menu_state=return_menu_state,
        )

    if item.type == "camera_orientation_settings":
        from src.engine.scenes.camera_orientation_settings import (
            CameraOrientationSettingsScene,
        )

        return _wrap(
            CameraOrientationSettingsScene(),
            item,
            return_menu_state=return_menu_state,
        )

    if item.type == "scanport_preview":
        from src.engine.scenes.scanport_preview import ScanportPreview

        return _wrap(
            ScanportPreview(bg_color=item.bg_color),
            item,
            return_menu_state=return_menu_state,
        )

    if item.type == "visual_hits_settings":
        from src.engine.scenes.visual_hits_settings import VisualHitsSettingsScene

        return _wrap(
            VisualHitsSettingsScene(bg_color=item.bg_color),
            item,
            return_menu_state=return_menu_state,
        )

    if item.type == "scanner_debug_settings":
        from src.engine.scenes.scanner_debug_settings import ScannerDebugSettingsScene

        return _wrap(
            ScannerDebugSettingsScene(bg_color=item.bg_color),
            item,
            return_menu_state=return_menu_state,
        )

    if item.type == "audio_peak_settings":
        from src.engine.scenes.audio_peak_settings import AudioPeakSettingsScene

        return _wrap(
            AudioPeakSettingsScene(bg_color=item.bg_color),
            item,
            return_menu_state=return_menu_state,
        )

    if item.type == "physical_setup_settings":
        from src.engine.scenes.physical_setup_settings import PhysicalSetupSettingsScene

        return _wrap(
            PhysicalSetupSettingsScene(),
            item,
            return_menu_state=return_menu_state,
        )

    if item.type == "led_settings":
        from src.engine.scenes.led_settings import LedSettingsScene

        return _wrap(
            LedSettingsScene(bg_color=item.bg_color),
            item,
            return_menu_state=return_menu_state,
        )

    raise ValueError(f"Unknown item type: {item.type}")