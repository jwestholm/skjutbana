from __future__ import annotations

from src.engine.content_loader import MenuItem


def build_scene_from_item(item: MenuItem):
    if item.type == "video":
        from src.engine.scenes.video import VideoScene
        return VideoScene(item.path, fit=item.fit, bg_color=item.bg_color)

    if item.type == "image":
        from src.engine.scenes.image import ImageScene
        return ImageScene(item.path, fit=item.fit, bg_color=item.bg_color)

    if item.type == "settings":
        from src.engine.scenes.calibrate import CalibrateViewportScene
        return CalibrateViewportScene()

    if item.type == "game":
        from src.engine.scenes.game import GameScene
        return GameScene(game_root=item.path, script_path=item.script)

    raise ValueError(f"Unknown item type: {item.type}")