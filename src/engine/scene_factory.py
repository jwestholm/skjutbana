from __future__ import annotations

from src.engine.content_loader import MenuItem


def build_scene_from_item(item: MenuItem):
    if item.type == "video":
        from src.engine.scenes.video import VideoScene
        return VideoScene(item.path)

    if item.type == "image":
        from src.engine.scenes.image import ImageScene
        return ImageScene(item.path)

    raise ValueError(f"Unknown item type: {item.type}")