from __future__ import annotations

from src.engine.ai.menu_extension import augment_menu

_bootstrapped = False


def apply_bootstrap() -> None:
    global _bootstrapped
    if _bootstrapped:
        return
    _patch_menu_loader()
    _patch_scene_factory()
    _patch_hit_scanner()
    _bootstrapped = True


def _patch_menu_loader() -> None:
    import src.engine.content_loader as content_loader

    original_load_menu = content_loader.load_menu

    def wrapped_load_menu(path):
        data = original_load_menu(path)
        return augment_menu(data)

    content_loader.load_menu = wrapped_load_menu


def _patch_scene_factory() -> None:
    import src.engine.scene_factory as scene_factory

    original_build_scene_from_item = scene_factory.build_scene_from_item

    def wrapped_build_scene_from_item(item, return_menu_state: dict | None = None):
        if item.type == "ai_settings":
            from src.engine.scenes.ai_settings import AISettingsScene

            return scene_factory._wrap(AISettingsScene(bg_color=item.bg_color), item, return_menu_state)
        if item.type == "ai_training":
            from src.engine.scenes.ai_training import AITrainingScene

            return scene_factory._wrap(AITrainingScene(bg_color=item.bg_color), item, return_menu_state)
        return original_build_scene_from_item(item, return_menu_state=return_menu_state)

    scene_factory.build_scene_from_item = wrapped_build_scene_from_item


def _patch_hit_scanner() -> None:
    from src.engine.camera.hit_scanner import HitScanner

    original_update = HitScanner.update
    original_emit = HitScanner._emit_track_result

    def wrapped_update(self: HitScanner, dt: float):
        result = original_update(self, dt)
        try:
            from src.engine.ai.runtime import get_ai_runtime

            runtime = get_ai_runtime()
            runtime.observe_scanner(self)
            self.candidate_limit = runtime.candidate_limit
        except Exception:
            pass
        return result

    def wrapped_emit(self: HitScanner, track, event):
        try:
            from src.engine.ai.runtime import get_ai_runtime

            runtime = get_ai_runtime()
            chosen = runtime.choose_for_emission(track.camera_x, track.camera_y)
            if chosen.get("apply"):
                old_x, old_y, old_score = track.camera_x, track.camera_y, track.best_score
                try:
                    track.camera_x = float(chosen["camera_x"])
                    track.camera_y = float(chosen["camera_y"])
                    track.best_score = max(track.best_score, float(chosen.get("confidence", 0.0)) * 10.0)
                    return original_emit(self, track, event)
                finally:
                    track.camera_x = old_x
                    track.camera_y = old_y
                    track.best_score = old_score
        except Exception:
            pass
        return original_emit(self, track, event)

    HitScanner.update = wrapped_update
    HitScanner._emit_track_result = wrapped_emit
