from __future__ import annotations

from src.engine.ai.menu_extension import augment_menu

_bootstrapped = False


def apply_bootstrap() -> None:
    global _bootstrapped

    if _bootstrapped:
        return

    _patch_menu_loader()
    _patch_scene_factory()
    _patch_ranker_v6()
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

            return scene_factory._wrap(
                AISettingsScene(bg_color=item.bg_color),
                item,
                return_menu_state,
            )

        if item.type == "ai_training":
            from src.engine.scenes.ai_training import AITrainingScene

            return scene_factory._wrap(
                AITrainingScene(bg_color=item.bg_color),
                item,
                return_menu_state,
            )

        if item.type == "ai_results":
            from src.engine.scenes.ai_results import AIResultsScene

            return scene_factory._wrap(
                AIResultsScene(bg_color=item.bg_color),
                item,
                return_menu_state,
            )

        return original_build_scene_from_item(
            item,
            return_menu_state=return_menu_state,
        )

    scene_factory.build_scene_from_item = wrapped_build_scene_from_item


def _patch_ranker_v6() -> None:
    # V2.7 deliberately does NOT install the older V4/V5 wrappers. Their files
    # may remain on disk for history/rollback, but a fresh process gets exactly
    # one ranking extension: filtered observations -> hypotheses -> V6.
    try:
        from src.engine.ai.ranker_v6_extension import install_ranker_v6_extension

        install_ranker_v6_extension()
    except Exception as exc:
        print(f"[RANKER-V6] unavailable, base ranker kept: {exc}")


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
            # AI remains fail-open: a diagnostics/runtime problem must never stop
            # the ordinary detector from updating.
            pass

        return result

    def wrapped_emit(self: HitScanner, track, event):
        """Synchronize the exact shot before AI is allowed to override it."""
        runtime = None
        chosen = {"apply": False}
        shot_id = int(getattr(event, "shot_id", 0) or 0)

        try:
            from src.engine.ai.runtime import get_ai_runtime

            runtime = get_ai_runtime()
            runtime.observe_scanner(self, event=event)
            chosen = runtime.choose_for_emission(
                track.camera_x,
                track.camera_y,
                shot_id=shot_id,
            )
        except Exception:
            chosen = {"apply": False}

        if chosen.get("apply"):
            old_x, old_y, old_score = (
                track.camera_x,
                track.camera_y,
                track.best_score,
            )
            try:
                track.camera_x = float(chosen["camera_x"])
                track.camera_y = float(chosen["camera_y"])
                track.best_score = max(
                    track.best_score,
                    float(chosen.get("confidence", 0.0)) * 10.0,
                )
                result = original_emit(self, track, event)
            finally:
                track.camera_x = old_x
                track.camera_y = old_y
                track.best_score = old_score
        else:
            result = original_emit(self, track, event)

        if runtime is not None:
            try:
                runtime.mark_shot_finished(
                    shot_id,
                    state=str(getattr(event, "state", "finished") or "finished"),
                )
            except Exception:
                pass

        return result

    HitScanner.update = wrapped_update
    HitScanner._emit_track_result = wrapped_emit
