from __future__ import annotations

import pygame

from src.engine.game_loader import load_game_module
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_viewport_rect


class GameScene(Scene):
    @property
    def shot_input_mode(self):
        return getattr(self.game, "shot_input_mode", "PHYSICAL_COORDINATE")

    @property
    def wants_hit_scanning(self):
        return self.shot_input_mode != "SHOT_ONLY"

    @property
    def hit_precision(self):
        # "exact" | "region" | "shot_only"; read live so a game may change it.
        return getattr(self.game, "hit_precision", None)

    @property
    def hit_aggregation(self):
        return getattr(self.game, "hit_aggregation", "per_burst")

    def on_precision_series(self, result):
        callback = getattr(self.game, "on_precision_series", None)
        if callable(callback):
            callback(result)

    @property
    def object_highlight(self):
        return getattr(self.game, "object_highlight", False) is True

    @property
    def precision_glow(self):
        return getattr(self.game, 'precision_glow', None)

    def get_precision_glow_context(self):
        callback=getattr(self.game, 'get_precision_glow_context', None)
        return callback() if callable(callback) else None

    def on_precision_glow_state(self, state):
        callback=getattr(self.game, 'on_precision_glow_state', None)
        if callable(callback):callback(state)

    @property
    def precision_flash(self):
        # "auto" | "off" | "on"; "off" always wins (see src/engine/precision_flash.py).
        return getattr(self.game, "precision_flash", "auto")

    @property
    def wants_mouse_simulated_hits(self):
        return self.shot_input_mode != "SHOT_ONLY"

    @property
    def shot_critical_updates_required(self):
        return bool(getattr(self.game, 'shot_critical_updates_required', False))

    @property
    def owns_hit_presentation(self):
        return bool(getattr(self.game, 'owns_hit_presentation', False))

    def on_object_shot_frozen(self, snapshot):
        callback = getattr(self.game, 'on_object_shot_frozen', None)
        if callable(callback): callback(snapshot)

    def on_frame_presented(self, timestamp):
        callback = getattr(self.game, 'on_frame_presented', None)
        if callable(callback): callback(timestamp)

    def __init__(self, game_root: str, script_path: str) -> None:
        self.game_root = game_root
        self.script_path = script_path
        self.viewport = None
        self.game = None

    def on_enter(self) -> None:
        self.viewport = load_viewport_rect()
        module = load_game_module(self.script_path)
        if not hasattr(module, "create_game"):
            raise AttributeError(
                f"{self.script_path} must define create_game(game_root, viewport)"
            )
        self.game = module.create_game(self.game_root, self.viewport)
        if hasattr(self.game, "on_enter"):
            self.game.on_enter()

    def on_exit(self) -> None:
        if self.game and hasattr(self.game, "on_exit"):
            self.game.on_exit()

    def get_hit_regions(self):
        """Return optional game-provided V2.24 hit-search regions.

        Games are not required to implement ``get_hit_regions``.  When absent,
        the engine returns an empty tuple and hit detection behaves exactly as
        before (global detector only).

        New games should return ``HitRegion`` instances from
        ``src.engine.input.hit_regions``.  Coordinates are viewport-local / game
        local, never camera coordinates.
        """
        if not self.game:
            return ()
        provider = getattr(self.game, "get_hit_regions", None)
        if not callable(provider):
            return ()
        values = provider()
        return () if values is None else values

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            from src.engine.scenes.menu import MenuScene
            return SceneSwitch(MenuScene())

        if self.game and hasattr(self.game, "handle_event"):
            result = self.game.handle_event(event)
            if result is not None:
                return result
        return None

    def update(self, dt: float):
        if self.game and hasattr(self.game, "update"):
            result = self.game.update(dt)
            if result is not None:
                return result
        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.fill((0, 0, 0))
        if not self.game:
            return

        old_clip = screen.get_clip()
        screen.set_clip(self.viewport)
        if hasattr(self.game, "render"):
            self.game.render(screen)
        screen.set_clip(old_clip)
