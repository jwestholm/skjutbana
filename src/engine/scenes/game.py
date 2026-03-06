from __future__ import annotations

import pygame

from src.engine.game_loader import load_game_module
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_viewport_rect


class GameScene(Scene):
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