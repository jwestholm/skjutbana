from __future__ import annotations
from dataclasses import dataclass
import pygame


@dataclass
class SceneSwitch:
    """Returneras av en Scene för att byta scen."""
    new_scene: "Scene"


class Scene:
    """Bas-klass för alla scener."""
    def on_enter(self) -> None:
        pass

    def on_exit(self) -> None:
        pass

    def handle_event(self, event: pygame.event.Event) -> SceneSwitch | None:
        return None

    def update(self, dt: float) -> SceneSwitch | None:
        return None

    def render(self, screen: pygame.Surface) -> None:
        pass