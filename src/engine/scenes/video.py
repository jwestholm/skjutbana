from __future__ import annotations
import pygame

from config import SCREEN_WIDTH, SCREEN_HEIGHT
from src.engine.scene import Scene
from src.engine.video_player import VideoPlayer


class VideoScene(Scene):
    def __init__(self, movie_path: str) -> None:
        self.movie_path = movie_path
        self.player: VideoPlayer | None = None
        self.last_frame: pygame.Surface | None = None

    def on_enter(self) -> None:
        self.player = VideoPlayer(self.movie_path, (SCREEN_WIDTH, SCREEN_HEIGHT))

    def on_exit(self) -> None:
        if self.player:
            self.player.close()

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            if self.player:
                self.player.toggle_pause()
        return None

    def update(self, dt: float):
        if not self.player:
            return None

        frame = self.player.update(dt)
        if frame is not None:
            self.last_frame = frame

        # Om filmen tar slut: pausa på sista frame (sen kan du byta till meny här)
        if self.player.finished:
            self.player.pause()

        return None

    def render(self, screen: pygame.Surface) -> None:
        if self.last_frame:
            screen.blit(self.last_frame, (0, 0))
        else:
            screen.fill((0, 0, 0))