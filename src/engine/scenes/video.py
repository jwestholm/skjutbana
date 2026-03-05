from __future__ import annotations
import pygame


from config import SCREEN_WIDTH, SCREEN_HEIGHT
from src.engine.scene import Scene
from src.engine.video_player import VideoPlayer
from src.engine.scenes.menu import MenuScene
from src.engine.scene import SceneSwitch
from src.engine.settings import load_viewport_rect

class VideoScene(Scene):
    def __init__(self, movie_path: str) -> None:
        self.movie_path = movie_path
        self.player: VideoPlayer | None = None
        self.last_frame: pygame.Surface | None = None
        self.viewport = None
        
    def on_enter(self) -> None:
        self.viewport = load_viewport_rect()
        self.player = VideoPlayer(self.movie_path, (self.viewport.w, self.viewport.h))
    
    def on_exit(self) -> None:
        if self.player:
            self.player.close()

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            if self.player:
                self.player.toggle_pause()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return SceneSwitch(MenuScene())
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
        screen.fill((0, 0, 0))
        if self.last_frame:
            screen.blit(self.last_frame, (self.viewport.x, self.viewport.y))