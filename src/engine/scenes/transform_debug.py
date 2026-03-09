from __future__ import annotations

import pygame

from src.engine.input.hit_input import HitEvent, hit_input
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_camera_calibration, load_viewport_rect


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT = (220, 220, 220)
MID = (180, 180, 180)
DARK = (120, 120, 120)
RED = (255, 60, 60)
CYAN = (80, 220, 255)
YELLOW = (255, 220, 80)
PANEL_BG = (0, 0, 0, 170)


class TransformDebugScene(Scene):
    """
    Grid / transform-test.

    Kontroller:
    - Mus vänsterknapp i viewporten = simulera träff via global hit_input
    - C = rensa senaste träffinfo
    - ESC = tillbaka
    """

    def __init__(self, bg_color=(0, 0, 0)) -> None:
        self.bg_color = bg_color
        self.viewport = None
        self.grid_surface = None

        self.font = None
        self.small = None
        self.tiny = None

        self.last_hit: HitEvent | None = None
        self.status_message = ""

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 26)
        self.tiny = pygame.font.Font(None, 24)

        self.viewport = load_viewport_rect()
        self.grid_surface = self._build_grid(self.viewport.w, self.viewport.h)

        self.last_hit = None

        calibration = load_camera_calibration()
        if calibration and calibration.get("is_calibrated"):
            self.status_message = "Kamerakalibrering hittad."
        else:
            self.status_message = "Ingen kamerakalibrering hittad. Mus-test fungerar ändå."

    def _build_grid(self, width: int, height: int) -> pygame.Surface:
        surface = pygame.Surface((width, height))
        surface.fill(WHITE)

        font = pygame.font.Font(None, 18)

        for x in range(0, width + 1, 10):
            color = LIGHT
            thickness = 1
            if x % 100 == 0:
                color = DARK
                thickness = 2
            elif x % 50 == 0:
                color = MID
            pygame.draw.line(surface, color, (x, 0), (x, height), thickness)

        for y in range(0, height + 1, 10):
            color = LIGHT
            thickness = 1
            if y % 100 == 0:
                color = DARK
                thickness = 2
            elif y % 50 == 0:
                color = MID
            pygame.draw.line(surface, color, (0, y), (width, y), thickness)

        for x in range(0, width, 100):
            for y in range(0, height, 100):
                pygame.draw.circle(surface, CYAN, (x, y), 3)
                text = font.render(f"{x},{y}", True, RED)
                surface.blit(text, (x + 4, y + 4))

        pygame.draw.rect(surface, BLACK, pygame.Rect(0, 0, width, height), 2)
        return surface.convert()

    def _go_back(self):
        from src.engine.scenes.menu import MenuScene
        return SceneSwitch(MenuScene())

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return self._go_back()
            if event.key == pygame.K_c:
                self.last_hit = None
                return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.viewport and self.viewport.collidepoint(event.pos):
                hit_input.push_mouse_hit(event.pos[0], event.pos[1])

        return None

    def update(self, dt: float):
        hit = hit_input.poll()
        if hit is not None:
            self.last_hit = hit
        return None

    def _draw_info(self, screen: pygame.Surface) -> None:
        panel = pygame.Surface((980, 215), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (20, 20))

        lines = [
            ("Grid / transform-test", self.font, WHITE),
            ("Rutnätet ritas exakt i viewporten: 10 px per ruta, kraftigare linjer varje 100 px.", self.small, WHITE),
            ("Klick i viewporten simulerar en träff via global hit_input.", self.tiny, WHITE),
            ("I framtiden ska kameran skicka in samma typ av träff-event.", self.tiny, WHITE),
            ("Visuella träffar visas av global hit visualizer om den är påslagen.", self.tiny, WHITE),
            ("C = rensa senaste träffinfo    ESC = tillbaka", self.tiny, WHITE),
        ]

        y = 32
        for text, font, color in lines:
            surf = font.render(text, True, color)
            screen.blit(surf, (36, y))
            y += surf.get_height() + 5

    def _draw_status(self, screen: pygame.Surface) -> None:
        lines: list[tuple[str, tuple[int, int, int]]] = []

        if self.viewport:
            lines.append(
                (
                    f"Viewport: x={self.viewport.x} y={self.viewport.y} w={self.viewport.w} h={self.viewport.h}",
                    WHITE,
                )
            )

        lines.append((self.status_message, WHITE))

        if self.last_hit is not None:
            lines.append((f"Senaste träffkälla: {self.last_hit.source}", YELLOW))
            lines.append(
                (
                    f"Spelets XY (lokalt i viewport): x={self.last_hit.game_x:.1f} y={self.last_hit.game_y:.1f}",
                    WHITE,
                )
            )
            lines.append(
                (
                    f"Skärm-XY: x={self.last_hit.screen_x:.1f} y={self.last_hit.screen_y:.1f}",
                    WHITE,
                )
            )
            lines.append(
                (
                    f"Kamera-XY: x={self.last_hit.camera_x:.1f} y={self.last_hit.camera_y:.1f}",
                    WHITE,
                )
            )

        panel_width = 820
        panel_height = (len(lines) * 28) + 20
        panel_x = 20
        panel_y = screen.get_height() - panel_height - 20

        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (panel_x, panel_y))

        y = panel_y + 10
        for text, color in lines:
            surf = self.tiny.render(text, True, color)
            screen.blit(surf, (panel_x + 16, y))
            y += 28

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)

        if self.viewport and self.grid_surface:
            screen.blit(self.grid_surface, self.viewport.topleft)

        self._draw_info(screen)
        self._draw_status(screen)