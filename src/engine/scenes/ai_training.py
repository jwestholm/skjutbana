from __future__ import annotations

import math

import pygame

from src.engine.ai.runtime import ai_runtime
from src.engine.ai.settings import load_ai_settings, save_ai_settings
from src.engine.ai.space_mapper import project_camera_point, project_screen_point
from src.engine.input.hit_input import hit_input
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_viewport_rect

WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
GREEN = (120, 255, 120)
YELLOW = (255, 220, 100)
CYAN = (120, 220, 255)
PANEL_BG = (0, 0, 0, 170)
CLICK_COLOR = (255, 140, 140)


class AITrainingScene(Scene):
    wants_hit_scanning = True

    def __init__(self, bg_color=(0, 0, 0)) -> None:
        self.bg_color = tuple(bg_color)
        self.font = None
        self.small = None
        self.tiny = None
        self.mode_index = 0
        self.mode_names = ["grid", "noise", "rings", "moving_box", "sweep"]
        self.t = 0.0
        self.last_camera_hit = None
        self.last_feedback = None
        self.awaiting_label = False
        self.last_click_screen = None

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 28)
        self.tiny = pygame.font.Font(None, 23)
        hit_input.subscribe(self._on_hit)
        settings = load_ai_settings()
        if settings.get("mode") == "off":
            settings["mode"] = "train_only"
            save_ai_settings(settings)

    def on_exit(self) -> None:
        hit_input.unsubscribe(self._on_hit)

    def _go_back(self):
        from src.engine.scenes.menu import MenuScene

        menu_state = getattr(self, "return_menu_state", None)
        return SceneSwitch(MenuScene(menu_state=menu_state))

    def _on_hit(self, event) -> None:
        if getattr(event, "source", "") != "camera":
            return
        self.last_camera_hit = event
        self.awaiting_label = True

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return self._go_back()
            if event.key == pygame.K_TAB:
                self.mode_index = (self.mode_index + 1) % len(self.mode_names)
                return None
            if event.key == pygame.K_r:
                ai_runtime.model.reset()
                return None
            return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            projected = project_screen_point(float(event.pos[0]), float(event.pos[1]))
            self.last_click_screen = (float(event.pos[0]), float(event.pos[1]))
            self.last_feedback = ai_runtime.learn_from_click(projected.camera_x, projected.camera_y)
            self.awaiting_label = False
            return None
        return None

    def update(self, dt: float):
        self.t += dt
        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)
        viewport = load_viewport_rect()
        if viewport is None:
            viewport = pygame.Rect(120, 100, 960, 540)
        pygame.draw.rect(screen, (25, 25, 25), viewport)
        self._render_training_content(screen, viewport)

        prediction = ai_runtime.latest_prediction or {"candidates": []}
        candidates = prediction.get("candidates", []) or []
        self._draw_candidates(screen, viewport, candidates)
        self._draw_last_click(screen)

        panel = pygame.Surface((1180, 220), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (30, 20))
        title = self.font.render("AI-träning", True, WHITE)
        screen.blit(title, (50, 35))
        lines = [
            f"Läge: {self.mode_names[self.mode_index]}",
            f"AI-kandidater: {len(candidates)} | Väntar på label: {'JA' if self.awaiting_label else 'NEJ'}",
            "Skjut ett skott. Kandidaterna numreras på tavlan. Klicka sedan ungefär där du träffade.",
            "AI tänker i videoplanet men visas i spelplanet via samma transformkedja som träffarna.",
            "TAB: byt träningsyta   R: nollställ modell   ESC: tillbaka",
        ]
        y = 86
        for line in lines:
            surf = self.small.render(line, True, SOFT_WHITE)
            screen.blit(surf, (50, y))
            y += 28

        if self.last_feedback is not None:
            accepted = bool(self.last_feedback.get("accepted", False))
            color = GREEN if accepted else YELLOW
            feedback_line = f"Senaste label: {'rätt kandidat hittades' if accepted else 'ingen kandidat låg nära klicket'}"
            surf = self.small.render(feedback_line, True, color)
            screen.blit(surf, (50, 196))

        info_x = viewport.right + 20
        if info_x < screen.get_width() - 220:
            info_y = viewport.y
            box = pygame.Surface((260, 460), pygame.SRCALPHA)
            box.fill((0, 0, 0, 150))
            screen.blit(box, (info_x, info_y))
            hdr = self.small.render("Topplista", True, CYAN)
            screen.blit(hdr, (info_x + 16, info_y + 16))
            y = info_y + 56
            for cand in candidates[:5]:
                line = f"#{cand.get('rank', '?')}  {cand.get('fused_score', 0.0):.2f}"
                color = YELLOW if int(cand.get("rank", 99)) == 1 else SOFT_WHITE
                surf = self.tiny.render(line, True, color)
                screen.blit(surf, (info_x + 16, y))
                y += 24
                xy = self.tiny.render(
                    f"G:{cand.get('game_x', 0.0):.1f},{cand.get('game_y', 0.0):.1f}  C:{cand.get('camera_x', 0.0):.1f},{cand.get('camera_y', 0.0):.1f}",
                    True,
                    (170, 170, 170),
                )
                screen.blit(xy, (info_x + 16, y))
                y += 24

    def _render_training_content(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        mode = self.mode_names[self.mode_index]
        if mode == "grid":
            self._draw_grid(screen, rect)
        elif mode == "noise":
            self._draw_noise(screen, rect)
        elif mode == "rings":
            self._draw_rings(screen, rect)
        elif mode == "moving_box":
            self._draw_moving_box(screen, rect)
        else:
            self._draw_sweep(screen, rect)

    def _draw_grid(self, screen, rect):
        for x in range(rect.left, rect.right, 48):
            pygame.draw.line(screen, (60, 60, 60), (x, rect.top), (x, rect.bottom), 1)
        for y in range(rect.top, rect.bottom, 48):
            pygame.draw.line(screen, (60, 60, 60), (rect.left, y), (rect.right, y), 1)

    def _draw_noise(self, screen, rect):
        cell = 12
        phase = int(self.t * 30)
        for gy in range(rect.top, rect.bottom, cell):
            for gx in range(rect.left, rect.right, cell):
                v = (gx * 13 + gy * 7 + phase * 17) % 255
                color = (v, v, v)
                pygame.draw.rect(screen, color, (gx, gy, cell, cell))

    def _draw_rings(self, screen, rect):
        center = rect.center
        for radius in (220, 170, 120, 70, 30):
            pygame.draw.circle(screen, (180, 180, 180), center, radius, 2)

    def _draw_moving_box(self, screen, rect):
        self._draw_grid(screen, rect)
        w, h = 120, 140
        x = rect.left + int((rect.w - w) * ((math.sin(self.t * 1.2) + 1.0) * 0.5))
        y = rect.top + int((rect.h - h) * ((math.cos(self.t * 0.9) + 1.0) * 0.5))
        pygame.draw.rect(screen, (90, 90, 90), (x, y, w, h))
        pygame.draw.rect(screen, (220, 220, 220), (x, y, w, h), 2)

    def _draw_sweep(self, screen, rect):
        self._draw_grid(screen, rect)
        y = rect.top + int((rect.h - 1) * ((math.sin(self.t * 1.4) + 1.0) * 0.5))
        pygame.draw.line(screen, (210, 210, 210), (rect.left, y), (rect.right, y), 3)

    def _draw_candidates(self, screen: pygame.Surface, viewport: pygame.Rect, candidates: list[dict]) -> None:
        for cand in candidates:
            x = float(cand.get("screen_x", viewport.left + cand.get("camera_x", 0.0)))
            y = float(cand.get("screen_y", viewport.top + cand.get("camera_y", 0.0)))
            rank = int(cand.get("rank", 99))
            color = YELLOW if rank == 1 else CYAN
            radius = 18 if rank == 1 else 12
            pygame.draw.circle(screen, color, (int(round(x)), int(round(y))), radius, 2)
            pygame.draw.line(screen, color, (int(round(x - radius - 5)), int(round(y))), (int(round(x + radius + 5)), int(round(y))), 1)
            pygame.draw.line(screen, color, (int(round(x)), int(round(y - radius - 5))), (int(round(x)), int(round(y + radius + 5))), 1)
            label = self.small.render(str(rank), True, color)
            screen.blit(label, (int(round(x)) + radius + 4, int(round(y)) - 12))

    def _draw_last_click(self, screen: pygame.Surface) -> None:
        if self.last_click_screen is None:
            return
        x, y = self.last_click_screen
        pygame.draw.circle(screen, CLICK_COLOR, (int(round(x)), int(round(y))), 10, 2)
        pygame.draw.line(screen, CLICK_COLOR, (int(round(x - 14)), int(round(y))), (int(round(x + 14)), int(round(y))), 2)
        pygame.draw.line(screen, CLICK_COLOR, (int(round(x)), int(round(y - 14))), (int(round(x)), int(round(y + 14))), 2)

