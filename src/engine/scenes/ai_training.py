from __future__ import annotations

import math

import pygame

from src.engine.ai.runtime import ai_runtime
from src.engine.ai.settings import load_ai_settings, save_ai_settings
from src.engine.ai.space_mapper import project_screen_point
from src.engine.input.hit_input import hit_input
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_viewport_rect

BG_WHITE = (248, 248, 248)
BG_BLACK = (18, 18, 18)
GRID_LINE_LIGHT = (222, 222, 222)
GRID_LINE_DARK = (58, 58, 58)
TEXT_LIGHT = (245, 245, 245)
TEXT_DARK = (30, 30, 30)
CYAN = (80, 220, 255)
YELLOW = (255, 210, 70)
ORANGE = (255, 150, 80)
CLICK_COLOR = (255, 110, 110)
HUD_BG = (0, 0, 0, 90)


class AITrainingScene(Scene):
    wants_hit_scanning = True

    def __init__(self, bg_mode: str = "white") -> None:
        self.bg_mode = str(bg_mode or 'white').strip().lower()
        self.font = None
        self.small = None
        self.tiny = None
        self.mode_index = 0
        self.mode_names = ["white", "black", "grid", "noise", "rings", "moving_box", "sweep"]
        if self.bg_mode in self.mode_names:
            self.mode_index = self.mode_names.index(self.bg_mode)
        self.t = 0.0
        self.last_camera_hit = None
        self.awaiting_label = False
        self.last_completed_shot_serial = None

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 34)
        self.small = pygame.font.Font(None, 24)
        self.tiny = pygame.font.Font(None, 18)
        hit_input.subscribe(self._on_hit)
        settings = load_ai_settings()
        updates = {}
        if settings.get("mode") == "off":
            updates["mode"] = "train_only"
        if int(settings.get("top_k", 5)) < 10:
            updates["top_k"] = 10
        if updates:
            save_ai_settings(updates)

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
        self.last_completed_shot_serial = None

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return self._go_back()
            if event.key == pygame.K_TAB:
                self.mode_index = (self.mode_index + 1) % len(self.mode_names)
                return None
            if event.key == pygame.K_r:
                ai_runtime.model.reset()
                self.awaiting_label = False
                self.last_completed_shot_serial = None
                return None
            return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.awaiting_label:
            projected = project_screen_point(float(event.pos[0]), float(event.pos[1]))
            feedback = ai_runtime.learn_from_click(projected.camera_x, projected.camera_y)
            if feedback is not None:
                self.last_completed_shot_serial = feedback.get("shot_serial")
            self.awaiting_label = False
            return None
        return None

    def update(self, dt: float):
        self.t += dt
        return None

    def render(self, screen: pygame.Surface) -> None:
        mode = self.mode_names[self.mode_index]
        if mode in {"white", "grid", "rings", "moving_box", "sweep", "noise"}:
            screen.fill(BG_WHITE)
            ink = TEXT_DARK
        else:
            screen.fill(BG_BLACK)
            ink = TEXT_LIGHT
        viewport = load_viewport_rect()
        if viewport is None:
            viewport = pygame.Rect(120, 100, 960, 540)
        self._render_training_content(screen, viewport, mode)

        prediction = ai_runtime.latest_prediction or {"candidates": []}
        candidates = prediction.get("candidates", []) or []
        shot_serial = prediction.get("shot_serial")
        show_candidates = bool(self.awaiting_label and shot_serial is not None and shot_serial != self.last_completed_shot_serial)
        if show_candidates:
            self._draw_candidates(screen, viewport, candidates)

        self._draw_minimal_hud(screen, viewport, ink, mode, show_candidates, candidates)

    def _draw_minimal_hud(self, screen: pygame.Surface, viewport: pygame.Rect, ink, mode: str, show_candidates: bool, candidates: list[dict]) -> None:
        top = pygame.Surface((260, 36), pygame.SRCALPHA)
        top.fill(HUD_BG)
        screen.blit(top, (16, 16))
        mode_label = {"white": "Vit", "black": "Svart", "grid": "Rutnät", "noise": "Brus", "rings": "Ringar", "moving_box": "Box", "sweep": "Sweep"}.get(mode, mode)
        screen.blit(self.small.render(f"AI-träning • {mode_label}", True, ink), (24, 24))

        status_text = "Skjut" if not self.awaiting_label else "Klicka"
        status_color = YELLOW if self.awaiting_label else CYAN
        pygame.draw.circle(screen, status_color, (viewport.right - 24, viewport.top + 24), 8)
        screen.blit(self.small.render(status_text, True, ink), (viewport.right - 88, viewport.top + 12))

        if show_candidates and candidates:
            box = pygame.Surface((190, 44), pygame.SRCALPHA)
            box.fill(HUD_BG)
            screen.blit(box, (viewport.left + 14, viewport.bottom - 58))
            best = candidates[0]
            txt = f"#1  {best.get('fused_score', 0.0):.2f}"
            screen.blit(self.small.render(txt, True, YELLOW), (viewport.left + 28, viewport.bottom - 48))
            hint = pygame.Surface((255, 30), pygame.SRCALPHA)
            hint.fill(HUD_BG)
            screen.blit(hint, (viewport.centerx - 128, viewport.top + 10))
            screen.blit(self.tiny.render("Klicka ungefär där du träffade", True, ink), (viewport.centerx - 104, viewport.top + 18))

    def _render_training_content(self, screen: pygame.Surface, rect: pygame.Rect, mode: str) -> None:
        if mode == "white":
            self._draw_plain(screen, rect, light=True)
        elif mode == "black":
            self._draw_plain(screen, rect, light=False)
        elif mode == "grid":
            self._draw_grid(screen, rect, dark=False)
        elif mode == "noise":
            self._draw_noise(screen, rect)
        elif mode == "rings":
            self._draw_rings(screen, rect)
        elif mode == "moving_box":
            self._draw_moving_box(screen, rect)
        else:
            self._draw_sweep(screen, rect)

    def _draw_plain(self, screen, rect, *, light: bool):
        color = BG_WHITE if light else BG_BLACK
        pygame.draw.rect(screen, color, rect)
        border = (210, 210, 210) if light else (80, 80, 80)
        pygame.draw.rect(screen, border, rect, 2)

    def _draw_grid(self, screen, rect, *, dark: bool):
        self._draw_plain(screen, rect, light=not dark)
        line = GRID_LINE_DARK if dark else GRID_LINE_LIGHT
        for x in range(rect.left, rect.right, 48):
            pygame.draw.line(screen, line, (x, rect.top), (x, rect.bottom), 1)
        for y in range(rect.top, rect.bottom, 48):
            pygame.draw.line(screen, line, (rect.left, y), (rect.right, y), 1)

    def _draw_noise(self, screen, rect):
        self._draw_plain(screen, rect, light=True)
        cell = 12
        phase = int(self.t * 25)
        for gy in range(rect.top, rect.bottom, cell):
            for gx in range(rect.left, rect.right, cell):
                v = 210 + ((gx * 11 + gy * 5 + phase * 13) % 40)
                pygame.draw.rect(screen, (v, v, v), (gx, gy, cell, cell))

    def _draw_rings(self, screen, rect):
        self._draw_plain(screen, rect, light=True)
        center = rect.center
        for radius in (220, 170, 120, 70, 30):
            pygame.draw.circle(screen, (170, 170, 170), center, radius, 2)

    def _draw_moving_box(self, screen, rect):
        self._draw_grid(screen, rect, dark=False)
        w, h = 120, 140
        x = rect.left + int((rect.w - w) * ((math.sin(self.t * 1.2) + 1.0) * 0.5))
        y = rect.top + int((rect.h - h) * ((math.cos(self.t * 0.9) + 1.0) * 0.5))
        pygame.draw.rect(screen, (85, 85, 85), (x, y, w, h))
        pygame.draw.rect(screen, (220, 220, 220), (x, y, w, h), 2)

    def _draw_sweep(self, screen, rect):
        self._draw_grid(screen, rect, dark=False)
        y = rect.top + int((rect.h - 1) * ((math.sin(self.t * 1.4) + 1.0) * 0.5))
        pygame.draw.line(screen, (80, 80, 80), (rect.left, y), (rect.right, y), 3)

    def _draw_candidates(self, screen: pygame.Surface, viewport: pygame.Rect, candidates: list[dict]) -> None:
        for cand in candidates[:10]:
            x = float(cand.get("screen_x", viewport.left + cand.get("camera_x", 0.0)))
            y = float(cand.get("screen_y", viewport.top + cand.get("camera_y", 0.0)))
            rank = int(cand.get("rank", 99))
            color = ORANGE if rank == 1 else (YELLOW if rank <= 3 else CYAN)
            radius = 20 if rank == 1 else (15 if rank <= 3 else 11)
            width = 3 if rank == 1 else 2
            pygame.draw.circle(screen, color, (int(round(x)), int(round(y))), radius, width)
            pygame.draw.line(screen, color, (int(round(x - radius - 5)), int(round(y))), (int(round(x + radius + 5)), int(round(y))), 1)
            pygame.draw.line(screen, color, (int(round(x)), int(round(y - radius - 5))), (int(round(x)), int(round(y + radius + 5))), 1)
            label = self.small.render(str(rank), True, color)
            screen.blit(label, (int(round(x)) + radius + 5, int(round(y)) - 12))
