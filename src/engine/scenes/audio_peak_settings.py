from __future__ import annotations

import pygame
import numpy as np

from src.engine.audio.audio_peak_detector import audio_peak_detector
from src.engine.scene import Scene, SceneSwitch


WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
GREEN = (120, 255, 120)
RED = (255, 120, 120)
GRAY = (120, 120, 120)
PANEL_BG = (0, 0, 0, 170)


class AudioPeakSettingsScene(Scene):
    """
    Inställningsscen för ljud-peak.

    Kontroller:
    - LEFT / RIGHT: sänk / höj threshold
    - SHIFT + LEFT / RIGHT: finjustera
    - R: återställ till 0.10
    - ESC: tillbaka
    """

    def __init__(self, bg_color=(0, 0, 0)) -> None:
        self.bg_color = tuple(bg_color)
        self.font = None
        self.small = None
        self.tiny = None

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 28)
        self.tiny = pygame.font.Font(None, 22)

        audio_peak_detector.start()

    def on_exit(self) -> None:
        pass

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            from src.engine.scenes.menu import MenuScene
            return SceneSwitch(MenuScene())

        mods = pygame.key.get_mods()
        fine = bool(mods & pygame.KMOD_SHIFT)
        step = 0.002 if fine else 0.01

        threshold = audio_peak_detector.get_peak_threshold()

        if event.key == pygame.K_LEFT:
            audio_peak_detector.set_peak_threshold(threshold - step)
        elif event.key == pygame.K_RIGHT:
            audio_peak_detector.set_peak_threshold(threshold + step)
        elif event.key == pygame.K_r:
            audio_peak_detector.set_peak_threshold(0.10)

        return None

    def update(self, dt: float):
        del dt
        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)

        waveform = audio_peak_detector.get_waveform_snapshot(max_points=max(400, screen.get_width() - 80))
        threshold = audio_peak_detector.get_peak_threshold()

        title_panel = pygame.Surface((screen.get_width() - 40, 110), pygame.SRCALPHA)
        title_panel.fill(PANEL_BG)
        screen.blit(title_panel, (20, 20))

        title = self.font.render("Ljud-peak / mikrofon", True, WHITE)
        screen.blit(title, (35, 32))

        hint = self.small.render(
            "LEFT/RIGHT: threshold  |  SHIFT: finjustera  |  R: reset  |  ESC: tillbaka",
            True,
            SOFT_WHITE,
        )
        screen.blit(hint, (35, 70))

        graph_rect = pygame.Rect(40, 150, screen.get_width() - 80, screen.get_height() - 260)
        pygame.draw.rect(screen, (30, 30, 30), graph_rect)
        pygame.draw.rect(screen, GRAY, graph_rect, 1)

        mid_y = graph_rect.centery
        pygame.draw.line(screen, GRAY, (graph_rect.left, mid_y), (graph_rect.right, mid_y), 1)

        # threshold-linjer (positiv och negativ eftersom waveform är signerad)
        thr_px = int(threshold * (graph_rect.height * 0.45))
        top_thr_y = mid_y - thr_px
        bot_thr_y = mid_y + thr_px
        self._draw_dashed_line(screen, (graph_rect.left, top_thr_y), (graph_rect.right, top_thr_y), RED, 8, 6, 2)
        self._draw_dashed_line(screen, (graph_rect.left, bot_thr_y), (graph_rect.right, bot_thr_y), RED, 8, 6, 2)

        if waveform.size > 1:
            points = []
            n = waveform.size
            for i, sample in enumerate(waveform):
                x = graph_rect.left + int((i / max(1, n - 1)) * graph_rect.width)
                y = mid_y - int(float(sample) * (graph_rect.height * 0.45))
                points.append((x, y))
            if len(points) >= 2:
                pygame.draw.lines(screen, WHITE, False, points, 1)

        status_panel = pygame.Surface((screen.get_width() - 40, 80), pygame.SRCALPHA)
        status_panel.fill(PANEL_BG)
        screen.blit(status_panel, (20, screen.get_height() - 100))

        latest_peak = audio_peak_detector.last_peak_value
        latest_rms = audio_peak_detector.last_rms
        backend = audio_peak_detector.backend_name
        error = audio_peak_detector.last_error

        line1 = self.small.render(
            f"Backend: {backend}   Peak: {latest_peak:.3f}   RMS: {latest_rms:.3f}   Threshold: {threshold:.3f}",
            True,
            GREEN if latest_peak >= threshold else SOFT_WHITE,
        )
        screen.blit(line1, (35, screen.get_height() - 88))

        if error:
            line2 = self.tiny.render(f"Fel: {error}", True, RED)
        else:
            line2 = self.tiny.render(
                "Tips: skjut eller knacka nära mikrofonen och justera så peaks går över röd nivå.",
                True,
                SOFT_WHITE,
            )
        screen.blit(line2, (35, screen.get_height() - 58))

    def _draw_dashed_line(
        self,
        screen: pygame.Surface,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        dash_len: int,
        gap_len: int,
        width: int,
    ) -> None:
        x1, y1 = start
        x2, y2 = end

        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length <= 0.0:
            return

        step = dash_len + gap_len
        count = int(length // step) + 1

        ux = dx / length
        uy = dy / length

        for i in range(count):
            seg_start = i * step
            seg_end = min(seg_start + dash_len, length)

            if seg_start >= length:
                break

            ax = int(round(x1 + ux * seg_start))
            ay = int(round(y1 + uy * seg_start))
            bx = int(round(x1 + ux * seg_end))
            by = int(round(y1 + uy * seg_end))

            pygame.draw.line(screen, color, (ax, ay), (bx, by), width)