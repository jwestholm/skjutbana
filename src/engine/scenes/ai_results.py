from __future__ import annotations

import time
from typing import Callable

import pygame

from config import SCREEN_HEIGHT, SCREEN_WIDTH
from src.engine.ai.result_repository import AIResultPoint, AIResultsRepository
from src.engine.ai.runtime import get_ai_runtime
from src.engine.scene import Scene


WHITE = (240, 240, 240)
SOFT_WHITE = (205, 205, 205)
MUTED = (145, 145, 145)
GREEN = (120, 255, 120)
RED = (255, 120, 120)
YELLOW = (255, 220, 100)
CYAN = (90, 220, 255)
PANEL_BG = (0, 0, 0, 180)
GRID = (80, 80, 80)


METRICS: list[tuple[str, str, Callable[[AIResultPoint], float | None]]] = [
    ("Found", "found_pct", lambda p: p.found_pct),
    ("Top-1", "top1_pct", lambda p: p.top1_pct),
    ("Top-3", "top3_pct", lambda p: p.top3_pct),
    ("AI rätt", "ai_correct_pct", lambda p: p.ai_correct_pct),
]


class AIResultsScene(Scene):
    """
    Historical AI benchmark dashboard.

    Reads BOTH:
      content/ai/reports/*.csv
      content/ai/automation_runs/**

    It does not modify content/menu.json. The scene is injected under the
    existing programmatic AI folder by menu_extension.py.

    Reset semantics are deliberately separated:
      R twice -> active learned AI memory only
      H twice -> benchmark/result history only
      T twice -> archived training examples only

    No reset command touches the other categories.
    """

    wants_hit_scanning = False
    wants_camera_preview = False

    def __init__(self, bg_color=(0, 0, 0), **kwargs) -> None:
        super().__init__()

        self.bg_color = (
            tuple(bg_color)
            if isinstance(bg_color, (tuple, list))
            else (0, 0, 0)
        )

        self.repository = AIResultsRepository()
        self.runtime = get_ai_runtime()

        self.points: list[AIResultPoint] = []
        self.backgrounds: list[str] = []
        self.background_index = 0  # 0 = all
        self.metric_index = 0

        self.font = None
        self.big = None
        self.small = None
        self.tiny = None

        self.status_message = ""
        self._confirm_action: str | None = None
        self._confirm_until = 0.0

    def on_enter(self) -> None:
        self.big = pygame.font.Font(None, 48)
        self.font = pygame.font.Font(None, 30)
        self.small = pygame.font.Font(None, 24)
        self.tiny = pygame.font.Font(None, 20)

        self.runtime = get_ai_runtime()
        self._reload()

    def on_exit(self) -> None:
        pass

    def update(self, dt: float):
        del dt

        if self._confirm_action and time.time() > self._confirm_until:
            self._confirm_action = None

        return None

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key in (pygame.K_1, pygame.K_KP1):
            self.metric_index = 0
            return None

        if event.key in (pygame.K_2, pygame.K_KP2):
            self.metric_index = 1
            return None

        if event.key in (pygame.K_3, pygame.K_KP3):
            self.metric_index = 2
            return None

        if event.key in (pygame.K_4, pygame.K_KP4):
            self.metric_index = 3
            return None

        if event.key in (pygame.K_LEFT, pygame.K_a):
            self._move_background(-1)
            return None

        if event.key in (pygame.K_RIGHT, pygame.K_d):
            self._move_background(+1)
            return None

        if event.key == pygame.K_F5:
            self._reload()
            self.status_message = "Resultathistoriken omläst."
            return None

        if event.key == pygame.K_r:
            self._confirm_or_execute(
                "memory",
                "R",
                self._reset_memory,
                "R igen inom 5 sek: nollställ AI-minnet.",
            )
            return None

        if event.key == pygame.K_h:
            self._confirm_or_execute(
                "history",
                "H",
                self._clear_history,
                "H igen inom 5 sek: radera resultathistoriken.",
            )
            return None

        if event.key == pygame.K_t:
            self._confirm_or_execute(
                "training_examples",
                "T",
                self._clear_training_examples,
                "T igen inom 5 sek: radera sparade träningsexempel.",
            )
            return None

        # ESC is handled by OverlayScene and returns to the exact prior menu
        # position via return_menu_state.
        return None

    def _confirm_or_execute(
        self,
        action: str,
        key_label: str,
        callback,
        warning: str,
    ) -> None:
        now = time.time()

        if self._confirm_action == action and now <= self._confirm_until:
            self._confirm_action = None
            self._confirm_until = 0.0

            try:
                callback()
            except Exception as exc:
                self.status_message = f"Åtgärden misslyckades: {exc}"

            return

        self._confirm_action = action
        self._confirm_until = now + 5.0
        self.status_message = warning

    def _reset_memory(self) -> None:
        self.runtime.memory.reset()
        self.status_message = (
            "AI-minnet nollställt. Resultat och träningsarkiv är kvar."
        )

    def _clear_history(self) -> None:
        result = self.repository.clear_result_history()
        self._reload()

        self.status_message = (
            "Resultathistorik rensad: "
            f"{result['legacy_reports_deleted']} CSV, "
            f"{result['automation_entries_deleted']} automation-filer. "
            "AI-minnet är kvar."
        )

    def _clear_training_examples(self) -> None:
        deleted = self.repository.clear_training_examples()

        self.status_message = (
            f"{deleted} sparade träningsexempel raderade. "
            "AI-minnet och resultathistoriken är kvar."
        )

    def _reload(self) -> None:
        self.points = self.repository.load_points()
        self.backgrounds = self.repository.backgrounds(self.points)

        max_index = len(self.backgrounds)
        self.background_index = max(
            0,
            min(self.background_index, max_index),
        )

    def _move_background(self, delta: int) -> None:
        options = len(self.backgrounds) + 1

        if options <= 1:
            self.background_index = 0
            return

        self.background_index = (self.background_index + delta) % options

    def _selected_background(self) -> str | None:
        if self.background_index <= 0:
            return None

        index = self.background_index - 1

        if 0 <= index < len(self.backgrounds):
            return self.backgrounds[index]

        return None

    def _filtered_points(self) -> list[AIResultPoint]:
        background = self._selected_background()

        if background is None:
            return list(self.points)

        return [
            point
            for point in self.points
            if point.background == background
        ]

    def _metric(self):
        return METRICS[self.metric_index]

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)

        if not all((self.big, self.font, self.small, self.tiny)):
            return

        pad = 30
        top_h = 170

        self._draw_header(screen, pad, top_h)

        graph_rect = pygame.Rect(
            pad,
            top_h + 20,
            SCREEN_WIDTH - pad * 2,
            max(260, SCREEN_HEIGHT - top_h - 145),
        )

        self._draw_graph(screen, graph_rect)
        self._draw_footer(screen, pad)

    def _draw_header(self, screen: pygame.Surface, pad: int, top_h: int) -> None:
        title = self.big.render("AI – Resultat", True, WHITE)
        screen.blit(title, (pad, 22))

        points = self._filtered_points()
        metric_label, _, getter = self._metric()
        values = [
            value
            for point in points
            if (value := getter(point)) is not None
        ]

        background = self._selected_background() or "Alla"
        memory = self.runtime.memory.summary()
        storage = self.repository.storage_summary()

        latest = values[-1] if values else None
        best = max(values) if values else None
        average = (sum(values) / len(values)) if values else None
        delta = (
            values[-1] - values[0]
            if len(values) >= 2
            else None
        )

        lines = [
            f"Bakgrund: {background}",
            f"Mätvärde: {metric_label}",
            f"Körningar: {len(points)}",
            f"Testskott: {sum(max(0, p.iterations) for p in points)}",
        ]

        x = pad
        y = 80
        for line in lines:
            surf = self.small.render(line, True, SOFT_WHITE)
            screen.blit(surf, (x, y))
            x += max(180, surf.get_width() + 30)

        stat_y = 118
        cards = [
            ("Senaste", latest),
            ("Bästa", best),
            ("Medel", average),
            ("Förändring", delta),
        ]

        x = pad
        for label, value in cards:
            if value is None:
                value_text = "–"
                color = MUTED
            else:
                value_text = f"{value:+.1f} pp" if label == "Förändring" else f"{value:.1f}%"
                color = GREEN if label != "Förändring" or value >= 0 else RED

            surf = self.small.render(f"{label}: {value_text}", True, color)
            screen.blit(surf, (x, stat_y))
            x += max(190, surf.get_width() + 30)

        right_lines = [
            f"AI-minne: {memory.get('positive_count', 0)} pos / {memory.get('negative_count', 0)} neg",
            f"Träningsexempel: {storage['training_example_files']}",
            f"CSV-rapporter: {storage['legacy_report_files']}",
            f"Automation-runs: {storage['automation_run_files']}",
        ]

        x = max(pad, SCREEN_WIDTH - 360)
        y = 28
        for line in right_lines:
            surf = self.tiny.render(line, True, CYAN)
            screen.blit(surf, (x, y))
            y += 22

    def _draw_graph(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        panel = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, rect.topleft)

        inner = rect.inflate(-70, -55)
        inner.y += 10
        inner.h -= 5

        # Y grid: 0..100%
        for pct in range(0, 101, 20):
            y = inner.bottom - int(inner.h * pct / 100.0)
            pygame.draw.line(screen, GRID, (inner.left, y), (inner.right, y), 1)
            label = self.tiny.render(f"{pct}%", True, MUTED)
            screen.blit(label, (inner.left - 42, y - 8))

        pygame.draw.line(
            screen,
            SOFT_WHITE,
            (inner.left, inner.bottom),
            (inner.right, inner.bottom),
            1,
        )
        pygame.draw.line(
            screen,
            SOFT_WHITE,
            (inner.left, inner.top),
            (inner.left, inner.bottom),
            1,
        )

        points = self._filtered_points()
        metric_label, _, getter = self._metric()

        plot = [
            (point, getter(point))
            for point in points
            if getter(point) is not None
        ]

        if not plot:
            text = self.font.render(
                "Ingen kompatibel resultathistorik hittades ännu.",
                True,
                SOFT_WHITE,
            )
            screen.blit(
                text,
                (
                    rect.centerx - text.get_width() // 2,
                    rect.centery - text.get_height() // 2,
                ),
            )
            return

        if len(plot) == 1:
            xs = [inner.centerx]
        else:
            xs = [
                inner.left + int(i * inner.w / (len(plot) - 1))
                for i in range(len(plot))
            ]

        coords = []
        for x, (_, value) in zip(xs, plot):
            safe_value = max(0.0, min(100.0, float(value)))
            y = inner.bottom - int(inner.h * safe_value / 100.0)
            coords.append((x, y))

        if len(coords) >= 2:
            pygame.draw.lines(screen, CYAN, False, coords, 3)

        for x, y in coords:
            pygame.draw.circle(screen, WHITE, (x, y), 4)
            pygame.draw.circle(screen, CYAN, (x, y), 3)

        graph_title = self.small.render(
            f"{metric_label} över tid – {self._selected_background() or 'alla bakgrunder'}",
            True,
            WHITE,
        )
        screen.blit(graph_title, (rect.x + 18, rect.y + 12))

        first = plot[0][0]
        last = plot[-1][0]

        left_label = self.tiny.render(
            self._point_label(first, 1),
            True,
            MUTED,
        )
        right_label = self.tiny.render(
            self._point_label(last, len(plot)),
            True,
            MUTED,
        )

        screen.blit(left_label, (inner.left, inner.bottom + 12))
        screen.blit(
            right_label,
            (
                inner.right - right_label.get_width(),
                inner.bottom + 12,
            ),
        )

    def _point_label(self, point: AIResultPoint, ordinal: int) -> str:
        stamp = (
            time.strftime("%m-%d %H:%M", time.localtime(point.timestamp))
            if point.timestamp > 0
            else "okänd tid"
        )
        return f"#{ordinal} {stamp}"

    def _draw_footer(self, screen: pygame.Surface, pad: int) -> None:
        help_text = (
            "1 Found  2 Top-1  3 Top-3  4 AI rätt   "
            "LEFT/RIGHT bakgrund   F5 läs om   "
            "R×2 AI-minne   H×2 historik   T×2 träningsexempel   ESC tillbaka"
        )

        help_surface = self.tiny.render(help_text, True, MUTED)
        screen.blit(help_surface, (pad, SCREEN_HEIGHT - 56))

        if self.status_message:
            color = YELLOW if self._confirm_action else GREEN
            status = self.small.render(self.status_message, True, color)
            screen.blit(status, (pad, SCREEN_HEIGHT - 88))
