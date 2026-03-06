from __future__ import annotations

from collections import deque
from pathlib import Path
import random

import pygame


def create_game(game_root: str, viewport: pygame.Rect):
    return ShootDontShootGame(game_root, viewport)


class ShootDontShootGame:
    def __init__(self, game_root: str, viewport: pygame.Rect) -> None:
        self.game_root = Path(game_root)
        self.viewport = viewport

        self.background = None
        self.mask = None

        self.characters: list[dict] = []
        self.hotspots: list[dict] = []
        self.active_slots: list[dict] = []

        self.font_big = None
        self.font_small = None

        self.state = "countdown"   # countdown -> action -> markera
        self.countdown_value = 3
        self.countdown_acc = 0.0

        self.timer_enabled = True
        self.action_time = 3.0
        self.action_remaining = self.action_time

    def on_enter(self) -> None:
        self.font_big = pygame.font.Font(None, 96)
        self.font_small = pygame.font.Font(None, 42)

        self.background = self._load_scaled_background()
        self.mask = self._load_mask()

        self.hotspots = self._extract_hotspots(self.mask)
        self.characters = self._load_characters()

        self._build_round()

    def _build_round(self) -> None:
        self.state = "countdown"
        self.countdown_value = 3
        self.countdown_acc = 0.0
        self.action_remaining = self.action_time

        if not self.hotspots or not self.characters:
            self.active_slots = []
            return

        num_people = random.randint(3, min(10, len(self.hotspots)))
        chosen_hotspots = random.sample(self.hotspots, num_people)
        chosen_characters = random.sample(self.characters, min(num_people, len(self.characters)))

        num_enemies = random.randint(0, min(3, num_people))
        enemy_indices = set(random.sample(range(num_people), num_enemies))

        self.active_slots = []
        for i, hotspot in enumerate(chosen_hotspots):
            char = chosen_characters[i % len(chosen_characters)]
            is_enemy = i in enemy_indices

            self.active_slots.append(
                {
                    "hotspot": hotspot,
                    "friendly": char["friendly"],
                    "hostile": char["hostile"],
                    "silhouette": char["silhouette"],
                    "is_enemy": is_enemy,
                }
            )

    def handle_event(self, event: pygame.event.Event):
        # ESC hanteras av GameScene
        # SPACE kan användas senare för att gå vidare / starta ny runda om du vill
        return None

    def update(self, dt: float):
        if self.state == "countdown":
            self.countdown_acc += dt
            if self.countdown_acc >= 1.0:
                self.countdown_acc = 0.0
                self.countdown_value -= 1
                if self.countdown_value <= 0:
                    self.state = "action"

        elif self.state == "action":
            if self.timer_enabled:
                self.action_remaining -= dt
                if self.action_remaining <= 0:
                    self.action_remaining = 0
                    self.state = "markera"

        return None

    def render(self, screen: pygame.Surface) -> None:
        # Bakgrunden ligger alltid i viewporten
        screen.blit(self.background, (self.viewport.x, self.viewport.y))

        for slot in self.active_slots:
            hotspot = slot["hotspot"]

            if self.state == "countdown":
                sprite = slot["silhouette"]
            else:
                sprite = slot["hostile"] if slot["is_enemy"] else slot["friendly"]

            scaled = self._scale_sprite(sprite, hotspot["scale"])
            x = self.viewport.x + hotspot["cx"] - scaled.get_width() // 2
            y = self.viewport.y + hotspot["cy"] - scaled.get_height()

            screen.blit(scaled, (x, y))

        # UI
        if self.state == "countdown":
            txt = "GO" if self.countdown_value <= 0 else str(self.countdown_value)
            self._draw_center_text(screen, txt)
        elif self.state == "action":
            if self.timer_enabled:
                txt = f"{self.action_remaining:0.1f}"
                self._draw_top_text(screen, txt)
        elif self.state == "markera":
            self._draw_top_text(screen, "MARKERA")

    # ---------- assets ----------
    def _load_scaled_background(self) -> pygame.Surface:
        bg_path = self.game_root / "b1.png"
        bg = pygame.image.load(str(bg_path)).convert()
        return pygame.transform.smoothscale(bg, (self.viewport.w, self.viewport.h))

    def _load_mask(self) -> pygame.Surface:
        mask_path = self.game_root / "m1.png"
        img = pygame.image.load(str(mask_path)).convert_alpha()
        return pygame.transform.smoothscale(img, (self.viewport.w, self.viewport.h))

    def _load_characters(self) -> list[dict]:
        out: list[dict] = []

        for path in sorted(self.game_root.glob("*.png")):
            name = path.name.lower()

            if name.startswith("b") or name.startswith("m"):
                continue

            stem = path.stem
            if not stem.isdigit():
                continue

            img = pygame.image.load(str(path)).convert_alpha()
            w, h = img.get_size()
            half = w // 2

            friendly = img.subsurface((0, 0, half, h)).copy()
            hostile = img.subsurface((half, 0, half, h)).copy()
            silhouette = self._make_silhouette(friendly, hostile)

            out.append(
                {
                    "friendly": friendly,
                    "hostile": hostile,
                    "silhouette": silhouette,
                }
            )

        return out

    def _make_silhouette(self, friendly: pygame.Surface, hostile: pygame.Surface) -> pygame.Surface:
        w = max(friendly.get_width(), hostile.get_width())
        h = max(friendly.get_height(), hostile.get_height())

        base = pygame.Surface((w, h), pygame.SRCALPHA)
        base.blit(friendly, ((w - friendly.get_width()) // 2, h - friendly.get_height()))
        base.blit(hostile, ((w - hostile.get_width()) // 2, h - hostile.get_height()))

        alpha = pygame.surfarray.array_alpha(base)
        silhouette = pygame.Surface((w, h), pygame.SRCALPHA)
        silhouette.fill((0, 0, 0, 0))

        # svart siluett med samma alpha
        arr = pygame.surfarray.pixels3d(silhouette)
        arr[:, :, 0] = 0
        arr[:, :, 1] = 0
        arr[:, :, 2] = 0
        del arr

        alpha_target = pygame.surfarray.pixels_alpha(silhouette)
        alpha_target[:, :] = alpha[:, :]
        del alpha_target

        return silhouette

    # ---------- hotspots ----------
    def _extract_hotspots(self, mask_surface: pygame.Surface) -> list[dict]:
        w, h = mask_surface.get_size()
        visited = [[False for _ in range(h)] for _ in range(w)]
        hotspots: list[dict] = []

        def is_hotspot_pixel(x: int, y: int) -> bool:
            color = mask_surface.get_at((x, y))
            # transparent = ej hotspot
            if len(color) >= 4 and color.a == 0:
                return False
            # helt svart = ej hotspot
            return not (color.r == 0 and color.g == 0 and color.b == 0)

        for sx in range(w):
            for sy in range(h):
                if visited[sx][sy]:
                    continue
                visited[sx][sy] = True

                if not is_hotspot_pixel(sx, sy):
                    continue

                q = deque()
                q.append((sx, sy))

                pixels = []
                sum_gray = 0

                while q:
                    x, y = q.popleft()
                    pixels.append((x, y))

                    c = mask_surface.get_at((x, y))
                    gray = (int(c.r) + int(c.g) + int(c.b)) / 3.0
                    sum_gray += gray

                    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if 0 <= nx < w and 0 <= ny < h and not visited[nx][ny]:
                            visited[nx][ny] = True
                            if is_hotspot_pixel(nx, ny):
                                q.append((nx, ny))

                if not pixels:
                    continue

                xs = [p[0] for p in pixels]
                ys = [p[1] for p in pixels]

                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)

                cx = (min_x + max_x) // 2
                cy = max_y  # stå på "golvet" i polygonen

                mean_gray = sum_gray / len(pixels)
                depth = mean_gray / 255.0

                # mörk = nära = större, ljus = långt bort = mindre
                min_scale = 0.35
                max_scale = 1.0
                scale = max_scale - (depth * (max_scale - min_scale))

                hotspots.append(
                    {
                        "cx": cx,
                        "cy": cy,
                        "scale": scale,
                        "bounds": (min_x, min_y, max_x, max_y),
                    }
                )

        # sortera ungefär bakifrån till framifrån för vettig ritordning
        hotspots.sort(key=lambda hs: hs["cy"])
        return hotspots

    # ---------- helpers ----------
    def _scale_sprite(self, sprite: pygame.Surface, scale: float) -> pygame.Surface:
        w, h = sprite.get_size()
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return pygame.transform.smoothscale(sprite, (new_w, new_h))

    def _draw_center_text(self, screen: pygame.Surface, text: str) -> None:
        surf = self.font_big.render(text, True, (255, 255, 255))
        x = self.viewport.x + (self.viewport.w - surf.get_width()) // 2
        y = self.viewport.y + 40
        screen.blit(surf, (x, y))

    def _draw_top_text(self, screen: pygame.Surface, text: str) -> None:
        surf = self.font_big.render(text, True, (255, 255, 255))
        x = self.viewport.x + (self.viewport.w - surf.get_width()) // 2
        y = self.viewport.y + 20
        screen.blit(surf, (x, y))