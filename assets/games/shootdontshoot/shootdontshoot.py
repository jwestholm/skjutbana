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

        self.background: pygame.Surface | None = None
        self.mask: pygame.Surface | None = None

        self.characters: list[dict] = []
        self.hotspots: list[dict] = []
        self.active_slots: list[dict] = []

        self.font_big: pygame.font.Font | None = None
        self.font_small: pygame.font.Font | None = None
        self.font_markera: pygame.font.Font | None = None

        self.state = "countdown"   # countdown -> action -> markera
        self.countdown_value = 10
        self.countdown_acc = 0.0

        self.timer_enabled = True
        self.action_time = 10.0
        self.action_remaining = self.action_time

        self._scaled_cache: dict[tuple[int, int], pygame.Surface] = {}

        # Ca 33 % mindre än förra versionen
        self.min_scale = 0.08
        self.max_scale = 0.30

        # Mask-analys i låg upplösning för fart
        self.mask_analysis_max_w = 320
        self.mask_analysis_max_h = 180
        self.min_region_pixels = 8
        self.gray_tolerance = 12

    def on_enter(self) -> None:
        self.font_big = pygame.font.Font(None, 96)
        self.font_small = pygame.font.Font(None, 42)
        self.font_markera = pygame.font.Font(None, 64)

        self.background = self._load_scaled_background()
        self.mask = self._load_mask()

        self.hotspots = self._extract_hotspots(self.mask)
        self.characters = self._load_characters()

        self._build_round()

    def _build_round(self) -> None:
        self.state = "countdown"
        self.countdown_value = 10
        self.countdown_acc = 0.0
        self.action_remaining = self.action_time
        self._scaled_cache.clear()

        if not self.hotspots or not self.characters:
            self.active_slots = []
            return

        max_people = min(10, len(self.hotspots))
        if max_people <= 0:
            self.active_slots = []
            return

        min_people = min(3, max_people)
        num_people = random.randint(min_people, max_people)

        chosen_hotspots = self._choose_spread_hotspots(num_people)

        self.active_slots = []
        for i, hotspot in enumerate(chosen_hotspots):
            char = random.choice(self.characters)
            is_enemy = False

            self.active_slots.append(
                {
                    "hotspot": hotspot,
                    "friendly": char["friendly"],
                    "hostile": char["hostile"],
                    "silhouette": char["silhouette"],
                    "is_enemy": is_enemy,
                }
            )

        num_enemies = random.randint(0, min(3, len(self.active_slots)))
        if num_enemies > 0:
            enemy_indices = set(random.sample(range(len(self.active_slots)), num_enemies))
            for i, slot in enumerate(self.active_slots):
                slot["is_enemy"] = i in enemy_indices

        self.active_slots.sort(key=lambda s: s["hotspot"]["cy"])

    def handle_event(self, event: pygame.event.Event):
        return None

    def update(self, dt: float):
        if self.state == "countdown":
            self.countdown_acc += dt
            while self.countdown_acc >= 1.0:
                self.countdown_acc -= 1.0
                self.countdown_value -= 1
                if self.countdown_value <= 0:
                    self.state = "action"
                    break

        elif self.state == "action":
            if self.timer_enabled:
                self.action_remaining -= dt
                if self.action_remaining <= 0:
                    self.action_remaining = 0
                    self.state = "markera"

        return None

    def render(self, screen: pygame.Surface) -> None:
        if self.background is None:
            return

        screen.blit(self.background, (self.viewport.x, self.viewport.y))

        for slot in self.active_slots:
            hotspot = slot["hotspot"]

            if self.state == "countdown":
                sprite = slot["silhouette"]
            else:
                sprite = slot["hostile"] if slot["is_enemy"] else slot["friendly"]

            scaled = self._scale_sprite(sprite, hotspot["scale"])

            # bottom-center ankare
            draw_x = self.viewport.x + hotspot["cx"] - scaled.get_width() // 2
            draw_y = self.viewport.y + hotspot["cy"] - scaled.get_height()

            screen.blit(scaled, (draw_x, draw_y))

        if self.state == "countdown":
            txt = "GO" if self.countdown_value <= 0 else str(self.countdown_value)
            self._draw_center_text(screen, txt)
        elif self.state == "action":
            if self.timer_enabled:
                txt = f"{self.action_remaining:0.1f}"
                self._draw_top_text(screen, txt)
        elif self.state == "markera":
            self._draw_markera_text(screen)

    # ---------- assets ----------
    def _load_scaled_background(self) -> pygame.Surface:
        backgrounds = sorted(
            p for p in self.game_root.glob("b*.png")
            if p.stem[1:].isdigit()
        )

        if not backgrounds:
            bg_path = self.game_root / "b1.png"
        else:
            bg_path = random.choice(backgrounds)

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

            if half <= 0:
                continue

            friendly = img.subsurface((0, 0, half, h)).copy().convert_alpha()
            hostile = img.subsurface((half, 0, w - half, h)).copy().convert_alpha()

            # Snabb vit-transparens via colorkey i stället för pixel-loop
            friendly = self._apply_white_colorkey(friendly)
            hostile = self._apply_white_colorkey(hostile)

            silhouette = self._make_silhouette(friendly, hostile)

            out.append(
                {
                    "friendly": friendly,
                    "hostile": hostile,
                    "silhouette": silhouette,
                }
            )

        return out

    def _apply_white_colorkey(self, surf: pygame.Surface) -> pygame.Surface:
        result = surf.copy().convert_alpha()
        result.set_colorkey((255, 255, 255))
        return result

    def _make_silhouette(self, friendly: pygame.Surface, hostile: pygame.Surface) -> pygame.Surface:
        w = max(friendly.get_width(), hostile.get_width())
        h = max(friendly.get_height(), hostile.get_height())

        base = pygame.Surface((w, h), pygame.SRCALPHA)
        base.blit(friendly, ((w - friendly.get_width()) // 2, h - friendly.get_height()))
        base.blit(hostile, ((w - hostile.get_width()) // 2, h - hostile.get_height()))

        alpha = pygame.surfarray.array_alpha(base)
        silhouette = pygame.Surface((w, h), pygame.SRCALPHA)

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
        """
        Snabbare och bättre mask-analys:
        - analysera en nedskalad kopia av masken
        - hitta sammanhängande regioner med liknande gråvärde
        - placera hotspot i nederkanten av varje region
        - skala tillbaka till viewport-koordinater
        """
        full_w, full_h = mask_surface.get_size()

        analysis_w = min(self.mask_analysis_max_w, full_w)
        analysis_h = min(self.mask_analysis_max_h, full_h)

        small_mask = pygame.transform.smoothscale(mask_surface, (analysis_w, analysis_h))
        w, h = small_mask.get_size()

        visited = [[False for _ in range(h)] for _ in range(w)]
        hotspots: list[dict] = []

        scale_x = full_w / w
        scale_y = full_h / h

        def pixel_info(x: int, y: int) -> tuple[int, int]:
            c = small_mask.get_at((x, y))
            gray = (int(c.r) + int(c.g) + int(c.b)) // 3
            return int(c.a), gray

        for sx in range(w):
            for sy in range(h):
                if visited[sx][sy]:
                    continue

                visited[sx][sy] = True
                a, seed_gray = pixel_info(sx, sy)

                if a == 0:
                    continue

                q = deque([(sx, sy)])
                pixels: list[tuple[int, int]] = []
                sum_gray = 0.0

                while q:
                    x, y = q.popleft()
                    a2, gray2 = pixel_info(x, y)
                    if a2 == 0:
                        continue

                    pixels.append((x, y))
                    sum_gray += gray2

                    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if 0 <= nx < w and 0 <= ny < h and not visited[nx][ny]:
                            visited[nx][ny] = True
                            na, ngray = pixel_info(nx, ny)
                            if na > 0 and abs(ngray - seed_gray) <= self.gray_tolerance:
                                q.append((nx, ny))

                if len(pixels) < self.min_region_pixels:
                    continue

                max_y = max(y for _, y in pixels)
                bottom_xs = [x for x, y in pixels if y == max_y]

                if not bottom_xs:
                    continue

                cx_small = sum(bottom_xs) / len(bottom_xs)
                cy_small = max_y

                mean_gray = sum_gray / len(pixels)
                depth = mean_gray / 255.0  # 0=svart/nära, 1=vit/långt bort
                scale = self.max_scale - depth * (self.max_scale - self.min_scale)

                cx = int(cx_small * scale_x)
                cy = int(cy_small * scale_y)

                hotspots.append(
                    {
                        "cx": max(0, min(full_w - 1, cx)),
                        "cy": max(0, min(full_h - 1, cy)),
                        "depth": depth,
                        "scale": scale,
                        "pixel_count": len(pixels),
                    }
                )

        # slå ihop hotspots som ligger för nära
        merged: list[dict] = []
        min_distance = max(28, min(full_w, full_h) // 14)

        for hs in sorted(hotspots, key=lambda item: (item["cy"], item["cx"])):
            too_close = False
            for kept in merged:
                dx = kept["cx"] - hs["cx"]
                dy = kept["cy"] - hs["cy"]
                if (dx * dx + dy * dy) < (min_distance * min_distance):
                    too_close = True
                    break
            if not too_close:
                merged.append(hs)

        merged.sort(key=lambda hs: hs["cy"])
        return merged

    def _choose_spread_hotspots(self, count: int) -> list[dict]:
        if count >= len(self.hotspots):
            return self.hotspots[:]

        sorted_hotspots = sorted(self.hotspots, key=lambda hs: hs["depth"])
        buckets = [[], [], []]

        for hs in sorted_hotspots:
            if hs["depth"] < 0.33:
                buckets[0].append(hs)
            elif hs["depth"] < 0.66:
                buckets[1].append(hs)
            else:
                buckets[2].append(hs)

        chosen: list[dict] = []
        used_ids: set[int] = set()

        while len(chosen) < count:
            available_buckets = [b for b in buckets if any(id(h) not in used_ids for h in b)]
            if not available_buckets:
                break

            bucket = random.choice(available_buckets)
            candidates = [h for h in bucket if id(h) not in used_ids]
            if not candidates:
                continue

            hs = random.choice(candidates)
            chosen.append(hs)
            used_ids.add(id(hs))

        if len(chosen) < count:
            remaining = [h for h in self.hotspots if id(h) not in used_ids]
            if remaining:
                chosen.extend(random.sample(remaining, min(len(remaining), count - len(chosen))))

        chosen.sort(key=lambda hs: hs["cy"])
        return chosen

    # ---------- helpers ----------
    def _scale_sprite(self, sprite: pygame.Surface, scale: float) -> pygame.Surface:
        key = (id(sprite), int(scale * 1000))
        cached = self._scaled_cache.get(key)
        if cached is not None:
            return cached

        w, h = sprite.get_size()
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        scaled = pygame.transform.smoothscale(sprite, (new_w, new_h))
        self._scaled_cache[key] = scaled
        return scaled

    def _draw_center_text(self, screen: pygame.Surface, text: str) -> None:
        assert self.font_big is not None
        surf = self.font_big.render(text, True, (255, 255, 255))
        x = self.viewport.x + (self.viewport.w - surf.get_width()) // 2
        y = self.viewport.y + 40
        screen.blit(surf, (x, y))

    def _draw_top_text(self, screen: pygame.Surface, text: str) -> None:
        assert self.font_big is not None
        surf = self.font_big.render(text, True, (255, 255, 255))
        x = self.viewport.x + (self.viewport.w - surf.get_width()) // 2
        y = self.viewport.y + 16
        screen.blit(surf, (x, y))

    def _draw_markera_text(self, screen: pygame.Surface) -> None:
        assert self.font_markera is not None
        surf = self.font_markera.render("MARKERA", True, (255, 255, 255))
        x = self.viewport.x + (self.viewport.w - surf.get_width()) // 2
        y = self.viewport.y + 8
        screen.blit(surf, (x, y))