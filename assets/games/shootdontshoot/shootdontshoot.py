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

        self.state = "countdown"   # countdown -> action -> markera
        self.countdown_value = 3
        self.countdown_acc = 0.0

        self.timer_enabled = True
        self.action_time = 3.0
        self.action_remaining = self.action_time

        self._scaled_cache: dict[tuple[int, int, int], pygame.Surface] = {}

        # Skalning: svart = nära = störst, vitt = långt bort = minst
        # Justera vid behov om figurer fortfarande känns stora/små.
        self.min_scale = 0.02
        self.max_scale = 0.10

        # Minsta sammanhängande område i masken för att räknas som hotspot
        self.min_hotspot_pixels = 40

        # Hur nära i gråvärde pixlar måste ligga för att räknas till samma hotspot
        self.gray_tolerance = 10

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
        self._scaled_cache.clear()

        if not self.hotspots or not self.characters:
            self.active_slots = []
            return

        max_people = min(10, len(self.hotspots), len(self.characters))
        min_people = min(3, max_people)

        if max_people <= 0:
            self.active_slots = []
            return

        num_people = random.randint(min_people, max_people)

        # Lite bättre variation: välj hotspots utspritt över djup
        chosen_hotspots = self._choose_spread_hotspots(num_people)
        chosen_characters = random.sample(self.characters, num_people)

        num_enemies = random.randint(0, min(3, num_people))
        enemy_indices = set(random.sample(range(num_people), num_enemies))

        self.active_slots = []
        for i, hotspot in enumerate(chosen_hotspots):
            char = chosen_characters[i]
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

        # Rita bakifrån till framifrån
        self.active_slots.sort(key=lambda s: s["hotspot"]["cy"])

    def handle_event(self, event: pygame.event.Event):
        # ESC hanteras av GameScene
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

            # Bottom-center ankare:
            # hotspot cx,cy = figurens fotpunkt
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
            self._draw_top_text(screen, "MARKERA")

    # ---------- assets ----------
    def _load_scaled_background(self) -> pygame.Surface:
        bg_candidates = []
        for path in sorted(self.game_root.glob("b*.png")):
            if path.stem[1:].isdigit():
                bg_candidates.append(path)

        if not bg_candidates:
            fallback = self.game_root / "b1.png"
            bg_candidates = [fallback]

        bg_path = random.choice(bg_candidates)
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
            img = self._cleanup_fake_transparency(img)

            w, h = img.get_size()
            half = w // 2

            if half <= 0:
                continue

            friendly = img.subsurface((0, 0, half, h)).copy()
            hostile = img.subsurface((half, 0, w - half, h)).copy()

            friendly = self._crop_to_alpha_bounds(friendly)
            hostile = self._crop_to_alpha_bounds(hostile)

            silhouette = self._make_silhouette(friendly, hostile)

            out.append(
                {
                    "friendly": friendly,
                    "hostile": hostile,
                    "silhouette": silhouette,
                }
            )

        return out

    def _cleanup_fake_transparency(self, surf: pygame.Surface) -> pygame.Surface:
        """
        Försök ta bort enkel rut-/bakgrund som ligger inbakad i bilden.
        Den här är försiktig:
        - floodfyller från hörnen
        - tar bara bort pixlar som liknar hörnfärgerna
        - bara om de är ganska neutrala/grå
        """
        w, h = surf.get_size()
        if w == 0 or h == 0:
            return surf

        result = surf.copy()

        corners = [
            (0, 0),
            (w - 1, 0),
            (0, h - 1),
            (w - 1, h - 1),
        ]

        corner_colors = []
        for x, y in corners:
            c = result.get_at((x, y))
            corner_colors.append((int(c.r), int(c.g), int(c.b), int(c.a)))

        def is_neutral_grayish(r: int, g: int, b: int) -> bool:
            return abs(r - g) <= 18 and abs(r - b) <= 18 and abs(g - b) <= 18

        def similar_to_any_corner(r: int, g: int, b: int, a: int) -> bool:
            if a == 0:
                return False
            for cr, cg, cb, ca in corner_colors:
                if abs(r - cr) <= 22 and abs(g - cg) <= 22 and abs(b - cb) <= 22:
                    return True
            return False

        visited = [[False for _ in range(h)] for _ in range(w)]
        q = deque()

        for x, y in corners:
            q.append((x, y))
            visited[x][y] = True

        while q:
            x, y = q.popleft()
            c = result.get_at((x, y))
            r, g, b, a = int(c.r), int(c.g), int(c.b), int(c.a)

            if similar_to_any_corner(r, g, b, a) and is_neutral_grayish(r, g, b):
                result.set_at((x, y), (0, 0, 0, 0))

                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < w and 0 <= ny < h and not visited[nx][ny]:
                        visited[nx][ny] = True
                        q.append((nx, ny))

        return result

    def _crop_to_alpha_bounds(self, surf: pygame.Surface) -> pygame.Surface:
        rect = surf.get_bounding_rect(min_alpha=1)
        if rect.width <= 0 or rect.height <= 0:
            return surf
        return surf.subsurface(rect).copy()

    def _make_silhouette(self, friendly: pygame.Surface, hostile: pygame.Surface) -> pygame.Surface:
        w = max(friendly.get_width(), hostile.get_width())
        h = max(friendly.get_height(), hostile.get_height())

        base = pygame.Surface((w, h), pygame.SRCALPHA)
        base.blit(friendly, ((w - friendly.get_width()) // 2, h - friendly.get_height()))
        base.blit(hostile, ((w - hostile.get_width()) // 2, h - hostile.get_height()))

        silhouette = pygame.Surface((w, h), pygame.SRCALPHA)

        alpha_src = pygame.surfarray.array_alpha(base)
        rgb_dst = pygame.surfarray.pixels3d(silhouette)
        alpha_dst = pygame.surfarray.pixels_alpha(silhouette)

        rgb_dst[:, :, 0] = 0
        rgb_dst[:, :, 1] = 0
        rgb_dst[:, :, 2] = 0
        alpha_dst[:, :] = alpha_src[:, :]

        del rgb_dst
        del alpha_dst

        return silhouette

    # ---------- hotspots ----------
    def _extract_hotspots(self, mask_surface: pygame.Surface) -> list[dict]:
        w, h = mask_surface.get_size()
        visited = [[False for _ in range(h)] for _ in range(w)]
        hotspots: list[dict] = []

        def pixel_info(x: int, y: int) -> tuple[int, int, int, int, int]:
            c = mask_surface.get_at((x, y))
            gray = (int(c.r) + int(c.g) + int(c.b)) // 3
            return int(c.r), int(c.g), int(c.b), int(c.a), gray

        def is_hotspot_pixel(x: int, y: int) -> bool:
            _, _, _, a, _ = pixel_info(x, y)
            return a > 0

        for sx in range(w):
            for sy in range(h):
                if visited[sx][sy]:
                    continue

                visited[sx][sy] = True

                if not is_hotspot_pixel(sx, sy):
                    continue

                _, _, _, _, seed_gray = pixel_info(sx, sy)

                q = deque()
                q.append((sx, sy))

                pixels: list[tuple[int, int]] = []
                sum_gray = 0.0

                while q:
                    x, y = q.popleft()
                    _, _, _, a, gray = pixel_info(x, y)

                    if a <= 0:
                        continue

                    pixels.append((x, y))
                    sum_gray += gray

                    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if 0 <= nx < w and 0 <= ny < h and not visited[nx][ny]:
                            visited[nx][ny] = True

                            nr, ng, nb, na, ngray = pixel_info(nx, ny)
                            if na > 0 and abs(ngray - seed_gray) <= self.gray_tolerance:
                                q.append((nx, ny))

                if len(pixels) < self.min_hotspot_pixels:
                    continue

                xs = [p[0] for p in pixels]
                ys = [p[1] for p in pixels]

                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)

                # Fotpunkt: nedersta raden i området
                bottom_pixels = [x for x, y in pixels if y == max_y]
                if bottom_pixels:
                    cx = sum(bottom_pixels) // len(bottom_pixels)
                else:
                    cx = (min_x + max_x) // 2

                cy = max_y

                mean_gray = sum_gray / len(pixels)
                depth = mean_gray / 255.0  # 0 = svart/nära, 1 = vit/långt bort

                scale = self.max_scale - depth * (self.max_scale - self.min_scale)
                scale = max(self.min_scale, min(self.max_scale, scale))

                hotspots.append(
                    {
                        "cx": cx,
                        "cy": cy,
                        "depth": depth,
                        "gray": mean_gray,
                        "scale": scale,
                        "bounds": (min_x, min_y, max_x, max_y),
                        "pixel_count": len(pixels),
                    }
                )

        hotspots.sort(key=lambda hs: hs["cy"])
        return hotspots

    def _choose_spread_hotspots(self, count: int) -> list[dict]:
        if count >= len(self.hotspots):
            return self.hotspots[:]

        # Dela in i djup-buckets för lite bättre spridning
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
            chosen.extend(random.sample(remaining, min(len(remaining), count - len(chosen))))

        chosen.sort(key=lambda hs: hs["cy"])
        return chosen

    # ---------- helpers ----------
    def _scale_sprite(self, sprite: pygame.Surface, scale: float) -> pygame.Surface:
        key = (id(sprite), int(scale * 10000), sprite.get_width() ^ sprite.get_height())
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
        y = self.viewport.y + 20
        screen.blit(surf, (x, y))