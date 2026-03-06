from __future__ import annotations

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

        self._scaled_cache: dict[tuple[int, int], pygame.Surface] = {}

        # Justerad skala efter din screenshot:
        # svart = nära = större
        # vitt = långt bort = mindre
        self.min_scale = 0.12
        self.max_scale = 0.45

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
        if max_people <= 0:
            self.active_slots = []
            return

        min_people = min(3, max_people)
        num_people = random.randint(min_people, max_people)

        chosen_hotspots = random.sample(self.hotspots, num_people)
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

            # Bottom-center ankare
            x = self.viewport.x + hotspot["cx"] - scaled.get_width() // 2
            y = self.viewport.y + hotspot["cy"] - scaled.get_height()

            screen.blit(scaled, (x, y))

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
            img = self._make_white_transparent(img)

            w, h = img.get_size()
            half = w // 2

            if half <= 0:
                continue

            friendly = img.subsurface((0, 0, half, h)).copy()
            hostile = img.subsurface((half, 0, w - half, h)).copy()

            silhouette = self._make_silhouette(friendly, hostile)

            out.append(
                {
                    "friendly": friendly,
                    "hostile": hostile,
                    "silhouette": silhouette,
                }
            )

        return out

    def _make_white_transparent(self, surf: pygame.Surface) -> pygame.Surface:
        """
        Gör nästan-vita bakgrunder transparenta.
        Detta hjälper om vissa PNG-filer egentligen saknar riktig alpha.
        """
        result = surf.copy().convert_alpha()
        w, h = result.get_size()

        for y in range(h):
            for x in range(w):
                c = result.get_at((x, y))
                if c.a == 0:
                    continue

                if c.r >= 245 and c.g >= 245 and c.b >= 245:
                    result.set_at((x, y), (255, 255, 255, 0))

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
        Enkel och snabb hotspot-extraktion:
        - sampla masken glest
        - alla icke-transparenta pixlar är giltiga
        - gråvärde styr scale
        - gruppera bort punkter som ligger för nära varandra
        """
        w, h = mask_surface.get_size()
        hotspots: list[dict] = []

        step_x = max(12, w // 80)
        step_y = max(12, h // 45)
        min_distance = max(24, min(w, h) // 18)

        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                c = mask_surface.get_at((x, y))
                if c.a == 0:
                    continue

                gray = (int(c.r) + int(c.g) + int(c.b)) / 3.0
                depth = gray / 255.0  # 0 = svart/nära, 1 = vit/långt bort
                scale = self.max_scale - depth * (self.max_scale - self.min_scale)

                candidate = {
                    "cx": x,
                    "cy": y,
                    "depth": depth,
                    "scale": scale,
                }

                too_close = False
                for hs in hotspots:
                    dx = hs["cx"] - candidate["cx"]
                    dy = hs["cy"] - candidate["cy"]
                    if (dx * dx + dy * dy) < (min_distance * min_distance):
                        too_close = True
                        break

                if not too_close:
                    hotspots.append(candidate)

        hotspots.sort(key=lambda hs: hs["cy"])
        return hotspots

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
        surf = self.font_big.render(text, True, (255, 255, 255))
        x = self.viewport.x + (self.viewport.w - surf.get_width()) // 2
        y = self.viewport.y + 40
        screen.blit(surf, (x, y))

    def _draw_top_text(self, screen: pygame.Surface, text: str) -> None:
        surf = self.font_big.render(text, True, (255, 255, 255))
        x = self.viewport.x + (self.viewport.w - surf.get_width()) // 2
        y = self.viewport.y + 20
        screen.blit(surf, (x, y))