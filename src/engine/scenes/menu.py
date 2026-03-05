from __future__ import annotations
import pygame
from pathlib import Path

from config import SCREEN_WIDTH, SCREEN_HEIGHT, LOADING_SCREEN_PATH
from src.engine.scene import Scene, SceneSwitch
from src.engine.content_loader import load_menu, MenuData, Category, MenuItem
from src.engine.scene_factory import build_scene_from_item


MENU_JSON_PATH = Path("content/menu.json")


class MenuScene(Scene):
    def __init__(self) -> None:
        self.menu_data: MenuData | None = None

        self.level = "categories"  # "categories" | "items"
        self.cat_index = 0
        self.item_index = 0

        self.font = None
        self.big = None
        self.small = None

        self.background = None
        self.overlay = None

        # cache för preview surfaces så det inte hackar
        self._preview_cache: dict[str, pygame.Surface] = {}

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 34)
        self.big = pygame.font.Font(None, 56)
        self.small = pygame.font.Font(None, 26)

        bg = pygame.image.load(str(LOADING_SCREEN_PATH)).convert()
        self.background = pygame.transform.smoothscale(bg, (SCREEN_WIDTH, SCREEN_HEIGHT))

        self.overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.overlay.fill((0, 0, 0, 140))

        self.menu_data = load_menu(MENU_JSON_PATH)

        self.level = "categories"
        self.cat_index = 0
        self.item_index = 0

    # ---------- navigation helpers ----------
    def _current_category(self) -> Category:
        assert self.menu_data is not None
        cats = self.menu_data.categories
        return cats[max(0, min(self.cat_index, len(cats) - 1))]

    def _current_item(self) -> MenuItem:
        cat = self._current_category()
        items = cat.items
        return items[max(0, min(self.item_index, len(items) - 1))]

    def _move_selection(self, delta: int) -> None:
        if self.level == "categories":
            cats = self.menu_data.categories
            self.cat_index = (self.cat_index + delta) % max(1, len(cats))
        else:
            items = self._current_category().items
            self.item_index = (self.item_index + delta) % max(1, len(items))

    def _enter(self):
        if self.level == "categories":
            # gå in i items
            self.level = "items"
            self.item_index = 0
            return None
        else:
            # starta bana
            item = self._current_item()
            return SceneSwitch(build_scene_from_item(item))

    def _back_or_quit(self):
        if self.level == "items":
            self.level = "categories"
            self.item_index = 0
            return None
        # ESC på root-menyn = avsluta
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        return None

    # ---------- preview helpers ----------
    def _load_preview(self, path: str) -> pygame.Surface:
        # cache key = path
        if path in self._preview_cache:
            return self._preview_cache[path]

        try:
            img = pygame.image.load(path).convert()
        except Exception:
            # fallback: tom ruta
            surf = pygame.Surface((640, 360))
            surf.fill((30, 30, 30))
            self._preview_cache[path] = surf
            return surf

        # Preview-rutan: 640x360-ish, men anpassa efter högerpanel
        max_w = int(SCREEN_WIDTH * 0.42)
        max_h = int(SCREEN_HEIGHT * 0.42)
        w, h = img.get_size()
        scale = min(max_w / w, max_h / h)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = pygame.transform.smoothscale(img, new_size)

        self._preview_cache[path] = img
        return img

    # ---------- input ----------
    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return self._back_or_quit()

            if event.key in (pygame.K_UP, pygame.K_w):
                self._move_selection(-1)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self._move_selection(+1)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                return self._enter()

        return None

    # ---------- render ----------
    def render(self, screen: pygame.Surface) -> None:
        screen.blit(self.background, (0, 0))
        screen.blit(self.overlay, (0, 0))

        assert self.menu_data is not None

        # Layout
        pad = 40
        left_w = int(SCREEN_WIDTH * 0.46)
        right_x = pad + left_w + 30
        top_y = 30

        # Breadcrumb / title
        if self.level == "categories":
            crumb = f"{self.menu_data.title} / Kategorier"
        else:
            crumb = f"{self.menu_data.title} / {self._current_category().title}"

        title = self.big.render(crumb, True, (240, 240, 240))
        screen.blit(title, (pad, top_y))

        # Left list box
        list_y = top_y + 90
        list_h = SCREEN_HEIGHT - list_y - 90
        list_rect = pygame.Rect(pad, list_y, left_w, list_h)

        # faint panel background
        panel = pygame.Surface((list_rect.w, list_rect.h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 90))
        screen.blit(panel, list_rect.topleft)

        # Build list entries
        if self.level == "categories":
            entries = [(c.title, c.description, c.preview) for c in self.menu_data.categories]
            selected = self.cat_index
        else:
            cat = self._current_category()
            entries = [(it.title, it.description, it.preview) for it in cat.items]
            selected = self.item_index

        # Draw entries
        y = list_rect.y + 18
        line_h = 44
        for i, (name, _desc, _prev) in enumerate(entries):
            is_sel = (i == selected)
            prefix = "▶ " if is_sel else "  "
            color = (255, 255, 255) if is_sel else (185, 185, 185)
            txt = self.font.render(prefix + name, True, color)
            screen.blit(txt, (list_rect.x + 18, y))
            y += line_h
            if y > list_rect.bottom - 10:
                break

        # Right preview + description
        # Decide what is "focused" for preview
        if self.level == "categories":
            focus_title = self._current_category().title
            focus_desc = self._current_category().description
            focus_preview = self._current_category().preview
        else:
            focus_title = self._current_item().title
            focus_desc = self._current_item().description
            focus_preview = self._current_item().preview

        # preview image
        prev = self._load_preview(focus_preview)
        prev_x = right_x
        prev_y = list_y
        screen.blit(prev, (prev_x, prev_y))

        # focus title
        t = self.font.render(focus_title, True, (240, 240, 240))
        screen.blit(t, (right_x, prev_y + prev.get_height() + 18))

        # description (simple word wrap)
        self._draw_wrapped_text(
            screen,
            focus_desc,
            (right_x, prev_y + prev.get_height() + 55),
            max_width=SCREEN_WIDTH - right_x - pad,
            color=(200, 200, 200),
        )

        # Footer hints
        hint = "UP/DOWN: välj   ENTER: öppna/starta   ESC: back/quit   (SPACE paus i video)"
        htxt = self.small.render(hint, True, (160, 160, 160))
        screen.blit(htxt, (pad, SCREEN_HEIGHT - 45))

    def _draw_wrapped_text(self, screen, text, pos, max_width, color):
        words = (text or "").split()
        if not words:
            return
        x, y = pos
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            surf = self.small.render(test, True, color)
            if surf.get_width() <= max_width:
                line = test
            else:
                screen.blit(self.small.render(line, True, color), (x, y))
                y += 26
                line = w
        if line:
            screen.blit(self.small.render(line, True, color), (x, y))