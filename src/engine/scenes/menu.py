from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygame

from config import LOADING_SCREEN_PATH, SCREEN_HEIGHT, SCREEN_WIDTH
from src.engine.content_loader import MenuData, MenuFolder, MenuItem, MenuNode, load_menu
from src.engine.scene import Scene, SceneSwitch
from src.engine.scene_factory import build_scene_from_item

MENU_JSON_PATH = Path("content/menu.json")


@dataclass(frozen=True)
class _BackEntry:
    title: str = "Tillbaka"
    description: str = "Gå tillbaka ett steg i menyträdet."
    preview: str = ""


class MenuScene(Scene):
    def __init__(self) -> None:
        self.menu_data: MenuData | None = None

        self.font = None
        self.big = None
        self.small = None

        self.background = None
        self.overlay = None

        self._preview_cache: dict[str, pygame.Surface] = {}

        # Trädnavigation
        self.folder_stack: list[MenuFolder] = []
        self.index_stack: list[int] = []

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 34)
        self.big = pygame.font.Font(None, 56)
        self.small = pygame.font.Font(None, 26)

        bg = pygame.image.load(str(LOADING_SCREEN_PATH)).convert()
        self.background = pygame.transform.smoothscale(bg, (SCREEN_WIDTH, SCREEN_HEIGHT))

        self.overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.overlay.fill((0, 0, 0, 140))

        self.menu_data = load_menu(MENU_JSON_PATH)

        self.folder_stack = [self.menu_data.root]
        self.index_stack = [0]

    # ---------- navigation helpers ----------
    def _current_folder(self) -> MenuFolder:
        return self.folder_stack[-1]

    def _is_root(self) -> bool:
        return len(self.folder_stack) == 1

    def _current_entries(self) -> list[_BackEntry | MenuNode]:
        entries: list[_BackEntry | MenuNode] = []

        if not self._is_root():
            entries.append(_BackEntry())

        entries.extend(self._current_folder().children)
        return entries

    def _current_index(self) -> int:
        return self.index_stack[-1]

    def _set_current_index(self, value: int) -> None:
        entries = self._current_entries()
        if not entries:
            self.index_stack[-1] = 0
            return
        self.index_stack[-1] = value % len(entries)

    def _selected_entry(self) -> _BackEntry | MenuNode | None:
        entries = self._current_entries()
        if not entries:
            return None
        idx = self._current_index()
        if idx < 0 or idx >= len(entries):
            return None
        return entries[idx]

    def _move_selection(self, delta: int) -> None:
        entries = self._current_entries()
        if not entries:
            return
        self._set_current_index(self._current_index() + delta)

    def _go_back(self):
        if self._is_root():
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            return None

        self.folder_stack.pop()
        self.index_stack.pop()
        return None

    def _enter_selected(self):
        selected = self._selected_entry()
        if selected is None:
            return None

        if isinstance(selected, _BackEntry):
            return self._go_back()

        if isinstance(selected, MenuFolder):
            self.folder_stack.append(selected)
            self.index_stack.append(0)
            return None

        if isinstance(selected, MenuItem):
            return SceneSwitch(build_scene_from_item(selected))

        return None

    # ---------- preview helpers ----------
    def _load_preview(self, path: str) -> pygame.Surface:
        if not path:
            surf = pygame.Surface((640, 360))
            surf.fill((30, 30, 30))
            return surf

        if path in self._preview_cache:
            return self._preview_cache[path]

        try:
            img = pygame.image.load(path).convert()
        except Exception:
            surf = pygame.Surface((640, 360))
            surf.fill((30, 30, 30))
            self._preview_cache[path] = surf
            return surf

        max_w = int(SCREEN_WIDTH * 0.42)
        max_h = int(SCREEN_HEIGHT * 0.42)
        w, h = img.get_size()
        scale = min(max_w / w, max_h / h)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = pygame.transform.smoothscale(img, new_size)

        self._preview_cache[path] = img
        return img

    def _focused_info(self) -> tuple[str, str, str]:
        selected = self._selected_entry()

        if selected is None:
            folder = self._current_folder()
            return folder.title, folder.description, folder.preview

        if isinstance(selected, _BackEntry):
            folder = self._current_folder()
            return selected.title, selected.description, folder.preview

        return selected.title, selected.description, selected.preview

    def _breadcrumb(self) -> str:
        assert self.menu_data is not None

        parts = [self.menu_data.title]

        # visa inte root-folderns titel om den bara är "Huvudmeny"
        if len(self.folder_stack) > 1:
            for folder in self.folder_stack[1:]:
                parts.append(folder.title)

        return " / ".join(parts)

    def _entry_label(self, entry: _BackEntry | MenuNode) -> str:
        if isinstance(entry, _BackEntry):
            return "◀ Tillbaka"

        if isinstance(entry, MenuFolder):
            return f"[Mapp] {entry.title}"

        return entry.title

    # ---------- input ----------
    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return self._go_back()

            if event.key in (pygame.K_UP, pygame.K_w):
                self._move_selection(-1)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self._move_selection(+1)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                return self._enter_selected()

        return None

    # ---------- render ----------
    def render(self, screen: pygame.Surface) -> None:
        screen.blit(self.background, (0, 0))
        screen.blit(self.overlay, (0, 0))

        assert self.menu_data is not None

        pad = 40
        left_w = int(SCREEN_WIDTH * 0.46)
        right_x = pad + left_w + 30
        top_y = 30

        crumb = self._breadcrumb()
        title = self.big.render(crumb, True, (240, 240, 240))
        screen.blit(title, (pad, top_y))

        list_y = top_y + 90
        list_h = SCREEN_HEIGHT - list_y - 90
        list_rect = pygame.Rect(pad, list_y, left_w, list_h)

        panel = pygame.Surface((list_rect.w, list_rect.h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 90))
        screen.blit(panel, list_rect.topleft)

        entries = self._current_entries()
        selected = self._current_index()

        y = list_rect.y + 18
        line_h = 44
        for i, entry in enumerate(entries):
            is_sel = i == selected
            prefix = "▶ " if is_sel else "  "
            color = (255, 255, 255) if is_sel else (185, 185, 185)
            txt = self.font.render(prefix + self._entry_label(entry), True, color)
            screen.blit(txt, (list_rect.x + 18, y))
            y += line_h
            if y > list_rect.bottom - 10:
                break

        focus_title, focus_desc, focus_preview = self._focused_info()

        prev = self._load_preview(focus_preview)
        prev_x = right_x
        prev_y = list_y
        screen.blit(prev, (prev_x, prev_y))

        t = self.font.render(focus_title, True, (240, 240, 240))
        screen.blit(t, (right_x, prev_y + prev.get_height() + 18))

        self._draw_wrapped_text(
            screen,
            focus_desc,
            (right_x, prev_y + prev.get_height() + 55),
            max_width=SCREEN_WIDTH - right_x - pad,
            color=(200, 200, 200),
        )

        hint = "UP/DOWN: välj   ENTER: öppna/starta   ESC: tillbaka/avsluta   (SPACE paus i video)"
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