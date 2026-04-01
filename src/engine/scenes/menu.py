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
    def __init__(self, menu_state: dict | None = None) -> None:
        self.menu_data: MenuData | None = None
        self.font = None
        self.big = None
        self.small = None
        self.background = None
        self.overlay = None
        self._preview_cache: dict[str, pygame.Surface] = {}

        self.folder_stack: list[MenuFolder] = []
        self.index_stack: list[int] = []

        self._initial_menu_state = menu_state or {}

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 34)
        self.big = pygame.font.Font(None, 56)
        self.small = pygame.font.Font(None, 26)

        bg = pygame.image.load(str(LOADING_SCREEN_PATH)).convert()
        self.background = pygame.transform.smoothscale(bg, (SCREEN_WIDTH, SCREEN_HEIGHT))

        self.overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.overlay.fill((0, 0, 0, 140))

        self.menu_data = load_menu(MENU_JSON_PATH)
        self._restore_menu_state()

    def export_menu_state(self) -> dict:
        return {
            "folder_ids": [folder.id for folder in self.folder_stack[1:]],
            "index_stack": list(self.index_stack),
        }

    def _restore_menu_state(self) -> None:
        assert self.menu_data is not None

        self.folder_stack = [self.menu_data.root]
        self.index_stack = [0]

        folder_ids = self._initial_menu_state.get("folder_ids", [])
        raw_index_stack = self._initial_menu_state.get("index_stack", [])

        if not isinstance(folder_ids, list):
            folder_ids = []
        if not isinstance(raw_index_stack, list):
            raw_index_stack = []

        current_folder = self.menu_data.root
        restored_folder_stack = [current_folder]

        for folder_id in folder_ids:
            if not isinstance(folder_id, str):
                break
            next_folder = self._find_child_folder(current_folder, folder_id)
            if next_folder is None:
                break
            restored_folder_stack.append(next_folder)
            current_folder = next_folder

        self.folder_stack = restored_folder_stack

        desired_depth = len(self.folder_stack)
        restored_index_stack: list[int] = []
        for i in range(desired_depth):
            if i < len(raw_index_stack):
                try:
                    restored_index_stack.append(int(raw_index_stack[i]))
                except Exception:
                    restored_index_stack.append(0)
            else:
                restored_index_stack.append(0)

        self.index_stack = restored_index_stack

        for level in range(len(self.index_stack)):
            self._clamp_index_for_level(level)

    def _find_child_folder(self, parent: MenuFolder, folder_id: str) -> MenuFolder | None:
        for child in parent.children:
            if isinstance(child, MenuFolder) and child.id == folder_id:
                return child
        return None

    def _entries_for_folder(self, folder: MenuFolder, is_root: bool) -> list[_BackEntry | MenuNode]:
        entries: list[_BackEntry | MenuNode] = []
        if not is_root:
            entries.append(_BackEntry())
        entries.extend(folder.children)
        return entries

    def _clamp_index_for_level(self, level: int) -> None:
        entries = self._entries_for_folder(self.folder_stack[level], is_root=(level == 0))
        if not entries:
            self.index_stack[level] = 0
            return
        self.index_stack[level] = max(0, min(self.index_stack[level], len(entries) - 1))

    def _current_folder(self) -> MenuFolder:
        return self.folder_stack[-1]

    def _is_root(self) -> bool:
        return len(self.folder_stack) == 1

    def _current_entries(self) -> list[_BackEntry | MenuNode]:
        return self._entries_for_folder(self._current_folder(), self._is_root())

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
            return SceneSwitch(
                build_scene_from_item(
                    selected,
                    return_menu_state=self.export_menu_state(),
                )
            )

        return None

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

        if len(self.folder_stack) > 1:
            for folder in self.folder_stack[1:]:
                parts.append(folder.title)

        return " / ".join(parts)

    def _entry_label(self, entry: _BackEntry | MenuNode) -> str:
        if isinstance(entry, _BackEntry):
            return " Tillbaka"
        if isinstance(entry, MenuFolder):
            return entry.title
        return entry.title

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
            prefix = "> " if is_sel else "  "
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

        hint = "UP/DOWN: välj ENTER: öppna/starta ESC: tillbaka/avsluta (SPACE paus i video)"
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