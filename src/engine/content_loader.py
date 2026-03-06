from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MenuItem:
    id: str
    type: str  # "image" | "video" | "settings" | "game"
    title: str
    description: str
    path: str
    preview: str
    fit: str
    bg_color: tuple[int, int, int]
    script: str


@dataclass(frozen=True)
class MenuFolder:
    id: str
    title: str
    description: str
    preview: str
    children: list["MenuNode"]


MenuNode = MenuFolder | MenuItem


@dataclass(frozen=True)
class MenuData:
    title: str
    root: MenuFolder


def _parse_color(value, fallback=(0, 0, 0)) -> tuple[int, int, int]:
    try:
        if isinstance(value, list) and len(value) == 3:
            r = max(0, min(255, int(value[0])))
            g = max(0, min(255, int(value[1])))
            b = max(0, min(255, int(value[2])))
            return (r, g, b)
    except Exception:
        pass
    return fallback


def _parse_fit(value, fallback="stretch") -> str:
    v = str(value or "").lower().strip()
    if v in ("stretch", "contain", "cover"):
        return v
    return fallback


def _parse_item(raw: dict, inherited_fit: str, inherited_bg: tuple[int, int, int]) -> MenuItem:
    item_fit = _parse_fit(raw.get("fit", inherited_fit), inherited_fit)
    item_bg = _parse_color(raw.get("bg_color", list(inherited_bg)), inherited_bg)

    return MenuItem(
        id=str(raw["id"]),
        type=str(raw["type"]),
        title=str(raw.get("title", "")),
        description=str(raw.get("description", "")),
        path=str(raw.get("path", "")),
        preview=str(raw.get("preview", "")),
        fit=item_fit,
        bg_color=item_bg,
        script=str(raw.get("script", "")),
    )


def _parse_folder(raw: dict, inherited_fit: str, inherited_bg: tuple[int, int, int]) -> MenuFolder:
    defaults = raw.get("defaults", {}) or {}
    folder_fit = _parse_fit(defaults.get("fit", inherited_fit), inherited_fit)
    folder_bg = _parse_color(defaults.get("bg_color", list(inherited_bg)), inherited_bg)

    children: list[MenuNode] = []
    for child in raw.get("children", []):
        kind = str(child.get("kind", "")).lower().strip()

        if kind == "folder":
            children.append(_parse_folder(child, folder_fit, folder_bg))
        elif kind == "item":
            children.append(_parse_item(child, folder_fit, folder_bg))
        else:
            if "children" in child:
                children.append(_parse_folder(child, folder_fit, folder_bg))
            else:
                children.append(_parse_item(child, folder_fit, folder_bg))

    return MenuFolder(
        id=str(raw.get("id", "")),
        title=str(raw.get("title", "")),
        description=str(raw.get("description", "")),
        preview=str(raw.get("preview", "")),
        children=children,
    )


def _load_tree_format(data: dict) -> MenuData:
    if "root" not in data:
        raise ValueError("menu.json saknar 'root'")

    root = _parse_folder(data["root"], "stretch", (0, 0, 0))
    return MenuData(
        title=str(data.get("title", "Menu")),
        root=root,
    )


def _load_legacy_categories_format(data: dict) -> MenuData:
    root_children: list[MenuNode] = []

    for c in data.get("categories", []):
        defaults = c.get("defaults", {}) or {}
        cat_fit = _parse_fit(defaults.get("fit", "stretch"), "stretch")
        cat_bg = _parse_color(defaults.get("bg_color", [0, 0, 0]), (0, 0, 0))

        children: list[MenuNode] = []
        for it in c.get("items", []):
            children.append(_parse_item(it, cat_fit, cat_bg))

        root_children.append(
            MenuFolder(
                id=str(c.get("id", "")),
                title=str(c.get("title", "")),
                description=str(c.get("description", "")),
                preview=str(c.get("preview", "")),
                children=children,
            )
        )

    root = MenuFolder(
        id="root",
        title="Huvudmeny",
        description="",
        preview="",
        children=root_children,
    )

    return MenuData(
        title=str(data.get("title", "Menu")),
        root=root,
    )


def load_menu(path: str | Path) -> MenuData:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    if data.get("version") != 1:
        raise ValueError("Unsupported menu.json version")

    if "root" in data:
        return _load_tree_format(data)

    return _load_legacy_categories_format(data)