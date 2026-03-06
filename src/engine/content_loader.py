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

    fit: str  # "stretch" | "contain" | "cover"
    bg_color: tuple[int, int, int]

    # Nytt: används av type="game"
    script: str


@dataclass(frozen=True)
class Category:
    id: str
    title: str
    description: str
    preview: str
    items: list[MenuItem]


@dataclass(frozen=True)
class MenuData:
    title: str
    categories: list[Category]


def _parse_color(value, fallback=(0, 0, 0)) -> tuple[int, int, int]:
    try:
        if isinstance(value, list) and len(value) == 3:
            r = int(value[0])
            g = int(value[1])
            b = int(value[2])
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            return (r, g, b)
    except Exception:
        pass
    return fallback


def _parse_fit(value, fallback="stretch") -> str:
    v = str(value or "").lower().strip()
    if v in ("stretch", "contain", "cover"):
        return v
    return fallback


def load_menu(path: str | Path) -> MenuData:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    if data.get("version") != 1:
        raise ValueError("Unsupported menu.json version")

    categories: list[Category] = []

    for c in data.get("categories", []):
        defaults = c.get("defaults", {}) or {}
        cat_fit = _parse_fit(defaults.get("fit", "stretch"), "stretch")
        cat_bg = _parse_color(defaults.get("bg_color", [0, 0, 0]), (0, 0, 0))

        items: list[MenuItem] = []
        for it in c.get("items", []):
            item_fit = _parse_fit(it.get("fit", cat_fit), cat_fit)
            item_bg = _parse_color(it.get("bg_color", list(cat_bg)), cat_bg)

            items.append(
                MenuItem(
                    id=str(it["id"]),
                    type=str(it["type"]),
                    title=str(it.get("title", "")),
                    description=str(it.get("description", "")),
                    path=str(it.get("path", "")),
                    preview=str(it.get("preview", "")),
                    fit=item_fit,
                    bg_color=item_bg,
                    script=str(it.get("script", "")),
                )
            )

        categories.append(
            Category(
                id=str(c["id"]),
                title=str(c.get("title", "")),
                description=str(c.get("description", "")),
                preview=str(c.get("preview", "")),
                items=items,
            )
        )

    return MenuData(title=str(data.get("title", "Menu")), categories=categories)