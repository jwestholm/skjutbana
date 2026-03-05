from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MenuItem:
    id: str
    type: str          # "image" | "video" | (senare "script")
    title: str
    description: str
    path: str
    preview: str


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


def load_menu(path: str | Path) -> MenuData:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    if data.get("version") != 1:
        raise ValueError("Unsupported menu.json version")

    categories: list[Category] = []
    for c in data.get("categories", []):
        items: list[MenuItem] = []
        for it in c.get("items", []):
            items.append(
                MenuItem(
                    id=str(it["id"]),
                    type=str(it["type"]),
                    title=str(it.get("title", "")),
                    description=str(it.get("description", "")),
                    path=str(it.get("path", "")),
                    preview=str(it.get("preview", "")),
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