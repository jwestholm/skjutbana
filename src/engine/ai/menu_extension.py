from __future__ import annotations

from src.engine.content_loader import MenuData, MenuFolder, MenuItem


AI_FOLDER_ID = "ai_tools"


def build_ai_menu_folder() -> MenuFolder:
    return MenuFolder(
        id=AI_FOLDER_ID,
        title="AI",
        description="AI-inställningar och träningsverktyg för träffdetektering.",
        preview="",
        children=[
            MenuItem(
                id="ai_settings",
                type="ai_settings",
                title="AI – Inställningar",
                description="Ställ in hur mycket systemet ska lita på AI:n och se modellstatus.",
                path="",
                preview="",
                fit="contain",
                bg_color=(0, 0, 0),
                script="",
                led_enabled=False,
                led_color=(255, 255, 255),
            ),
            MenuItem(
                id="ai_training",
                type="ai_training",
                title="AI-träning",
                description="Skjut, se kandidatlistan och klicka ungefär där du träffade så lär sig modellen.",
                path="",
                preview="",
                fit="contain",
                bg_color=(0, 0, 0),
                script="",
                led_enabled=False,
                led_color=(255, 255, 255),
            ),
        ],
    )



def augment_menu(menu_data: MenuData) -> MenuData:
    root = menu_data.root
    if any(getattr(child, "id", "") == AI_FOLDER_ID for child in root.children):
        return menu_data
    new_children = list(root.children) + [build_ai_menu_folder()]
    new_root = MenuFolder(
        id=root.id,
        title=root.title,
        description=root.description,
        preview=root.preview,
        children=new_children,
    )
    return MenuData(title=menu_data.title, root=new_root)
