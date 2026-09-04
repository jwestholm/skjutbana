from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def yesno(value: bool) -> str:
    return "YES" if value else "NO"


def main() -> None:
    print("V2.25.0 STATUS")
    print("===============")
    files = [
        "src/engine/game_objects/__init__.py",
        "src/engine/game_objects/model.py",
        "src/engine/game_objects/manager.py",
        "src/engine/shot_context_v250.py",
        "content/games/game_objects_test_v250.py",
        "GAME_OBJECT_SYSTEM.md",
        "AI_GAME_OBJECTS.md",
    ]
    for rel in files:
        print(f"{rel}: {yesno((ROOT / rel).exists())}")
    main_py = ROOT / "main.py"
    print(f"runtime hook: {yesno(main_py.exists() and 'install_v250_runtime(App)' in main_py.read_text(encoding='utf-8'))}")
    menu = ROOT / "content/menu.json"
    print(f"menu prepared: {yesno(menu.exists() and 'game_objects_test_v250' in menu.read_text(encoding='utf-8'))}")
    print("Hit authority: physical HitEvent XY unchanged")
    print("Camera HitEvent: scanner shot_id carried before game subscribers")
    print("Frozen collision: exact V2.25 shape metadata selected by shot_id")
    print("Object model: composition (shape/body/damage/motion/reactions), living/breakable are presets")
    print("Effects: event requests only; no audio/particle/physics service in V2.25.0")
    print("Next candidate: V2.25.1 moving-object continuity using shot_id snapshots")


if __name__ == "__main__":
    main()
