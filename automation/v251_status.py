from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def yn(value: bool) -> str:
    return "YES" if value else "NO"


def main() -> None:
    print("V2.25.1 STATUS")
    print("===============")
    source = ROOT / "src/engine/shot_region_proposal_v251.py"
    main_py = ROOT / "main.py"
    menu = ROOT / "content/menu.json"
    print(f"runtime module: {yn(source.exists())}")
    print(f"runtime hook: {yn(main_py.exists() and 'install_v251_runtime(App)' in main_py.read_text(encoding='utf-8'))}")
    print(f"test scene: {yn((ROOT/'content/games/game_objects_test_v250.py').exists())}")
    print(f"menu prepared: {yn(menu.exists() and 'Game Objects Test (V2.25.1)' in menu.read_text(encoding='utf-8'))}")
    print("First pass: one physical proposal opportunity per frozen region group")
    print("Overlap: near-identical areas grouped; object identities retained")
    print("Confirmation: bounded per region, <= configured global total")
    print("Track selection: confirmed physical evidence; NO role/target preference")
    print("FULL rescue: V2.22.5 global path preserved")
    print("XY snapping: NEVER")
    print("GameObject model/damage/penetration: unchanged from V2.25.0")
    print("Next after physical acceptance: V2.25.2 moving-object continuity")


if __name__ == "__main__":
    main()
