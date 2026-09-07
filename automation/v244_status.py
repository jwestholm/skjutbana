from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def yesno(value: bool) -> str:
    return "YES" if value else "NO"


def main() -> None:
    runtime = ROOT / "src/engine/shot_object_local_v244.py"
    main_py = ROOT / "main.py"
    scene = ROOT / "content/games/hit_context_test_v242.py"
    menu = ROOT / "content/menu.json"
    print("V2.24.4 STATUS")
    print("==============")
    print(f"runtime file: {yesno(runtime.exists())}")
    print(f"main hook: {yesno(main_py.exists() and 'install_v244_runtime(App)' in main_py.read_text(encoding='utf-8'))}")
    print(f"V2.24.4 testscene: {yesno(scene.exists() and 'V2.24.4 HIT CONTEXT TEST' in scene.read_text(encoding='utf-8'))}")
    print(f"menu prepared: {yesno(menu.exists() and 'Hit Context Test (V2.24.4)' in menu.read_text(encoding='utf-8'))}")
    print("Expected physical logs: [V2.24.4 ROI-MAP] + [V2.24.4 LOCAL-ROI] on region shots; optional ROI-RECOVERY; GLOBAL-RESCUE-ROI only for V2.22.5 rescue.")


if __name__ == "__main__":
    main()
