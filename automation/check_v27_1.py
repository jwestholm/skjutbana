from pathlib import Path

root = Path(__file__).resolve().parents[1]
camera = root / "src/engine/camera/__init__.py"
bootstrap = root / "src/engine/ai/bootstrap.py"

camera_text = camera.read_text(encoding="utf-8") if camera.exists() else ""
bootstrap_text = bootstrap.read_text(encoding="utf-8") if bootstrap.exists() else ""

print("V2.7.1 static install status")
print("---------------------------")
print("V2.7 files present:", all((root / p).exists() for p in [
    "src/engine/ai/hypothesis_v27.py",
    "src/engine/ai/ranker_v6.py",
    "src/engine/ai/ranker_v6_extension.py",
]))
print("Camera autoload hook:", "# --- V2.7.1 RANKER-V6 AUTOLOAD ---" in camera_text)
print("Bootstrap V2.7 hook:", (
    "def _patch_ranker_v6(" in bootstrap_text and "_patch_ranker_v6()" in bootstrap_text
) or "# --- V2.7.1 RANKER-V6 BOOTSTRAP FALLBACK ---" in bootstrap_text)
print()
print("Runtime proof still requires a full game restart and the startup line:")
print("[RANKER-V6] V2.7 hypothesis clustering + validated ranker installed")
