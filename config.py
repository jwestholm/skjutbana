from pathlib import Path

# --- Window / timing ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Default "safe drawing area" (x, y, w, h) inom skärmen.
# Används om content/settings.json inte finns ännu.
SETTINGS_PATH = "content/settings.json"
DEFAULT_VIEWPORT = (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)

# --- Assets ---
ASSETS_DIR = Path("assets")
LOADING_SCREEN_PATH = ASSETS_DIR / "loading_screen.png"
HOSTAGE_MOVIE_PATH = ASSETS_DIR / "movies" / "hostage.mp4"