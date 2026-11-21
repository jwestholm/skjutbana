"""Start Can Shootout directly with the dense OBJECT decision and raw source recording (live test helper).

    cd <checkout> && python3 automation/play_can_dense.py
Sets (unless already set): SKJUTBANA_OBJECT_DENSE=1, SKJUTBANA_SOURCE_RECORD=1 (raw camera+audio to
/data/skjutbana/source_sessions, ~1.5 GB per 4 min), SKJUTBANA_PHYSICAL_TRACE_DISABLE=1.
Every shot prints one [OBJECT-HIT] and one [OBJECT-DENSE] line (reason, frames, dense peaks).
"""
import os
import sys

for k, v in (("SKJUTBANA_OBJECT_DENSE", "1"), ("SKJUTBANA_SOURCE_RECORD", "1"), ("SKJUTBANA_PHYSICAL_TRACE_DISABLE", "1")):
    os.environ.setdefault(k, v)
sys.path.insert(0, os.getcwd())
sys.argv = [sys.argv[0]]
import main  # noqa: F401,E402  installs runtimes exactly like main.py
from src.engine.app import App  # noqa: E402
from src.engine.scenes.game import GameScene  # noqa: E402
from src.engine.scenes.overlay_scene import OverlayScene  # noqa: E402

app = App()
app._switch_to(OverlayScene(GameScene(game_root="", script_path="content/games/can_shootout_game.py")))
try:
    app._automation_window.position = (2135, 95)
except Exception:
    pass
app.run()
