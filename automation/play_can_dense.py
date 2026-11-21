"""Start Can Shootout directly with the dense OBJECT decision and raw source recording (live test helper).

    cd <checkout> && python3 automation/play_can_dense.py
Sets (unless already set): SKJUTBANA_OBJECT_DENSE=1, SKJUTBANA_SOURCE_RECORD=1 (raw camera+audio to
/data/skjutbana/source_sessions, ~1.5 GB per 4 min), SKJUTBANA_PHYSICAL_TRACE_DISABLE=1,
SKJUTBANA_AUDIO_COOLDOWN_S=0.15 (merge the 65-97 ms follower peak; audio_shot_signature_20261121).
Every shot prints one [OBJECT-HIT] and one [OBJECT-DENSE] line (reason, frames, dense peaks).
"""
import os
import sys

for k, v in (("SKJUTBANA_OBJECT_DENSE", "1"), ("SKJUTBANA_SOURCE_RECORD", "1"), ("SKJUTBANA_PHYSICAL_TRACE_DISABLE", "1"),
             ("SKJUTBANA_AUDIO_COOLDOWN_S", "0.15")):
    os.environ.setdefault(k, v)
# Measured x correction (+1.35 game px; projected dots 2026-11-21, held-out |dx| 1.31 -> 0.42) is only
# valid for the calibration it was fitted on.
import json  # noqa: E402
try:
    _cal = json.load(open("content/settings.json"))["camera_calibration"].get("calibrated_at")
except Exception:
    _cal = None
if _cal == "2026-11-21T17:55:49":
    os.environ.setdefault("SKJUTBANA_OBJECT_DENSE_GAME_DX", "1.35")
else:
    print(f"[play_can_dense] calibration {_cal!r} differs from the fitted one: no x correction", flush=True)
# SHADOW-only candidate-peak model (logs [OBJECT-DENSE-MODEL]; never decides gameplay)
_MODEL = "/data/skjutbana/evaluation_runs/can_peak_model_20261121/v4/model.json"   # logreg v4, tau .994
if os.path.exists(_MODEL):
    os.environ.setdefault("SKJUTBANA_OBJECT_DENSE_MODEL", _MODEL)
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
