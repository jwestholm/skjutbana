"""Ljusjakten launcher. Only running main() starts hardware; --api uses the existing app API."""
import argparse
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--api', action='store_true', help='start registered game in an already running app')
    p.add_argument('--menu', action='store_true', help='launch app in its ordinary menu with capture enabled')
    a = p.parse_args(argv)
    os.chdir(REPO)
    if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
    if a.api:
        from src.engine.communication.tcp_network_handler import send_command
        print(send_command('startGame', {'id':'light_targets'})); return
    for k,v in dict(SKJUTBANA_OBJECT_DENSE='1', SKJUTBANA_AUDIO_COOLDOWN_S='0.15',
        SKJUTBANA_SOURCE_RECORD='1', SKJUTBANA_SOURCE_RECORD_WINDOW='1',
        SKJUTBANA_PHYSICAL_TRACE_DISABLE='1').items(): os.environ.setdefault(k,v)
    # No calibration-dependent GAME_DX is enabled. Rule is authority, model remains SHADOW.
    model=Path('/data/skjutbana/evaluation_runs/can_peak_model_20261121/v6aug/model.json')
    if model.is_file(): os.environ.setdefault('SKJUTBANA_OBJECT_DENSE_MODEL',str(model))
    import main as startup  # install the same runtime layers as the normal application
    from src.engine.app import App
    from src.engine.scene_factory import game_scene_from_menu_id
    app=App()
    if not a.menu: app._switch_to(game_scene_from_menu_id('light_targets'))
    try:
        app._automation_window.position = (2135, 95)   # same projector placement as the Can Shootout launcher
    except Exception:
        pass
    app.run()

if __name__=='__main__': main()
