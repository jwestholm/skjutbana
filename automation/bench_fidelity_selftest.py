"""Live/offline parity guard: the replay runtime must install exactly the live patches.

Runs two fresh subprocesses: one executes main.py's real import+installer block
(no window, no hardware), the other installs VirtualRuntime. The resolved
module:qualname of every HitScanner / AudioPeakDetector / CameraManager method
must be identical. A difference means offline results do not test live code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROBE = r'''
import json, os, sys
sys.path.insert(0, os.getcwd())
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["SKJUTBANA_PHYSICAL_TRACE_DISABLE"] = "1"
mode = sys.argv[1]
if mode == "live":
    src = open("main.py").read()
    start = src.index("from src.engine.app import App")
    end = src.index("install_v253_runtime(App)") + len("install_v253_runtime(App)")
    exec(compile(src[start:end], "main.py", "exec"), {"__name__": "main_probe"})
else:
    from src.engine.offline.virtual_runtime import VirtualRuntime
    VirtualRuntime().install()
from src.engine.camera.hit_scanner import HitScanner
from src.engine.audio.audio_peak_detector import AudioPeakDetector
from src.engine.camera.camera_manager import CameraManager
out = {}
for cls in (HitScanner, AudioPeakDetector, CameraManager):
    for name in sorted(dir(cls)):
        if name.startswith("__"):
            continue
        fn = getattr(cls, name)
        if callable(fn):
            out[f"{cls.__name__}.{name}"] = f"{getattr(fn, '__module__', '?')}:{getattr(fn, '__qualname__', '?')}"
print("PARITY" + json.dumps(out))
'''


def _probe(mode: str) -> dict:
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    proc = subprocess.run([sys.executable, "-c", PROBE, mode], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=180)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("PARITY")), None)
    if line is None:
        raise AssertionError(f"{mode} probe failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return json.loads(line[len("PARITY"):])


def main():
    live, replay = _probe("live"), _probe("replay")
    diff = {k: (live.get(k), replay.get(k)) for k in set(live) | set(replay) if live.get(k) != replay.get(k)}
    if diff:
        raise AssertionError("live/offline patch mismatch: " + json.dumps(diff, indent=2))
    if "resolve_v2225" not in live["HitScanner._resolve_audio_events"]:
        raise AssertionError("unexpected live resolve chain head")
    print(f"PASS replay runtime installs the same {len(live)} live methods as main.py startup")


if __name__ == "__main__":
    main()
