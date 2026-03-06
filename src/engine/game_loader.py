from __future__ import annotations

import importlib.util
from pathlib import Path


def load_game_module(script_path: str):
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"Game script not found: {script_path}")

    module_name = f"game_module_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for game script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module