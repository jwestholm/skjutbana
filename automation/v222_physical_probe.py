"""Probe the installed V2.21.5 physical model/module without changing anything.

Run from repository root:
    python3 -m automation.v222_physical_probe

This is diagnostic only. V2.22 does not execute the heavy full-frame V2.21.5
pipeline synchronously in the live emission path.
"""
from __future__ import annotations

from pathlib import Path
import importlib
import inspect


def main() -> None:
    print("V2.22 PHYSICAL EXPERT PROBE")
    print("===========================")
    module_names = [
        "src.engine.offline.physical_dense_v2215",
        "automation.physical_dense_v2215_benchmark",
        "automation.physical_dense_v2215_train",
    ]
    for name in module_names:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            print(f"[WARN] import {name}: {type(exc).__name__}: {exc}")
            continue
        path = getattr(module, "__file__", None)
        print(f"[PASS] import {name} -> {path}")
        interesting = []
        for symbol, value in inspect.getmembers(module):
            lower = symbol.lower()
            if symbol.startswith("_"):
                continue
            if not any(token in lower for token in ("dense", "candidate", "rank", "model", "feature", "load", "infer", "proposal")):
                continue
            if inspect.isfunction(value) or inspect.isclass(value):
                try:
                    sig = str(inspect.signature(value))
                except Exception:
                    sig = "(?)"
                interesting.append(f"  {symbol}{sig}")
        for line in interesting[:60]:
            print(line)

    model_path = Path("content/ai/reports/v2215/physical_dense_ranker_v2215.npz")
    if not model_path.exists():
        print(f"[WARN] frozen model missing: {model_path}")
        return
    print(f"[PASS] frozen model exists: {model_path} ({model_path.stat().st_size} bytes)")
    try:
        import numpy as np
        with np.load(model_path, allow_pickle=False) as data:
            print("NPZ keys:")
            for key in data.files:
                value = data[key]
                print(f"  {key}: shape={getattr(value, 'shape', None)} dtype={getattr(value, 'dtype', None)}")
    except Exception as exc:
        print(f"[WARN] model metadata probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
