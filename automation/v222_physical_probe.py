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
        # This is a trusted locally-trained model. V2.21.5 intentionally stores
        # feature_names/metadata_json as NumPy object arrays, and its official
        # loader therefore uses allow_pickle=True. Use that loader instead of
        # incorrectly treating the file as an untrusted generic NPZ.
        from src.engine.offline.physical_dense_v2215 import ListwiseModelV2215

        model = ListwiseModelV2215.load(model_path)
        print("Model metadata:")
        print(f"  features: {len(model.feature_names)}")
        print(f"  mean: shape={model.mean.shape} dtype={model.mean.dtype}")
        print(f"  scale: shape={model.scale.shape} dtype={model.scale.dtype}")
        print(f"  weights: shape={model.weights.shape} dtype={model.weights.dtype}")
        print(f"  metadata keys: {sorted(model.metadata.keys())}")
        print("[PASS] V2.21.5 frozen model loaded through official loader")
    except Exception as exc:
        print(f"[WARN] model metadata probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
