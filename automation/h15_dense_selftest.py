"""Selftest for src/engine/shot_dense_h15.py helpers (default-off H15).
  * _median: exact per-pixel median of <= SKJUTBANA_H15_MAX_FRAMES evenly spaced frames, odd count
    (even counts drop the oldest frame) - must equal np.median of that same frame subset.
  * _own_executor: None unless SKJUTBANA_H15_OWN_THREAD=1 (default behaviour unchanged).
Run: PYTHONPATH=. python3 automation/h15_dense_selftest.py
"""
from __future__ import annotations

import os

import numpy as np

from src.engine import shot_dense_h15 as h15


def test_median_exact() -> None:
    rng = np.random.default_rng(0)
    max_n = 9
    for n in range(1, 16):
        frames = [rng.integers(0, 256, (37, 53), dtype=np.uint8) for _ in range(n)]
        sub = frames
        if n > max_n:
            idx = np.linspace(0, n - 1, max_n).round().astype(int)
            sub = [frames[i] for i in idx]
        if len(sub) % 2 == 0 and len(sub) > 1:
            sub = sub[1:]
        ref = np.median(np.stack(sub).astype(np.float32), axis=0)
        got = h15._median(frames)
        assert got.dtype == np.float32 and np.array_equal(got, ref), f"median mismatch n={n}"
    assert h15._median([]) is None


def test_own_executor_default_off() -> None:
    os.environ.pop("SKJUTBANA_H15_OWN_THREAD", None)
    assert h15._own_executor() is None
    os.environ["SKJUTBANA_H15_OWN_THREAD"] = "1"
    try:
        ex = h15._own_executor()
        assert ex is not None and ex is h15._own_executor()
        assert ex.submit(lambda: 7).result(timeout=5) == 7
    finally:
        os.environ.pop("SKJUTBANA_H15_OWN_THREAD", None)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    if h15._OWN_EXECUTOR is not None:
        h15._OWN_EXECUTOR.shutdown(wait=True)
