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


def test_stale_generation_dropped() -> None:
    """A job that finishes after a scanner reset (new owner) must not deliver its result."""
    import numpy as np
    captured = []

    class FakeExecutor:
        def submit(self, fn):
            captured.append(fn)
            class F:
                def done(self):
                    return False
                def cancel(self):
                    return False
            return F()

    orig = h15._own_executor
    h15._own_executor = lambda: FakeExecutor()
    try:
        h15._S.reset(owner=111)
        h15._S.roi = (0, 0, 64, 64)
        f = np.full((64, 64), 120, np.uint8)
        h15._submit((5, 100.0), 100.0, 100.6, [f], [f], dict(post_first_ts=100.2, post_last_ts=100.6))
        h15._S.reset(owner=222)          # scanner restarted before the job ran
        captured[0]()
        assert (5, 100.0) not in h15._S.results, "stale job delivered into a new generation"
        h15._S.reset(owner=333)
        h15._S.roi = (0, 0, 64, 64)
        h15._submit((6, 200.0), 200.0, 200.6, [f], [f], dict(post_first_ts=200.2, post_last_ts=200.6))
        captured[1]()
        assert (6, 200.0) in h15._S.results and list(h15._S.results) == [(6, 200.0)], "result keyed to its own shot"
    finally:
        h15._own_executor = orig
        h15._S.reset(owner=None)


def test_log_default_off(tmp=None) -> None:
    import tempfile
    os.environ.pop("SKJUTBANA_H15_LOG", None)
    h15._log(dict(kind="x"))           # no-op, no file
    with tempfile.TemporaryDirectory() as d:
        os.environ["SKJUTBANA_H15_LOG"] = d + "/h15.jsonl"
        try:
            h15._log(dict(kind="job", shot_id=1, peaks=[(1.0, 2.0, 3.0)]))
            assert open(d + "/h15.jsonl").read().count("\n") == 1
        finally:
            os.environ.pop("SKJUTBANA_H15_LOG", None)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    if h15._OWN_EXECUTOR is not None:
        h15._OWN_EXECUTOR.shutdown(wait=True)
