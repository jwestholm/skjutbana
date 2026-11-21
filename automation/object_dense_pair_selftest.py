"""Selftest: under SKJUTBANA_OBJECT_DENSE a follower audio peak 65-80 ms after a shot (muzzle + impact /
CO2 double report, seen live 2026-11-20) must not cut the first shot's OBJECT query; with the flag off the
old behaviour (finish at next peak) is preserved, and a genuinely later shot still cuts it under the flag.

    python3 -m automation.object_dense_pair_selftest
"""
from __future__ import annotations

import os
import threading
from types import SimpleNamespace

from src.engine.input.object_hit_v2223 import ObjectHitRegistryV2223


class FakeQuery:
    def __init__(self, peak_ts):
        self.peak_ts = peak_ts
        self.last_frame_ts = peak_ts - 1.0
        self.final = False
        self.published = False
        self.object_hit_requested = False
        self.seen = []
        self.result = None

    def observe(self, frame, delivered_ts=None):
        self.seen.append(frame.timestamp)
        self.last_frame_ts = frame.timestamp

    def _finish(self, why):
        self.final = True
        self.result = dict(reason=why, timing={})

    def finish_deadline(self, now=None):
        self._finish('deadline_or_next_peak')

    def finish(self, status, reason, now=None):
        self._finish(reason)


def run(dense: bool, follower_gap: float) -> FakeQuery:
    if dense:
        os.environ["SKJUTBANA_OBJECT_DENSE"] = "1"
    else:
        os.environ.pop("SKJUTBANA_OBJECT_DENSE", None)
    t0 = 1000.0
    q1, q2 = FakeQuery(t0), FakeQuery(t0 + follower_gap)
    reg = SimpleNamespace(_lock=threading.Lock(), _snapshots={
        1: SimpleNamespace(peak_ts=q1.peak_ts, object_query=q1, object_gameplay={}, object_current={}),
        2: SimpleNamespace(peak_ts=q2.peak_ts, object_query=q2, object_gameplay={}, object_current={})})
    frames = [SimpleNamespace(timestamp=t0 + 0.033 * i) for i in range(1, 8)]   # 30 fps, up to +0.23 s
    cam = SimpleNamespace(get_ring_snapshot=lambda: frames)
    # one tick at +0.25 s: the follower (if any) has already happened
    for _tick in range(3):   # at most two frames per query per tick
        ObjectHitRegistryV2223.advance_object_queries(reg, None, cam, now=t0 + 0.25)
    return q1


def main() -> None:
    q = run(dense=False, follower_gap=0.07)
    assert q.final and len(q.seen) <= 2, ("flag off must keep old cut-at-next-peak", q.seen, q.final)
    q = run(dense=True, follower_gap=0.07)
    assert not q.final and max(q.seen) > 1000.07, ("follower must not cut", q.seen, q.final)
    q = run(dense=True, follower_gap=0.6)
    assert not q.final, "later shot 0.6 s away does not affect a tick at +0.25 s"
    q1 = run(dense=True, follower_gap=0.3)
    # 0.3 s > pair window 0.25 s: frames after the second peak are excluded, so it is a separate shot
    assert all(ts < 1000.3 for ts in q1.seen)
    # follower that finishes first (fast UNKNOWN) must not be published before its owner
    os.environ["SKJUTBANA_OBJECT_DENSE"] = "1"
    import src.engine.events as ev
    emitted = []
    orig = ev.event_bus.emit
    ev.event_bus.emit = lambda name, data, **kw: emitted.append(data['sid'])
    try:
        t0 = 2000.0
        own, fol = FakeQuery(t0), FakeQuery(t0 + 0.07)
        fol._finish('fast_unknown'); fol.result['sid'] = 'follower'
        reg = SimpleNamespace(_lock=threading.Lock(), _snapshots={
            1: SimpleNamespace(peak_ts=own.peak_ts, object_query=own, object_gameplay={}, object_current={}),
            2: SimpleNamespace(peak_ts=fol.peak_ts, object_query=fol, object_gameplay={}, object_current={})})
        cam = SimpleNamespace(get_ring_snapshot=lambda: [SimpleNamespace(timestamp=t0 + 0.033 * i) for i in range(1, 6)])
        ObjectHitRegistryV2223.advance_object_queries(reg, None, cam, now=t0 + 0.2)
        assert emitted == [] and not fol.published, emitted
        own._finish('owner_done'); own.result['sid'] = 'owner'
        ObjectHitRegistryV2223.advance_object_queries(reg, None, cam, now=t0 + 0.4)
        assert emitted == ['owner', 'follower'], emitted
    finally:
        ev.event_bus.emit = orig
    os.environ.pop("SKJUTBANA_OBJECT_DENSE", None)
    print("OK")


if __name__ == "__main__":
    main()
