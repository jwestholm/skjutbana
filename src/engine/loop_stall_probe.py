"""Main-loop stall probe (measurement only, default off).

Enable with SKJUTBANA_STALL_PROBE=<path.jsonl>.  The game loop calls ``begin()`` after frame pacing,
``mark(name)`` between phases and ``end()`` after the flip.  A watchdog thread samples the stacks of ALL
threads every SAMPLE_S while an iteration is older than THRESHOLD_S, and reads per-thread CPU time from
/proc/self/task at stall start/end, so a stall can be attributed to: the phase it happened in, the main
thread's own code (stack), another thread holding the GIL (CPU moved to that thread), the garbage
collector (gc.callbacks durations) or blocking I/O/locks (main thread not on CPU).

Nothing here changes detector or game behaviour; with the variable unset every hook is a cheap no-op.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
import traceback

THRESHOLD_S = float(os.environ.get("SKJUTBANA_STALL_PROBE_MS", "150")) / 1000.0
SAMPLE_S = 0.02
_PATH = os.environ.get("SKJUTBANA_STALL_PROBE") or None
_CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


class _Probe:
    def __init__(self, path: str) -> None:
        self.path = path
        self.main_ident = threading.main_thread().ident
        self.t0 = None          # current iteration start (monotonic) or None
        self.iter_no = 0
        self.marks: list = []
        self.samples: list = []
        self.cpu_start = None
        self.gc_events: list = []
        self._gc_t = None
        self._lock = threading.Lock()
        self._fh = open(path, "a", buffering=1)
        gc.callbacks.append(self._gc_cb)
        threading.Thread(target=self._watch, name="stall-probe", daemon=True).start()

    # ---- gc
    def _gc_cb(self, phase, info):
        now = time.monotonic()
        if phase == "start":
            self._gc_t = now
        elif self._gc_t is not None:
            self.gc_events.append((self._gc_t, now, info.get("generation"), info.get("collected")))
            self._gc_t = None
            if len(self.gc_events) > 2000:
                del self.gc_events[:1000]

    # ---- per-thread cpu
    @staticmethod
    def _thread_cpu() -> dict:
        out = {}
        try:
            for tid in os.listdir("/proc/self/task"):
                try:
                    with open(f"/proc/self/task/{tid}/stat") as f:
                        st = f.read()
                    rest = st[st.rindex(")") + 2:].split()
                    name = st[st.index("(") + 1:st.rindex(")")]
                    out[int(tid)] = (name, (int(rest[11]) + int(rest[12])) / _CLK, rest[0])
                except (OSError, ValueError, IndexError):
                    pass
        except OSError:
            pass
        return out

    # ---- loop hooks
    def begin(self) -> None:
        with self._lock:
            self.marks = []
            self.samples = []
            self.cpu_start = None
            self.iter_no += 1
            self.t0 = time.monotonic()

    def mark(self, name: str) -> None:
        if self.t0 is not None:
            self.marks.append((name, time.monotonic()))

    def end(self) -> None:
        t1 = time.monotonic()
        with self._lock:
            t0 = self.t0
            self.t0 = None
        if t0 is None:
            return
        dur = t1 - t0
        if dur < THRESHOLD_S:
            return
        with self._lock:
            samples, cpu0 = self.samples, self.cpu_start
            self.samples = []
        cpu1 = self._thread_cpu()
        names = {t.native_id: t.name for t in threading.enumerate()}
        cpu = []
        if cpu0:
            for tid, (comm, c1, _s) in cpu1.items():
                c0 = cpu0.get(tid, (comm, c1, ""))[1]
                if c1 - c0 > 0.005:
                    cpu.append(dict(tid=tid, thread=names.get(tid, comm), cpu_s=round(c1 - c0, 3)))
        cpu.sort(key=lambda r: -r["cpu_s"])
        phases, prev = [], t0
        for name, t in self.marks:
            phases.append((name, round((t - prev) * 1000, 1)))
            prev = t
        phases.append(("tail", round((t1 - prev) * 1000, 1)))
        gcs = [dict(gen=g, ms=round((b - a) * 1000, 1), start_ms=round((a - t0) * 1000, 1), collected=c)
               for a, b, g, c in list(self.gc_events) if b >= t0 and a <= t1]
        rec = dict(ts=time.time(), iter=self.iter_no, dur_ms=round(dur * 1000, 1), phases_ms=phases,
                   gc=gcs, cpu_since_stall_detect=cpu,
                   cpu_window_s=round(dur - samples[0]["t"], 3) if samples else 0.0,
                   samples=[dict(t_ms=s["t_ms"], main=s["main"], other=s["other"]) for s in samples])
        try:
            self._fh.write(json.dumps(rec) + "\n")
        except (OSError, ValueError):
            pass

    # ---- watchdog
    def _watch(self) -> None:
        names = {}
        while True:
            time.sleep(SAMPLE_S)
            with self._lock:
                t0, it = self.t0, self.iter_no
            if t0 is None:
                continue
            age = time.monotonic() - t0
            if age < THRESHOLD_S:
                continue
            if self.cpu_start is None:
                cpu = self._thread_cpu()
                with self._lock:
                    if self.iter_no == it and self.cpu_start is None:
                        self.cpu_start = cpu
            frames = sys._current_frames()
            names = {t.ident: t.name for t in threading.enumerate()}
            main = frames.get(self.main_ident)
            other = {}
            for ident, fr in frames.items():
                if ident in (self.main_ident, threading.get_ident()):
                    continue
                top = traceback.extract_stack(fr, limit=6)
                if top:
                    other[names.get(ident, str(ident))] = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}"
                                                           for f in top[-3:]]
            stack = ([f"{os.path.relpath(f.filename) if 'skjutbana' in f.filename else os.path.basename(f.filename)}"
                      f":{f.lineno}:{f.name}" for f in traceback.extract_stack(main, limit=40)][-14:]
                     if main is not None else [])
            with self._lock:
                if self.iter_no == it and self.t0 is not None and len(self.samples) < 200:
                    self.samples.append(dict(t=age, t_ms=round(age * 1000, 1), main=stack, other=other))


_PROBE: _Probe | None = None


def _get() -> _Probe | None:
    global _PROBE
    if _PATH is None:
        return None
    if _PROBE is None:
        _PROBE = _Probe(_PATH)
    return _PROBE


def begin() -> None:
    if _PATH is not None:
        _get().begin()


def mark(name: str) -> None:
    if _PROBE is not None:
        _PROBE.mark(name)


def end() -> None:
    if _PROBE is not None:
        _PROBE.end()
