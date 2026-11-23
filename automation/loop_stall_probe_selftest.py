"""loop_stall_probe: default off (no thread, no file); when enabled records only iterations >= threshold,
with phases, main-thread stack samples naming the stalling function, and gc events.
Run: PYTHONPATH=. python3 automation/loop_stall_probe_selftest.py
"""
import importlib, json, os, sys, tempfile, threading, time
from unittest.mock import patch


def slow_function(probe):
    # Wait for the real watchdog, then do measurable CPU work after its first
    # sample. A fixed 300 ms spin can end before a loaded host schedules it.
    start = time.monotonic()
    sampled_cpu = None
    while time.monotonic() - start < 15.0:
        sum(range(10000))
        with probe._lock:
            sampled = bool(probe.samples)
        if sampled and sampled_cpu is None:
            sampled_cpu = time.thread_time()
        if (time.monotonic() - start >= .3 and sampled_cpu is not None
                and time.thread_time() - sampled_cpu >= .08):
            return
        time.sleep(0)
    raise AssertionError(f'watchdog did not collect stacks and CPU within fifteen seconds: samples={len(probe.samples)}')


@patch.dict(os.environ, {'SKJUTBANA_STALL_PROBE_MS':'150'})
def main():
    os.environ.pop("SKJUTBANA_STALL_PROBE", None)
    import src.engine.loop_stall_probe as p
    p = importlib.reload(p)
    n = threading.active_count()
    p.begin(); p.mark("x"); p.end()
    assert p._PROBE is None and threading.active_count() == n, "must be inert when unset"
    with tempfile.TemporaryDirectory() as d:
        os.environ["SKJUTBANA_STALL_PROBE"] = d + "/s.jsonl"
        p = importlib.reload(p)
        p.begin(); p.mark("fast"); p.end()                    # short iteration: not recorded
        p.begin(); p.mark("a"); slow_function(p._PROBE); p.mark("slow_phase"); p.end()
        rows = [json.loads(l) for l in open(d + "/s.jsonl")]
        assert len(rows) == 1, rows
        r = rows[0]
        assert r["dur_ms"] >= 290 and r["phases_ms"][1][0] == "slow_phase" and r["phases_ms"][1][1] >= 280, r["phases_ms"]
        assert r["samples"] and any("slow_function" in fr for s in r["samples"] for fr in s["main"]), r["samples"][:1]
        assert any(c["cpu_s"] > 0.05 for c in r["cpu_since_stall_detect"]), r["cpu_since_stall_detect"]
        os.environ.pop("SKJUTBANA_STALL_PROBE")
    print("ok loop_stall_probe")


if __name__ == "__main__":
    main()
