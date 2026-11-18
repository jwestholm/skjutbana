"""Run all selftest modules in private Linux test environments."""
from __future__ import annotations

import argparse
import asyncio
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent.parent
STAMP_FIXTURES = "/data/skjutbana/hole_library"


async def run_module(module: str, workers: asyncio.Semaphore, timeout: float,
                     command: list[str], env: dict[str, str]) -> bool:
    async with workers:
        start = time.monotonic()
        with tempfile.TemporaryFile() as log:
            try:
                process = await asyncio.create_subprocess_exec(
                    *command, module, cwd=ROOT, env=env, start_new_session=True,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                )
            except OSError as exc:
                status = f"FAIL ({exc})"
            else:
                try:
                    code = await asyncio.wait_for(process.wait(), timeout)
                    status = "PASS" if code == 0 else f"FAIL (exit {code})"
                except asyncio.TimeoutError:
                    status = "TIMEOUT"
                finally:
                    if process.returncode is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        await process.wait()
            print(f"{status:12} {time.monotonic() - start:8.2f}s  {module}", flush=True)
            if status != "PASS":
                size = log.seek(0, os.SEEK_END)
                log.seek(max(0, size - 16_384))
                print(log.read().decode(errors="replace"), flush=True)
            return status == "PASS"


async def run_suite(jobs: int, timeout: float) -> int:
    runner = Path(__file__).resolve()
    modules = [f"automation.{path.stem}" for path in sorted(runner.parent.glob("*selftest*.py"))
               if path != runner]
    if not modules:
        print("No selftest modules found", file=sys.stderr)
        return 1
    if not shutil.which("bwrap"):
        print("Install bubblewrap to run the suite with isolated test data", file=sys.stderr)
        return 1
    command = [
        "bwrap", "--unshare-pid", "--die-with-parent", "--unshare-net",
        "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
        "--tmpfs", "/data", "--dir", "/data/skjutbana/evaluation_runs",
        "--dir", "/data/skjutbana/physical_traces",
        "--ro-bind-try", STAMP_FIXTURES, STAMP_FIXTURES, "--ro-bind", str(ROOT), str(ROOT),
        sys.executable, "-m",
    ]
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", TMPDIR="/tmp",
               XDG_CACHE_HOME="/tmp/cache", MPLCONFIGDIR="/tmp/matplotlib",
               PYTHONDONTWRITEBYTECODE="1")
    paths = [str(ROOT), str(ROOT / "src")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    workers = asyncio.Semaphore(jobs)
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        results = await asyncio.gather(*(run_module(m, workers, timeout, command, env) for m in modules))
    except asyncio.CancelledError:
        print("Selftest run interrupted", file=sys.stderr)
        return 1
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
    passed = sum(results)
    print(f"Modules: {len(results)}, passed: {passed}, failed: {len(results) - passed}")
    return int(passed != len(results))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, 4), default=3,
                        help="number of concurrent modules (default 3)")
    parser.add_argument("--timeout", type=float, default=600.0, metavar="SECONDS",
                        help="time limit per module (default 600 seconds)")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    return asyncio.run(run_suite(args.jobs, args.timeout))


if __name__ == "__main__":
    raise SystemExit(main())
