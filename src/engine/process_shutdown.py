"""Bounded POSIX shutdown for the opt-in interactive application process."""
from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time

# Only the supervisor owns the deadline. A child stuck in native code or in
# Python's ThreadPoolExecutor interpreter-exit join cannot cancel that deadline.
PIPE_ENV = "_SKJUTBANA_SHUTDOWN_PIPE"
GRACE_SECONDS = 15.0


class ShutdownRequest:
    def __init__(self, notify_fd: int):
        self.notify_fd = notify_fd
        self.requested = False
        self.app = None

    def request(self, signum, frame):
        # A signal sent directly to the child also starts the parent deadline,
        # even if its main loop subsequently blocks before reaching finally.
        self.requested = True
        fd = self.notify_fd
        if fd is not None:
            try:
                os.write(fd, b"!")  # readable until the supervisor exits
            except (BrokenPipeError, BlockingIOError):
                pass

    def begin_cleanup(self) -> None:
        if self.notify_fd is not None:
            fd, self.notify_fd = self.notify_fd, None
            os.close(fd)


active: ShutdownRequest | None = None


def install_child_handlers() -> ShutdownRequest:
    global active
    fd = int(os.environ.pop(PIPE_ENV))
    os.set_inheritable(fd, False)
    os.set_blocking(fd, False)
    active = ShutdownRequest(fd)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, active.request)
    return active


def supervise(command: list[str], grace_s: float = GRACE_SECONDS) -> int:
    """Run one engine process in its own session and supervise its entire exit."""
    first_signal = None
    escalate = False

    def request(signum, frame):
        nonlocal first_signal, escalate
        if first_signal is None:
            first_signal = signum
        elif signum != first_signal:
            # The capture launcher can forward the same terminal SIGINT. Its
            # later SIGTERM is an explicit request to escalate.
            escalate = True

    previous = {sig: signal.signal(sig, request) for sig in (signal.SIGINT, signal.SIGTERM)}
    reader, writer = os.pipe()
    child = None
    expired = False
    try:
        try:
            child = subprocess.Popen(command, start_new_session=True, pass_fds=(writer,),
                                     env=dict(os.environ, **{PIPE_ENV: str(writer)}))
        finally:
            os.close(writer)
        deadline = None
        forwarded = False
        # WNOWAIT retains ownership of the child's PID/PGID until group cleanup.
        while os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
            if first_signal is not None and not forwarded:
                os.kill(child.pid, signal.SIGTERM)
                forwarded = True
            if deadline is None and (first_signal is not None or select.select([reader], [], [], 0)[0]):
                deadline = time.monotonic() + grace_s
            if escalate or (deadline is not None and time.monotonic() >= deadline):
                expired = True
                break
            time.sleep(0.05)
    finally:
        os.close(reader)
        try:
            if child is not None:
                # The group may still contain a capture subprocess after its
                # parent exits. Signal it before reaping the session leader.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    expired = True
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if expired:
        print("Avstängningen tvångsavslutades.", file=sys.stderr, flush=True)
        return 124
    return child.returncode if child.returncode >= 0 else 128 - child.returncode
