"""IPC bridge: Skjutbanan audio peaks -> MANUELL shot trigger.

Never creates physical coordinates. Never queues shots during MARKERA.
"""
from __future__ import annotations

import queue
import sys
import threading
import time

from trench_3d_test import Trench


class SkjutbanaTrench(Trench):
    def __init__(self):
        super().__init__()

        self.external_shots = queue.Queue(maxsize=1)
        self.external_exit = threading.Event()

        if not self.set_manual_mode(True):
            raise RuntimeError("Kunde inte aktivera MANUELL")

        threading.Thread(
            target=self._read_commands,
            name="skjutbana-shot-input",
            daemon=True,
        ).start()

        print("[BRIDGE] MANUELL + SHOT_ONLY aktiv", flush=True)

    def _read_commands(self):
        for line in sys.stdin:
            message = line.strip()

            if message == "QUIT":
                self.external_exit.set()
                return

            if not message.startswith("SHOT "):
                continue

            try:
                triggered_at = float(message.split()[1])
            except (ValueError, IndexError):
                continue

            # Reject old IPC messages and all events received
            # while waiting for manual adjudication.
            if time.monotonic() - triggered_at > 0.5:
                continue
            if self.demo_pending or not self.demo_shot_ready:
                continue
            if not self.demo_mode or not self.running:
                continue

            try:
                self.external_shots.put_nowait(triggered_at)
            except queue.Full:
                pass

        self.external_exit.set()

    def handle(self):
        super().handle()

        if self.external_exit.is_set():
            self.running = False
            self._manual_cancel()
            return

        if not self.running:
            return

        # Maximum one trigger per game frame.
        # Discard stale triggers; never replay after SPACE/M.
        try:
            triggered_at = self.external_shots.get_nowait()
        except queue.Empty:
            return

        if time.monotonic() - triggered_at > 0.5:
            return

        if self.demo_mode and not self.demo_pending:
            accepted = self.begin_demo_shot()
            if accepted:
                print("[BRIDGE] SHOT_ONLY -> MARKERA", flush=True)


if __name__ == "__main__":
    SkjutbanaTrench().run()
