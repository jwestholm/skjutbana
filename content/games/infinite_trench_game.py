"""Skjutbanan adapter for the standalone ModernGL trench game.

SHOT_ONLY: microphone trigger, no physical X/Y coordinate.
The game owns MANUELL HIT/MISS adjudication.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pygame

from src.engine.audio.audio_peak_detector import audio_peak_detector
from src.engine.scene import SceneSwitch


class InfiniteTrenchGame:
    shot_input_mode = "SHOT_ONLY"
    wants_hit_scanning = False
    wants_mouse_simulated_hits = False

    def __init__(self, game_root, viewport):
        self.game_dir = (Path(game_root) / "3d").resolve()
        self.viewport = viewport.copy()
        self.process = None
        self.font = None
        self.log_path = None

    def on_enter(self):
        script = self.game_dir / "skjutbana_bridge.py"
        if not script.is_file():
            raise FileNotFoundError(script)

        log_dir = Path("/data/skjutbana")
        if not log_dir.is_dir() or not os.access(log_dir, os.W_OK):
            log_dir = Path(tempfile.gettempdir())

        self.log_path = log_dir / "skyttegrav_3d_bridge.log"
        logfile = open(self.log_path, "w", encoding="utf-8")

        try:
            self.process = subprocess.Popen(
                [sys.executable, "-u", str(script)],
                cwd=str(self.game_dir),
                stdin=subprocess.PIPE,
                stdout=logfile,
                stderr=subprocess.STDOUT,
                bufsize=0,
                start_new_session=True,
            )
        finally:
            logfile.close()

        os.set_blocking(self.process.stdin.fileno(), False)
        self.font = pygame.font.Font(None, 27)

        audio_peak_detector.subscribe(self._on_audio_peak)

        print("[SKYTTEGRAV] shot_input_mode=SHOT_ONLY")
        print("[SKYTTEGRAV] Kamera/HitScanner behövs inte för detta spel.")
        print(f"[SKYTTEGRAV] Logg: {self.log_path}")

    def _send(self, command):
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            os.write(self.process.stdin.fileno(), command.encode("ascii"))
            return True
        except (BlockingIOError, BrokenPipeError, OSError):
            return False

    def _on_audio_peak(self, event):
        # Called by AudioPeakDetector in Skjutbanan's main thread.
        # No fabricated game coordinates or image verification.
        self._send(f"SHOT {time.monotonic():.9f}\n")

    def on_exit(self):
        audio_peak_detector.unsubscribe(self._on_audio_peak)

        if self.process is None:
            return

        if self.process.poll() is None:
            self._send("QUIT\n")
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

        if self.process.stdin:
            self.process.stdin.close()
        self.process = None

    def handle_event(self, event):
        return None

    def update(self, dt):
        if self.process is not None and self.process.poll() is not None:
            print(f"[SKYTTEGRAV] Spelprocess avslutades: {self.process.returncode}")
            from src.engine.scenes.menu import MenuScene
            return SceneSwitch(MenuScene())
        return None

    def render(self, screen):
        if self.font:
            lines = (
                "Den oandliga skyttegraven - 3D",
                "Skottlage: SHOT_ONLY",
                "Mikrofontrigger utan traffkoordinat",
                "MANUELL: SPACE = TRAFF, M = BOM",
                "ESC = aterga till menyn",
            )
            y = self.viewport.top + 30
            for line in lines:
                surface = self.font.render(line, True, (230, 230, 230))
                screen.blit(surface, (self.viewport.left + 20, y))
                y += 32

    def get_hit_regions(self):
        return ()


def create_game(game_root, viewport):
    return InfiniteTrenchGame(game_root, viewport)
