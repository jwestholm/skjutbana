from __future__ import annotations

import sys
import time
import threading
import queue
from dataclasses import dataclass
from typing import Optional

import pygame

import config


# ----------------------------
# Domain events (from sensors / game)
# ----------------------------

@dataclass(frozen=True)
class DomainEvent:
    """Events coming from sensors or internal systems."""
    type: str
    t: float  # monotonic timestamp (seconds)
    data: dict


EVENT_SHOT_DETECTED = "SHOT_DETECTED"          # mic -> shot trigger
EVENT_HIT_COORDS_READY = "HIT_COORDS_READY"    # camera/CV -> x,y in calibrated space
EVENT_QUIT = "QUIT"                            # internal request to quit


# ----------------------------
# State machine
# ----------------------------

class GameState:
    def handle_os_event(self, event: pygame.event.Event) -> Optional[DomainEvent]:
        """Translate OS/pygame events to domain events (or None)."""
        return None

    def handle_domain_event(self, event: DomainEvent) -> Optional[DomainEvent]:
        """Handle sensor/internal events; may emit follow-up domain events."""
        return None

    def update_fixed(self, dt: float) -> None:
        """Fixed-timestep simulation update."""
        return None

    def render(self, screen: pygame.Surface) -> None:
        """Draw current frame."""
        raise NotImplementedError


class LoadingState(GameState):
    def __init__(self, screen: pygame.Surface) -> None:
        self._screen_size = screen.get_size()
        img = pygame.image.load(config.LOADING_SCREEN_PATH).convert()
        self._img = pygame.transform.scale(img, self._screen_size)

    def handle_os_event(self, event: pygame.event.Event) -> Optional[DomainEvent]:
        if event.type == pygame.QUIT:
            return DomainEvent(EVENT_QUIT, time.monotonic(), {})
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return DomainEvent(EVENT_QUIT, time.monotonic(), {})
        return None

    def render(self, screen: pygame.Surface) -> None:
        screen.blit(self._img, (0, 0))


# ----------------------------
# Sensor worker stubs (MIC / CAMERA)
# Replace internals later without touching main loop.
# ----------------------------

class MicShotListener(threading.Thread):
    """
    Listens to microphone and pushes EVENT_SHOT_DETECTED to event_queue.
    Stub for now: does nothing.
    """
    def __init__(self, event_queue: "queue.Queue[DomainEvent]", stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self._q = event_queue
        self._stop = stop_event

    def run(self) -> None:
        # Future:
        # - open audio stream (sounddevice)
        # - detect impulse/peak
        # - self._q.put(DomainEvent(EVENT_SHOT_DETECTED, time.monotonic(), {...}))
        while not self._stop.is_set():
            time.sleep(0.01)


class CameraProcessor(threading.Thread):
    """
    Captures frames continuously (ring buffer) and when a shot is detected,
    analyzes frames around that timestamp and emits EVENT_HIT_COORDS_READY.
    Stub for now: does nothing.
    """
    def __init__(self, event_queue: "queue.Queue[DomainEvent]", stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self._q = event_queue
        self._stop = stop_event

    def notify_shot(self, shot_time: float) -> None:
        # Future:
        # - pick frames just before/after shot_time from ring buffer
        # - run CV to compute hit x,y in calibrated space
        # - self._q.put(DomainEvent(EVENT_HIT_COORDS_READY, time.monotonic(), {"x":..., "y":...}))
        return None

    def run(self) -> None:
        # Future:
        # - cv2.VideoCapture loop
        # - store (timestamp, frame) in ring buffer
        while not self._stop.is_set():
            time.sleep(0.01)


# ----------------------------
# Game runtime
# ----------------------------

class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Skjutbana")
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.FULLSCREEN)

        # Domain event queue (thread-safe). Sensors push here; main thread consumes.
        self.event_queue: "queue.Queue[DomainEvent]" = queue.Queue()

        # Stop flag for background workers
        self.stop_event = threading.Event()

        # Sensors (stubbed, but wired)
        self.mic = MicShotListener(self.event_queue, self.stop_event)
        self.cam = CameraProcessor(self.event_queue, self.stop_event)

        # Start state
        self.state: GameState = LoadingState(self.screen)

        # For fixed timestep loop
        self.UPDATE_HZ = 60.0
        self.FIXED_DT = 1.0 / self.UPDATE_HZ
        self.accumulator = 0.0
        self.prev_time = time.monotonic()

        self.running = True

    def shutdown(self) -> None:
        self.stop_event.set()
        # Join quickly; daemon threads won’t block exit, but this is cleaner.
        self.mic.join(timeout=0.2)
        self.cam.join(timeout=0.2)
        pygame.quit()

    def start_workers(self) -> None:
        self.mic.start()
        self.cam.start()

    def pump_os_events(self) -> None:
        for e in pygame.event.get():
            dom = self.state.handle_os_event(e)
            if dom:
                self.handle_domain_event(dom)

    def pump_domain_events(self) -> None:
        # Drain queue without blocking
        while True:
            try:
                dom = self.event_queue.get_nowait()
            except queue.Empty:
                break
            self.handle_domain_event(dom)

    def handle_domain_event(self, event: DomainEvent) -> None:
        if event.type == EVENT_QUIT:
            self.running = False
            return

        # Route shot events to camera processor later (sync point)
        if event.type == EVENT_SHOT_DETECTED:
            # Notify camera worker to analyze around this timestamp.
            # In a real implementation, you likely also create a "Shot" object with id/time.
            self.cam.notify_shot(event.t)

        # Let current state handle it
        follow_up = self.state.handle_domain_event(event)
        if follow_up:
            # Rare but supported: state emits internal domain events
            self.handle_domain_event(follow_up)

    def fixed_update_loop(self) -> None:
        now = time.monotonic()
        frame_time = now - self.prev_time
        self.prev_time = now

        # Avoid "spiral of death" if the app stalls (debugger, window dragged, etc.)
        if frame_time > 0.25:
            frame_time = 0.25

        self.accumulator += frame_time

        while self.accumulator >= self.FIXED_DT:
            self.state.update_fixed(self.FIXED_DT)
            self.accumulator -= self.FIXED_DT

    def render(self) -> None:
        self.state.render(self.screen)
        pygame.display.flip()

    def run(self) -> int:
        self.start_workers()

        # Main loop: OS events → sensor events → fixed updates → render
        while self.running:
            self.pump_os_events()
            self.pump_domain_events()
            self.fixed_update_loop()
            self.render()

            # Optional: tiny sleep to reduce CPU burn without affecting fixed updates much.
            # You can remove this later if you want max render throughput.
            time.sleep(0.001)

        self.shutdown()
        return 0


def main() -> int:
    try:
        game = Game()
        return game.run()
    except pygame.error as e:
        print(f"Pygame error: {e}")
        return 1
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())