"""V2.22.3 shot-critical runtime.

A real audio shot outranks ordinary game-engine work.  The microphone reader
thread remains producer-only; the main thread acknowledges and dispatches queued
AudioPeakEvents before camera housekeeping, automation, ordinary scene update
and decorative rendering.

This module is installed explicitly by ``main.py``.  It is fail-open: if the
V2.22.3 installer is not called, the existing App/HitScanner path is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from src.engine.input.object_hit_v2223 import object_hit_registry_v2223, viewport_center_prior

SCHEMA_VERSION = "2.22.3"
_INSTALLED = False


def _rect_xywh(rect: Any) -> tuple[float, float, float, float] | None:
    if rect is None:
        return None
    try:
        w = getattr(rect, "w", getattr(rect, "width"))
        h = getattr(rect, "h", getattr(rect, "height"))
        return float(rect.x), float(rect.y), float(w), float(h)
    except Exception:
        return None


def select_recent_pre_frame_v2223(
    ring: Sequence[Any],
    peak_ts: float,
    *,
    target_offset_s: float = 0.35,
    latest_safe_offset_s: float = 0.08,
):
    """Select a timestamped PRE frame while forbidding post-shot leakage."""
    values = [cf for cf in ring if float(getattr(cf, "timestamp", 0.0)) <= float(peak_ts) - float(latest_safe_offset_s)]
    if not values:
        return None
    target = float(peak_ts) - float(target_offset_s)
    return min(values, key=lambda cf: abs(float(getattr(cf, "timestamp", 0.0)) - target))


@dataclass
class ShotTimingV2223:
    shot_id: int
    peak_ts: float
    main_ack_ts: float
    dispatch_delay_ms: float
    audio_dispatch_ms: float = 0.0
    camera_update_ms: float = 0.0
    scanner_update_ms: float = 0.0
    detector_done_ts: float = 0.0
    hit_state: str = "pending"
    visible_ts: float = 0.0
    region_count: int = 0
    object_shadow_logged: bool = False


class ShotCriticalControllerV2223:
    def __init__(self) -> None:
        self.last_seen_peak_ts = 0.0
        self.last_scanner_shot_id = 0
        self.timings: dict[int, ShotTimingV2223] = {}
        self._just_finished: list[int] = []
        self.mouse_debug_enabled = False
        self.latency_cursor_enabled = False
        self._cursor_saved_pos: tuple[int, int] | None = None
        self._cursor_saved_visible: bool | None = None
        self._cursor_wait_active = False
        self._ack_ts_by_peak: dict[float, float] = {}
        self._prepared_object_shots: set[int] = set()

    def pending_audio(self, detector: Any) -> bool:
        return float(getattr(detector, "last_peak_ts", 0.0) or 0.0) > self.last_seen_peak_ts + 1e-9

    @staticmethod
    def scanner_has_open_shot(scanner: Any) -> bool:
        try:
            return any(str(getattr(ev, "state", "")) == "pending" for ev in scanner.audio_events)
        except Exception:
            return False

    def _set_wait_cursor(self, app: Any) -> None:
        if not self.latency_cursor_enabled:
            return
        try:
            import pygame
            self._cursor_saved_pos = tuple(int(v) for v in pygame.mouse.get_pos())
            self._cursor_saved_visible = bool(pygame.mouse.get_visible())
            # Keep the diagnostic cursor outside the calibrated playfield ROI.
            pygame.mouse.set_pos((4, 4))
            try:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_WAIT)
            except Exception:
                pass
            pygame.mouse.set_visible(True)
            self._cursor_wait_active = True
        except Exception:
            self._cursor_wait_active = False

    def _restore_wait_cursor(self) -> None:
        if not self._cursor_wait_active:
            return
        try:
            import pygame
            try:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
            except Exception:
                pass
            if self._cursor_saved_pos is not None:
                pygame.mouse.set_pos(self._cursor_saved_pos)
            if self._cursor_saved_visible is not None:
                pygame.mouse.set_visible(self._cursor_saved_visible)
        except Exception:
            pass
        self._cursor_wait_active = False

    def begin_pending_audio(self, app: Any, detector: Any) -> float | None:
        peak_ts = float(getattr(detector, "last_peak_ts", 0.0) or 0.0)
        if peak_ts <= self.last_seen_peak_ts + 1e-9:
            return None
        self.last_seen_peak_ts = peak_ts
        now = time.time()
        self._ack_ts_by_peak[peak_ts] = now
        delay_ms = max(0.0, (now - peak_ts) * 1000.0)
        self._set_wait_cursor(app)
        print(f"[V2.22.3 AUDIO-PRIORITY] peak={peak_ts:.6f} main_ack={delay_ms:.1f}ms")
        return peak_ts

    def prepare_object_snapshot(self, scanner: Any, scene: Any, peak_ts: float | None) -> None:
        """Freeze game regions before audio dispatch creates the AI shot context.

        HitScanner allocates monotonically increasing ``_next_shot_id`` values.
        Preparing that id before ``audio_peak_detector.update()`` lets the
        existing V2.22 AIRuntime game-context provider see the frozen snapshot
        during its shot-context creation callback.  This remains shadow-only.
        """
        if peak_ts is None:
            return
        try:
            if not bool(getattr(scanner, "enabled", False)):
                return
            sid = int(getattr(scanner, "_next_shot_id", 0) or 0)
            if sid <= 0 or sid in self._prepared_object_shots:
                return
            object_hit_registry_v2223.snapshot(sid, float(peak_ts), scene=scene)
            self._prepared_object_shots.add(sid)
        except Exception:
            pass

    def discover_scanner_events(self, scanner: Any, scene: Any) -> None:
        try:
            events = list(scanner.audio_events)
        except Exception:
            return
        for event in events:
            sid = int(getattr(event, "shot_id", 0) or 0)
            if sid <= 0 or sid in self.timings:
                continue
            peak_ts = float(getattr(event, "peak_ts", 0.0) or 0.0)
            now = time.time()
            # If main.py prepared this shot before audio dispatch, preserve that
            # exact frozen geometry. Otherwise fall back to snapshotting now.
            snap = object_hit_registry_v2223.snapshot_for_shot(sid)
            if snap is None:
                snap = object_hit_registry_v2223.snapshot(sid, peak_ts, scene=scene)
            ack_ts = float(self._ack_ts_by_peak.pop(peak_ts, now))
            self.timings[sid] = ShotTimingV2223(
                shot_id=sid,
                peak_ts=peak_ts,
                main_ack_ts=ack_ts,
                dispatch_delay_ms=max(0.0, (ack_ts - peak_ts) * 1000.0),
                region_count=len(snap.regions),
            )
            self.last_scanner_shot_id = max(self.last_scanner_shot_id, sid)
            print(
                f"[V2.22.3 SHOT] shot={sid} dispatch={self.timings[sid].dispatch_delay_ms:.1f}ms "
                f"objects={len(snap.regions)}"
            )

    def note_stage(self, *, audio_dispatch_ms: float = 0.0, camera_update_ms: float = 0.0, scanner_update_ms: float = 0.0) -> None:
        pending = [t for t in self.timings.values() if t.hit_state == "pending"]
        for timing in pending:
            timing.audio_dispatch_ms += float(audio_dispatch_ms)
            timing.camera_update_ms += float(camera_update_ms)
            timing.scanner_update_ms += float(scanner_update_ms)

    def enrich_spatial_context(self, scanner: Any) -> None:
        """Add non-authoritative viewport priors to current candidates."""
        try:
            from src.engine.input.hit_input import hit_input
            from src.engine.settings import load_viewport_rect

            vp = _rect_xywh(load_viewport_rect())
            if vp is None:
                return
            for candidate in list(getattr(scanner, "last_candidates", []) or []):
                try:
                    sx, sy = hit_input._canonical_camera_to_screen(
                        float(candidate["camera_x"]), float(candidate["camera_y"])
                    )
                    centre, edge = viewport_center_prior(sx, sy, vp)
                    candidate["v2223_screen_x"] = float(sx)
                    candidate["v2223_screen_y"] = float(sy)
                    candidate["v2223_center_prior"] = float(centre)
                    candidate["v2223_edge_distance_norm"] = float(edge)
                except Exception:
                    continue
        except Exception:
            pass

    def evaluate_object_shadow(self, scanner: Any) -> None:
        try:
            from src.engine.input.hit_input import hit_input
            from src.engine.settings import load_viewport_rect

            vp = _rect_xywh(load_viewport_rect())
            candidates = list(getattr(scanner, "last_candidates", []) or [])
            for sid, timing in list(self.timings.items()):
                if timing.object_shadow_logged or timing.region_count <= 0:
                    continue
                results = object_hit_registry_v2223.evaluate_candidates(
                    sid,
                    candidates,
                    camera_to_screen=hit_input._canonical_camera_to_screen,
                    viewport_rect_xywh=vp,
                )
                if not results:
                    continue
                hits = [r for r in results if r.hit]
                hits.sort(key=lambda r: (r.confidence, r.candidate_score), reverse=True)
                top = hits[0] if hits else None
                if top is None:
                    print(f"[V2.22.3 OBJECT SHADOW] shot={sid} regions={len(results)} hit=none")
                else:
                    print(
                        f"[V2.22.3 OBJECT SHADOW] shot={sid} regions={len(results)} "
                        f"object={top.object_id} conf={top.confidence:.3f} rank={top.candidate_rank} "
                        f"local=({top.local_x:.3f},{top.local_y:.3f}) novelty={top.shot_novelty:.2f}"
                    )
                timing.object_shadow_logged = True
        except Exception:
            pass

    def update_finished(self, scanner: Any) -> None:
        try:
            events = list(scanner.audio_events)
        except Exception:
            return
        by_id = {int(getattr(ev, "shot_id", 0) or 0): ev for ev in events}
        now = time.time()
        for sid, timing in list(self.timings.items()):
            if timing.hit_state != "pending":
                continue
            ev = by_id.get(sid)
            if ev is None:
                continue
            state = str(getattr(ev, "state", "pending") or "pending")
            if state == "pending":
                continue
            timing.hit_state = state
            timing.detector_done_ts = now
            self._just_finished.append(sid)
            e2e = max(0.0, (now - timing.peak_ts) * 1000.0)
            print(
                f"[V2.22.3 LATENCY] shot={sid} state={state} "
                f"dispatch={timing.dispatch_delay_ms:.1f}ms camera={timing.camera_update_ms:.1f}ms "
                f"scanner={timing.scanner_update_ms:.1f}ms hit={e2e:.1f}ms objects={timing.region_count}"
            )
            self._restore_wait_cursor()

    def mark_visible_after_flip(self) -> None:
        if not self._just_finished:
            return
        now = time.time()
        finished = list(self._just_finished)
        self._just_finished.clear()
        for sid in finished:
            timing = self.timings.get(sid)
            if timing is None:
                continue
            timing.visible_ts = now
            visible = max(0.0, (now - timing.peak_ts) * 1000.0)
            print(f"[V2.22.3 VISIBLE] shot={sid} marker_frame={visible:.1f}ms")

    def should_defer_scene_work(self, scene: Any, scanner: Any) -> bool:
        if not self.scanner_has_open_shot(scanner):
            return False
        # Synthetic/F2 training needs scene update + render to reveal the fake
        # hole. Never freeze that path. Real/manual shots may defer decorative
        # scene work until the detector has resolved the shot.
        if bool(getattr(scene, "auto_training_enabled", False)):
            return False
        if bool(getattr(scene, "single_synth_round_active", False)):
            return False
        if bool(getattr(scene, "synthetic_trigger_pending", False)):
            return False
        if bool(getattr(scene, "synthetic_reveal_pending", False)):
            return False
        return not bool(getattr(scene, "shot_critical_updates_required", False))


shot_critical_controller_v2223 = ShotCriticalControllerV2223()


def _install_camera_manager_fast_update() -> None:
    from src.engine.camera.camera_manager import CameraManager

    if getattr(CameraManager, "_v2223_fast_update_patch", False):
        return

    def patched_update(self) -> None:
        """Main-thread pickup only; capability probing is not a frame task.

        Crucially, this does NOT mutate ``_last_pickup_count``.  That cursor
        belongs to HitScanner.get_new_frames_since_last_pickup().  The previous
        update() consumed the count before HitScanner read it, which is why the
        V2.22.2 diagnostic could show frames=0->0 despite a 30 FPS camera.
        """
        if not self.running:
            return
        with self._lock:
            if self._ring:
                self.latest_frame = self._ring[-1]
            self.last_error = None

    def refresh_capabilities_v2223(self, *, force: bool = False) -> None:
        # Explicit/slow-path refresh for settings/status screens.  start() still
        # performs the authoritative initial probe.
        now = time.monotonic()
        last = float(getattr(self, "_v2223_last_capability_probe_mono", 0.0) or 0.0)
        if not force and now - last < 5.0:
            return
        if self.cap is None:
            return
        try:
            from src.engine.camera.camera_capabilities import probe_camera_capabilities
            self.capabilities = probe_camera_capabilities(self.cap)
            self._v2223_last_capability_probe_mono = now
        except Exception:
            pass

    CameraManager.update = patched_update
    CameraManager.refresh_capabilities_v2223 = refresh_capabilities_v2223
    CameraManager._v2223_fast_update_patch = True


def _install_recent_pre_snapshot() -> None:
    """Make diagnostic/AI PRE mean 'what existed before THIS shot'."""
    from src.engine.camera.hit_scanner import HitScanner, camera_manager

    if getattr(HitScanner, "_v2223_recent_pre_patch", False):
        return

    def patched_capture_pre_shot_snapshot(self, peak_ts: float) -> None:
        ring = camera_manager.get_ring_snapshot()
        best = select_recent_pre_frame_v2223(ring, float(peak_ts))
        if best is not None:
            try:
                self.pre_shot_snapshot = cv2.cvtColor(best.frame_bgr, cv2.COLOR_BGR2GRAY)
                self.pre_shot_snapshot_ts = float(best.timestamp)
                age_ms = max(0.0, (float(peak_ts) - float(best.timestamp)) * 1000.0)
                print(f"[PRE-SHOT] recent ring frame age={age_ms:.1f}ms (V2.22.3)")
                return
            except Exception:
                pass
        # Static scene reference remains a surface/repair/projector baseline and
        # a fail-open fallback when no safe ring frame exists.
        if getattr(self, "scene_reference_gray", None) is not None:
            self.pre_shot_snapshot = self.scene_reference_gray.copy()
            self.pre_shot_snapshot_ts = float(peak_ts) - 1.0
            print("[PRE-SHOT] static scene_reference fallback (V2.22.3)")
            return
        self.pre_shot_snapshot = None
        self.pre_shot_snapshot_ts = 0.0

    HitScanner._capture_pre_shot_snapshot = patched_capture_pre_shot_snapshot
    HitScanner._v2223_recent_pre_patch = True


def _install_ai_training_cursor_policy() -> None:
    """Hide pointer while armed; keep explicit F3 mouse-shot debug capability."""
    try:
        from src.engine.scenes.ai_training import AITrainingScene
    except Exception:
        return
    if getattr(AITrainingScene, "_v2223_cursor_policy_patch", False):
        return

    original_on_enter = AITrainingScene.on_enter
    original_update = AITrainingScene.update
    original_handle_event = AITrainingScene.handle_event

    def desired_visible(self) -> bool:
        if bool(getattr(self, "_v2223_mouse_shot_debug", False)):
            return True
        if getattr(self, "_auto_cal_phase", None) is not None:
            return True
        if bool(getattr(self, "awaiting_click", False)):
            return True
        if bool(getattr(self, "_reviewing", False)):
            return True
        if bool(getattr(self, "auto_report_visible", False)):
            return True
        return False

    def patched_on_enter(self):
        result = original_on_enter(self)
        self._v2223_mouse_shot_debug = False
        self._v2223_latency_cursor = True
        shot_critical_controller_v2223.latency_cursor_enabled = True
        return result

    def patched_handle_event(self, event):
        try:
            import pygame
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F3:
                self._v2223_mouse_shot_debug = not bool(getattr(self, "_v2223_mouse_shot_debug", False))
                self.status_message = (
                    "Mus-skott DEBUG: PÅ (F3 för av)" if self._v2223_mouse_shot_debug
                    else "Mus-skott DEBUG: AV (F3 för på)"
                )
                self._set_cursor_visible(desired_visible(self))
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F4:
                self._v2223_latency_cursor = not bool(getattr(self, "_v2223_latency_cursor", True))
                shot_critical_controller_v2223.latency_cursor_enabled = bool(self._v2223_latency_cursor)
                self.status_message = (
                    "Latency-timglas: PÅ (F4 för av)" if self._v2223_latency_cursor
                    else "Latency-timglas: AV (F4 för på)"
                )
                return None
        except Exception:
            pass
        return original_handle_event(self, event)

    def patched_update(self, dt: float):
        result = original_update(self, dt)
        # V2.22.2 may restore the old cursor after resolving the shot.  Re-apply
        # the scene's semantic cursor state after the scene has updated.
        if not shot_critical_controller_v2223._cursor_wait_active:
            try:
                self._set_cursor_visible(desired_visible(self))
            except Exception:
                pass
        return result

    AITrainingScene.on_enter = patched_on_enter
    AITrainingScene.handle_event = patched_handle_event
    AITrainingScene.update = patched_update
    AITrainingScene._v2223_cursor_policy_patch = True


def _install_game_context_bridge() -> None:
    try:
        from src.engine.ai.runtime import get_ai_runtime
        runtime = get_ai_runtime()

        def provider(*, shot_id: int, peak_ts: float, runtime=None):
            return object_hit_registry_v2223.game_context(int(shot_id))

        runtime.set_game_context_provider(provider)
    except Exception:
        pass


def install_v2223_runtime(AppClass: Any) -> None:
    """Install V2.22.3 once. Called explicitly by main.py."""
    global _INSTALLED
    if _INSTALLED:
        return

    _install_camera_manager_fast_update()
    _install_recent_pre_snapshot()
    _install_ai_training_cursor_policy()
    _install_game_context_bridge()

    if getattr(AppClass, "_v2223_shot_critical_patch", False):
        _INSTALLED = True
        return

    def run_v2223(self) -> None:
        import pygame
        from config import FPS
        from src.engine.app import AUTOMATION_EVENT
        from src.engine.audio.audio_peak_detector import audio_peak_detector
        from src.engine.camera.camera_manager import camera_manager
        from src.engine.camera.hit_scanner import hit_scanner
        from src.engine.output.led_service import led_service

        controller = shot_critical_controller_v2223
        controller.last_seen_peak_ts = float(getattr(audio_peak_detector, "last_peak_ts", 0.0) or 0.0)

        while self.running:
            # AUDIO/SHOT CHECK IS THE FIRST ENGINE DECISION IN THE LOOP.
            # Do not even wait for the normal FPS limiter when the audio thread
            # has already observed a shot or a shot is still being resolved.
            priority_pending = (
                controller.pending_audio(audio_peak_detector)
                or controller.scanner_has_open_shot(hit_scanner)
            )
            dt = self.clock.tick(0 if priority_pending else FPS) / 1000.0

            # 1) A microphone shot outranks every ordinary engine task.  The
            # reader thread has already timestamped it; dispatch that queued
            # event before camera housekeeping, automation, scene simulation or
            # rendering.
            new_peak = controller.begin_pending_audio(self, audio_peak_detector)
            controller.prepare_object_snapshot(hit_scanner, self.scene, new_peak)
            t = time.perf_counter()
            audio_peak_detector.update()
            audio_ms = (time.perf_counter() - t) * 1000.0

            # The audio callback creates HitScanner's AudioShotEvent. Snapshot
            # game-object geometry immediately, before moving objects update.
            controller.discover_scanner_events(hit_scanner, self.scene)

            # 2) Cheap frame pickup only. Camera capability probing was removed
            # from this hot path by _install_camera_manager_fast_update().
            t = time.perf_counter()
            camera_manager.update()
            camera_ms = (time.perf_counter() - t) * 1000.0

            # 3) Service the scanner before automation, scene update or render.
            # With the CameraManager pickup-cursor fix, HitScanner now owns its
            # new-frame cursor and can actually consume the timestamped ring.
            t = time.perf_counter()
            hit_scanner.update(dt)
            scanner_ms = (time.perf_counter() - t) * 1000.0
            controller.discover_scanner_events(hit_scanner, self.scene)
            controller.note_stage(
                audio_dispatch_ms=audio_ms if new_peak is not None else 0.0,
                camera_update_ms=camera_ms,
                scanner_update_ms=scanner_ms,
            )
            controller.enrich_spatial_context(hit_scanner)
            controller.evaluate_object_shadow(hit_scanner)
            controller.update_finished(hit_scanner)

            # During a REAL physical shot, freeze nonessential scene simulation
            # and decorative rendering until HitScanner resolves it. The last
            # projected frame remains on the wall, which also stabilises the
            # physical evidence. Synthetic/F2 training explicitly bypasses the
            # freeze because the fake hole must be revealed/rendered.
            if controller.should_defer_scene_work(self.scene, hit_scanner):
                try:
                    pygame.event.pump()
                except Exception:
                    pass
                continue

            # 4) Everything below is ordinary engine work and therefore lower
            # priority than a pending physical shot.
            self._post_automation_events()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                    break
                if event.type == AUTOMATION_EVENT:
                    self._handle_automation_event(event)
                    continue
                switch = self.scene.handle_event(event)
                if switch:
                    self._switch_to(switch.new_scene)
                    break

            if not self.running:
                break

            switch = self.scene.update(dt)
            if switch:
                self._switch_to(switch.new_scene)

            self._update_window_caption()
            self.scene.render(self.screen)
            pygame.display.flip()
            controller.mark_visible_after_flip()

            # Deliberately NO automatic camera-capability probe here. Even a
            # five-second throttled probe could begin just before a shot and
            # block the main thread. refresh_capabilities_v2223(force=True) is
            # available for explicit settings/status actions only.

        self.scene.on_exit()
        hit_scanner.disable()
        try:
            led_service.stop()
        except Exception:
            pass
        audio_peak_detector.stop()
        camera_manager.stop()
        self.communication_server.stop()
        pygame.quit()

    AppClass.run = run_v2223
    AppClass._v2223_shot_critical_patch = True
    _INSTALLED = True
    print("[V2.22.3] shot-critical runtime + object-hit shadow foundation installed")


__all__ = [
    "SCHEMA_VERSION",
    "ShotTimingV2223",
    "ShotCriticalControllerV2223",
    "shot_critical_controller_v2223",
    "select_recent_pre_frame_v2223",
    "install_v2223_runtime",
]
