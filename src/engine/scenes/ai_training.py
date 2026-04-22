"""
AI Training Scene.

Flow:
1. Show a clean training surface (white/black/grid etc.)
2. Wait for a shot (audio peak triggers hit_scanner)
3. AI captures pre/post frames and ranks all candidates
4. User clicks roughly where the hit was
5. AI learns immediately from that click
6. Reset and repeat

The click is transformed to camera space. The AI compares it against
shown candidates and learns positive/negative examples.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any

import pygame

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

from src.engine.ai.runtime import get_ai_runtime
from src.engine.ai.space_mapper import project_screen_point
from src.engine.camera.camera_manager import camera_manager
from src.engine.camera.hit_scanner import hit_scanner
from src.engine.input.hit_input import hit_input
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_viewport_rect


BG_WHITE = (248, 248, 248)
BG_BLACK = (18, 18, 18)
TEXT_LIGHT = (245, 245, 245)
TEXT_DARK = (30, 30, 30)
CYAN = (80, 220, 255)
YELLOW = (255, 210, 70)
ORANGE = (255, 150, 80)
GREEN = (120, 255, 120)
CLICK_COLOR = (255, 110, 110)
HUD_BG = (0, 0, 0, 120)
GRID_LINE = (222, 222, 222)
WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)


class AITrainingScene(Scene):
    """AI training scene following the standard Scene protocol."""

    wants_hit_scanning = True
    wants_camera_preview = False

    MODE_NAMES = ["white", "white_grid", "gray", "black", "checker", "checker_anim", "bubbles"]

    def __init__(self, bg_color=None, **kwargs) -> None:
        super().__init__()
        self.runtime = get_ai_runtime()

        self._bubbles: list[dict] = []  # For bubbles mode

        # Determine initial background mode from bg_color hint
        if isinstance(bg_color, (tuple, list)) and len(bg_color) >= 3:
            brightness = sum(int(c) for c in bg_color[:3]) / 3.0
            self.bg_mode_index = 1 if brightness < 96 else 0
        else:
            self.bg_mode_index = 0

        self.font: pygame.font.Font | None = None
        self.small: pygame.font.Font | None = None
        self.tiny: pygame.font.Font | None = None

        self.viewport: pygame.Rect | None = None

        # Shot state
        self.awaiting_click = False
        self.ranked_candidates: list[dict[str, Any]] = []
        self.clicked_camera_xy: tuple[float, float] | None = None
        self.last_learning_result: dict[str, Any] | None = None
        self.click_flash_timer = 0.0
        self.status_message = ""

        # Deferred capture state — clean render before camera capture
        self._pending_click_camera: tuple[float, float] | None = None
        self._pending_click_phase: str | None = None  # "clean_render" → "wait_frame" → "capture"
        self._pending_wait_frames: int = 0

        # Review state — shows pre/post patches after training
        self._reviewing = False
        self._review_pre_surface: pygame.Surface | None = None
        self._review_post_surface: pygame.Surface | None = None

        # Animation
        self.t = 0.0
        self._animation_frozen = False
        self._last_peak_ts = 0.0

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 34)
        self.small = pygame.font.Font(None, 24)
        self.tiny = pygame.font.Font(None, 18)
        self.viewport = load_viewport_rect()
        self.runtime = get_ai_runtime()
        self._reset_shot_state()
        # Sync peak timestamp to avoid false trigger from old peaks
        from src.engine.audio.audio_peak_detector import audio_peak_detector
        self._last_peak_ts = audio_peak_detector.last_peak_ts

    def on_exit(self) -> None:
        pass

    def _reset_shot_state(self) -> None:
        self.awaiting_click = False
        self.ranked_candidates = []
        self.clicked_camera_xy = None
        self.last_learning_result = None
        self.click_flash_timer = 0.0
        self._animation_frozen = False
        self._reviewing = False
        self._review_pre_surface = None
        self._review_post_surface = None
        self._pending_click_camera = None
        self._pending_click_phase = None
        self._pending_wait_frames = 0
        # Sync so old peaks don't re-trigger freeze
        from src.engine.audio.audio_peak_detector import audio_peak_detector
        self._last_peak_ts = audio_peak_detector.last_peak_ts

    def _enter_review(self, click_camera_xy: tuple[float, float], fresh_post_gray=None) -> None:
        """Build zoomed pre/post patch surfaces for visual review."""
        self._reviewing = True
        self._review_pre_surface = None
        self._review_post_surface = None

        gray_pre = self.runtime.pre_shot_gray
        gray_post = fresh_post_gray if fresh_post_gray is not None else self.runtime.post_shot_gray
        if gray_post is None:
            return

        ix, iy = int(round(click_camera_xy[0])), int(round(click_camera_xy[1]))
        h, w = gray_post.shape[:2]
        patch_r = 80  # Large area — ~160x160 pixels from 4K camera
        x0, y0 = max(0, ix - patch_r), max(0, iy - patch_r)
        x1, y1 = min(w, ix + patch_r + 1), min(h, iy + patch_r + 1)

        if x1 <= x0 or y1 <= y0:
            return

        post_patch = gray_post[y0:y1, x0:x1]
        self._review_post_surface = self._gray_patch_to_surface(post_patch)

        if gray_pre is not None and gray_pre.shape == gray_post.shape:
            pre_patch = gray_pre[y0:y1, x0:x1]
            self._review_pre_surface = self._gray_patch_to_surface(pre_patch)

    @staticmethod
    def _gray_patch_to_surface(patch) -> pygame.Surface | None:
        """Convert a small grayscale numpy patch to a pygame surface."""
        if patch is None or patch.size == 0:
            return None
        try:
            if cv2 is not None:
                rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
            else:
                rgb = np.stack([patch, patch, patch], axis=-1)
            h, w = rgb.shape[:2]
            return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
        except Exception:
            return None

    @staticmethod
    def _save_hole_image(click_camera_xy, post_gray) -> None:
        """Save the post-shot patch as a transparent PNG for the hole image bank."""
        if post_gray is None or np is None:
            return
        try:
            from pathlib import Path

            ix, iy = int(round(click_camera_xy[0])), int(round(click_camera_xy[1]))
            h, w = post_gray.shape[:2]
            patch_r = 64  # 128x128 patch
            x0, y0 = max(0, ix - patch_r), max(0, iy - patch_r)
            x1, y1 = min(w, ix + patch_r), min(h, iy + patch_r)

            if x1 <= x0 or y1 <= y0:
                return

            patch = post_gray[y0:y1, x0:x1]
            if patch.size == 0:
                return

            # Create RGBA with alpha based on distance from white (background).
            # White pixels → transparent, dark pixels (hole) → opaque.
            # Alpha = 75% max to keep some background context.
            if cv2 is not None:
                rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
            else:
                rgb = np.stack([patch, patch, patch], axis=-1)

            # Alpha: darker pixels = more opaque, white = transparent
            # Invert brightness: 255 (white) → 0 alpha, 0 (black) → 191 alpha (75%)
            alpha = (255 - patch).astype(np.float32) * (191.0 / 255.0)
            alpha = alpha.astype(np.uint8)

            rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = rgb
            rgba[:, :, 3] = alpha

            # Save to content/ai/holes/
            holes_dir = Path("content/ai/holes")
            holes_dir.mkdir(parents=True, exist_ok=True)

            # Find next number
            existing = sorted(holes_dir.glob("*.png"))
            next_num = 1
            for f in existing:
                try:
                    num = int(f.stem)
                    next_num = max(next_num, num + 1)
                except ValueError:
                    pass

            path = holes_dir / f"{next_num}.png"

            if cv2 is not None:
                # cv2 expects BGRA
                bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(str(path), bgra)
            else:
                # Fallback: save as pygame surface
                surf = pygame.image.frombuffer(rgba.tobytes(), (rgba.shape[1], rgba.shape[0]), "RGBA")
                pygame.image.save(surf, str(path))

            print(f"[AI HOLE] saved {path.name} ({rgba.shape[1]}x{rgba.shape[0]})")
        except Exception as exc:
            print(f"[AI HOLE] save failed: {exc}")

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event):
        # If reviewing, any click or key dismisses and resumes
        if self._reviewing:
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN):
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    from src.engine.scenes.menu import MenuScene
                    return SceneSwitch(MenuScene())
                self._reset_shot_state()
                return None
            return None

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                from src.engine.scenes.menu import MenuScene
                return SceneSwitch(MenuScene())

            if event.key == pygame.K_TAB:
                self.bg_mode_index = (self.bg_mode_index + 1) % len(self.MODE_NAMES)
                return None

            if event.key == pygame.K_r:
                self.runtime.memory.reset()
                self._reset_shot_state()
                self.status_message = "AI nollställd."
                return None

            if event.key == pygame.K_SPACE and not self.awaiting_click:
                self._on_shot_detected()
                return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.awaiting_click:
                self._on_training_click(event.pos)
            return None

        return None

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, dt: float):
        # Check for audio peak directly from audio thread's last_peak_ts.
        from src.engine.audio.audio_peak_detector import audio_peak_detector
        current_peak_ts = audio_peak_detector.last_peak_ts
        if current_peak_ts > self._last_peak_ts and not self._animation_frozen:
            self._animation_frozen = True
        self._last_peak_ts = current_peak_ts

        # Only advance animation time when not frozen
        if not self._animation_frozen:
            self.t += dt

        # Check if AI runtime detected a new shot (for candidate display)
        if not self.awaiting_click and not self._reviewing and self.runtime.has_new_shot:
            self._on_shot_detected()

        # Deferred capture for review images
        if self._pending_click_phase == "wait_frame":
            self._pending_wait_frames += 1
            if self._pending_wait_frames >= 5:
                self._pending_click_phase = "capture"
        elif self._pending_click_phase == "capture":
            self._do_clean_capture()

        return None

    def _do_clean_capture(self) -> None:
        """Capture clean camera frame, train AI, and show review."""
        click_camera = self._pending_click_camera
        self._pending_click_phase = None
        self._pending_click_camera = None
        self._pending_wait_frames = 0
        pygame.mouse.set_visible(True)

        if click_camera is None:
            return

        # Capture fresh clean post-shot (no overlays/cursor on projector)
        clean_post_gray = None
        try:
            frame_bgr = camera_manager.get_latest_frame()
            if frame_bgr is not None and cv2 is not None:
                clean_post_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass

        post_gray = clean_post_gray if clean_post_gray is not None else self.runtime.post_shot_gray
        pre_gray = self.runtime.pre_shot_gray

        # Debug: compare pre and post
        import time as _dbg_time
        now = _dbg_time.time()
        pre_ts = getattr(self.runtime, '_pre_shot_ts', 0.0)
        print(f"[AI CAPTURE] now={now:.3f}")
        print(f"[AI CAPTURE] pre_gray ts={pre_ts:.3f} (age={((now - pre_ts) * 1000):.0f}ms)")
        print(f"[AI CAPTURE] post_gray: {'FRESH' if clean_post_gray is not None else 'RUNTIME'} (taken now)")
        print(f"[AI CAPTURE] same object: {pre_gray is post_gray}")
        if pre_gray is not None and post_gray is not None and pre_gray.shape == post_gray.shape:
            diff = float(np.mean(np.abs(pre_gray.astype(np.int16) - post_gray.astype(np.int16))))
            print(f"[AI CAPTURE] mean pixel diff: {diff:.2f} (>1 = different frames)")
        else:
            print(f"[AI CAPTURE] cannot compare (different shapes or None)")

        # Train AI on the CLEAN image — same image we show in review
        result = self.runtime.learn_from_click(
            click_camera_xy=click_camera,
            shown_candidates=self.ranked_candidates,
            gray_pre=pre_gray,
            gray_post=post_gray,
        )
        self.last_learning_result = result

        self.runtime.study_click_area(
            click_camera_xy=click_camera,
            gray_pre=pre_gray,
            gray_post=post_gray,
        )

        if result.get("positive_added"):
            dist = result.get("nearest_distance", 999)
            if dist <= float(self.runtime.settings.get("click_match_radius_px", 42)):
                idx = result.get("nearest_index", 0)
                self.status_message = (
                    f"Tränade: kandidat #{idx + 1} (avstånd {dist:.0f}px). "
                    f"+{result.get('negatives_added', 0)} neg. "
                    f"Totalt: {result['total_positives']} pos / {result['total_negatives']} neg"
                )
            else:
                self.status_message = (
                    f"Syntetiskt positivt (ingen kandidat nära). "
                    f"Totalt: {result['total_positives']} pos / {result['total_negatives']} neg"
                )

        # Save hole image to build up training image bank
        self._save_hole_image(click_camera, post_gray)

        # Show review — same clean images AI trained on
        self._enter_review(click_camera, post_gray)

    def _on_shot_detected(self) -> None:
        """A shot was detected — rank candidates and wait for click."""
        # Get all candidates from hit_scanner (not just top 10)
        all_candidates = list(hit_scanner.last_candidates)
        if not all_candidates:
            self.status_message = "Skott detekterat men inga kandidater."
            return

        self.ranked_candidates = self.runtime.rank_candidates(all_candidates, limit=50)
        self.awaiting_click = True
        self.clicked_camera_xy = None
        self.status_message = f"Skott! {len(self.ranked_candidates)} kandidater. Klicka var du träffade."

    def _on_training_click(self, screen_pos: tuple[int, int]) -> None:
        """User clicked to indicate where the hit was."""
        projected = project_screen_point(float(screen_pos[0]), float(screen_pos[1]))
        click_camera = (projected.camera_x, projected.camera_y)
        self.clicked_camera_xy = click_camera

        # Schedule clean capture — ALL training happens AFTER we get a clean
        # camera image without overlays/cursor. That way AI trains on the same
        # clean image that we show in review.
        self._pending_click_camera = click_camera
        self._pending_click_phase = "clean_render"
        self.awaiting_click = False

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, screen: pygame.Surface) -> None:
        mode = self.MODE_NAMES[self.bg_mode_index]
        vp = self.viewport or pygame.Rect(0, 0, screen.get_width(), screen.get_height())

        # Fill area outside viewport with dark border
        screen.fill((30, 30, 30))

        self._render_background(screen, mode, vp)

        # During clean capture: render only background, hide overlays + cursor
        if self._pending_click_phase in ("clean_render", "wait_frame", "capture"):
            if self._pending_click_phase == "clean_render":
                self._pending_click_phase = "wait_frame"
                self._pending_wait_frames = 0
                pygame.mouse.set_visible(False)
            return

        self._render_candidates(screen, vp)
        self._render_click_feedback(screen)

        # Review overlay — shows pre/post patches after training click
        if self._reviewing:
            self._render_review(screen, vp)

        self._render_hud(screen, mode, vp)

    def _render_review(self, screen: pygame.Surface, vp: pygame.Rect) -> None:
        """Show zoomed pre/post patches side by side in the center of viewport."""
        zoom = 3  # Scale factor — 160px patch × 3 = 480px display
        gap = 24

        pre = self._review_pre_surface
        post = self._review_post_surface

        if post is None:
            return

        pw, ph = post.get_size()
        scaled_w, scaled_h = pw * zoom, ph * zoom
        label_h = 22
        total_w = scaled_w * 2 + gap if pre is not None else scaled_w
        total_h = scaled_h + label_h + 30

        # Background panel
        panel_rect = pygame.Rect(0, 0, total_w + 40, total_h + 20)
        panel_rect.center = (vp.centerx, vp.centery)
        panel = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel.fill((0, 0, 0, 200))
        pygame.draw.rect(panel, (120, 120, 120), panel.get_rect(), 2)
        screen.blit(panel, panel_rect.topleft)

        x_start = panel_rect.x + 20
        y_start = panel_rect.y + 10

        # Pre-shot (left)
        if pre is not None:
            scaled_pre = pygame.transform.scale(pre, (scaled_w, scaled_h))
            screen.blit(scaled_pre, (x_start, y_start + label_h))
            pygame.draw.rect(screen, CYAN, (x_start, y_start + label_h, scaled_w, scaled_h), 2)
            cx_pre = x_start + scaled_w // 2
            cy_pre = y_start + label_h + scaled_h // 2
            pygame.draw.line(screen, CYAN, (cx_pre - 10, cy_pre), (cx_pre + 10, cy_pre), 1)
            pygame.draw.line(screen, CYAN, (cx_pre, cy_pre - 10), (cx_pre, cy_pre + 10), 1)
            if self.tiny:
                label = self.tiny.render("FÖRE skott", True, CYAN)
                screen.blit(label, (x_start, y_start))

        # Post-shot (right, or center if no pre)
        post_x = x_start + (scaled_w + gap if pre is not None else 0)
        scaled_post = pygame.transform.scale(post, (scaled_w, scaled_h))
        screen.blit(scaled_post, (post_x, y_start + label_h))
        pygame.draw.rect(screen, ORANGE, (post_x, y_start + label_h, scaled_w, scaled_h), 2)
        cx_post = post_x + scaled_w // 2
        cy_post = y_start + label_h + scaled_h // 2
        pygame.draw.line(screen, ORANGE, (cx_post - 10, cy_post), (cx_post + 10, cy_post), 1)
        pygame.draw.line(screen, ORANGE, (cx_post, cy_post - 10), (cx_post, cy_post + 10), 1)
        if self.tiny:
            label = self.tiny.render("EFTER skott", True, ORANGE)
            screen.blit(label, (post_x, y_start))

        # Hint
        if self.tiny:
            hint = self.tiny.render("Klicka eller tryck för att fortsätta", True, WHITE)
            hint_rect = hint.get_rect(centerx=panel_rect.centerx, top=y_start + label_h + scaled_h + 6)
            screen.blit(hint, hint_rect)

    def _render_background(self, screen: pygame.Surface, mode: str, vp: pygame.Rect) -> None:
        if mode == "black":
            pygame.draw.rect(screen, BG_BLACK, vp)
        elif mode == "gray":
            pygame.draw.rect(screen, (128, 128, 128), vp)
        elif mode == "white_grid":
            pygame.draw.rect(screen, BG_WHITE, vp)
            for x in range(vp.left, vp.right, 48):
                pygame.draw.line(screen, GRID_LINE, (x, vp.top), (x, vp.bottom), 1)
            for y in range(vp.top, vp.bottom, 48):
                pygame.draw.line(screen, GRID_LINE, (vp.left, y), (vp.right, y), 1)
        elif mode == "checker":
            self._draw_checker_static(screen, vp)
        elif mode == "checker_anim":
            self._draw_checker_anim(screen, vp)
        elif mode == "bubbles":
            self._draw_bubbles(screen, vp)
        else:
            pygame.draw.rect(screen, BG_WHITE, vp)

        # Viewport border
        pygame.draw.rect(screen, (0, 180, 0), vp, 2)

    def _draw_checker_static(self, screen: pygame.Surface, vp: pygame.Rect) -> None:
        """Static checkerboard — tests detection against high-contrast edges."""
        cell = 40
        colors = [(220, 220, 220), (60, 60, 60)]
        for row, y in enumerate(range(vp.top, vp.bottom, cell)):
            for col, x in enumerate(range(vp.left, vp.right, cell)):
                color = colors[(row + col) % 2]
                rect = pygame.Rect(x, y, min(cell, vp.right - x), min(cell, vp.bottom - y))
                pygame.draw.rect(screen, color, rect)

    def _draw_checker_anim(self, screen: pygame.Surface, vp: pygame.Rect) -> None:
        """Animated checkerboard — scrolls diagonally, freezes on shot."""
        cell = 40
        colors = [(220, 220, 220), (60, 60, 60)]
        # Offset scrolls when not frozen
        if not self._animation_frozen:
            offset = int(self.t * 60) % (cell * 2)
        else:
            if not hasattr(self, "_checker_frozen_offset"):
                self._checker_frozen_offset = 0
            offset = self._checker_frozen_offset

        if not self._animation_frozen:
            self._checker_frozen_offset = offset

        for y in range(vp.top - cell, vp.bottom + cell, cell):
            for x in range(vp.left - cell, vp.right + cell, cell):
                ax = x + offset
                ay = y + offset
                row = (ay - vp.top) // cell
                col = (ax - vp.left) // cell
                color = colors[(row + col) % 2]
                rect = pygame.Rect(ax, ay, cell, cell).clip(vp)
                if rect.w > 0 and rect.h > 0:
                    pygame.draw.rect(screen, color, rect)

    def _draw_bubbles(self, screen: pygame.Surface, vp: pygame.Rect) -> None:
        """Moving shapes that freeze when awaiting click — fun to shoot at."""
        pygame.draw.rect(screen, BG_WHITE, vp)

        # Spawn bubbles if empty — each with unique speed
        if not self._bubbles:
            for _ in range(15):
                speed = random.uniform(0.06, 0.25)
                angle = random.uniform(0, 2 * math.pi)
                self._bubbles.append({
                    "x": random.uniform(0.1, 0.9),
                    "y": random.uniform(0.1, 0.9),
                    "r": random.uniform(0.03, 0.08),
                    "dx": math.cos(angle) * speed,
                    "dy": math.sin(angle) * speed,
                    "color": (
                        random.randint(40, 220),
                        random.randint(40, 220),
                        random.randint(40, 220),
                    ),
                    "shape": random.choice(["circle", "rect", "triangle"]),
                })

        # Move bubbles only when NOT frozen (freeze instantly on shot)
        if not self._animation_frozen:
            dt = 1.0 / 60.0
            for b in self._bubbles:
                b["x"] += b["dx"] * dt
                b["y"] += b["dy"] * dt
                # Bounce with slight random deflection
                if b["x"] < 0.05 or b["x"] > 0.95:
                    b["dx"] *= -1
                    b["dx"] += random.uniform(-0.02, 0.02)
                    b["dy"] += random.uniform(-0.01, 0.01)
                    b["x"] = max(0.05, min(0.95, b["x"]))
                if b["y"] < 0.05 or b["y"] > 0.95:
                    b["dy"] *= -1
                    b["dy"] += random.uniform(-0.02, 0.02)
                    b["dx"] += random.uniform(-0.01, 0.01)
                    b["y"] = max(0.05, min(0.95, b["y"]))

        # Draw
        for b in self._bubbles:
            cx = int(vp.left + b["x"] * vp.w)
            cy = int(vp.top + b["y"] * vp.h)
            r = int(b["r"] * min(vp.w, vp.h))
            color = b["color"]

            if b["shape"] == "circle":
                pygame.draw.circle(screen, color, (cx, cy), r)
                pygame.draw.circle(screen, (30, 30, 30), (cx, cy), r, 2)
            elif b["shape"] == "rect":
                rect = pygame.Rect(cx - r, cy - r, r * 2, r * 2)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (30, 30, 30), rect, 2)
            elif b["shape"] == "triangle":
                points = [
                    (cx, cy - r),
                    (cx + r, cy + r),
                    (cx - r, cy + r),
                ]
                pygame.draw.polygon(screen, color, points)
                pygame.draw.polygon(screen, (30, 30, 30), points, 2)

    def _render_candidates(self, screen: pygame.Surface, vp: pygame.Rect) -> None:
        if not self.awaiting_click or not self.ranked_candidates:
            return

        for cand in self.ranked_candidates:
            rank = int(cand.get("rank", 99))
            cam_x = cand.get("camera_x", 0.0)
            cam_y = cand.get("camera_y", 0.0)

            # Transform to screen
            try:
                from src.engine.input.hit_input import hit_input
                sx, sy = hit_input._canonical_camera_to_screen(cam_x, cam_y)
            except Exception:
                continue

            if not (math.isfinite(sx) and math.isfinite(sy)):
                continue

            ix, iy = int(round(sx)), int(round(sy))

            # Color by rank
            if rank == 1:
                color = ORANGE
                radius = 18
                width = 3
            elif rank <= 3:
                color = YELLOW
                radius = 14
                width = 2
            else:
                color = CYAN
                radius = 10
                width = 2

            pygame.draw.circle(screen, color, (ix, iy), radius, width)
            # Crosshair
            pygame.draw.line(screen, color, (ix - radius - 4, iy), (ix + radius + 4, iy), 1)
            pygame.draw.line(screen, color, (ix, iy - radius - 4), (ix, iy + radius + 4), 1)

            # Rank number
            if self.small is not None:
                label = self.small.render(str(rank), True, color)
                screen.blit(label, (ix + radius + 4, iy - 10))

            # Score
            if self.tiny is not None:
                ai_score = cand.get("ai_score", 0.0)
                combined = cand.get("combined_score", 0.0)
                score_text = f"ai:{ai_score:.2f} c:{combined:.2f}"
                score_surf = self.tiny.render(score_text, True, color)
                screen.blit(score_surf, (ix + radius + 4, iy + 6))

    def _render_click_feedback(self, screen: pygame.Surface) -> None:
        if self.clicked_camera_xy is None:
            return
        try:
            from src.engine.input.hit_input import hit_input
            sx, sy = hit_input._canonical_camera_to_screen(
                self.clicked_camera_xy[0], self.clicked_camera_xy[1]
            )
        except Exception:
            return

        ix, iy = int(round(sx)), int(round(sy))
        pygame.draw.circle(screen, CLICK_COLOR, (ix, iy), 16, 3)
        pygame.draw.line(screen, CLICK_COLOR, (ix - 22, iy), (ix + 22, iy), 2)
        pygame.draw.line(screen, CLICK_COLOR, (ix, iy - 22), (ix, iy + 22), 2)

    def _render_hud(self, screen: pygame.Surface, mode: str, vp: pygame.Rect) -> None:
        if self.font is None or self.small is None or self.tiny is None:
            return

        sw, sh = screen.get_size()
        is_dark = mode == "black"
        ink = TEXT_LIGHT if is_dark else TEXT_DARK

        # Top bar
        top_bar = pygame.Surface((sw, 36), pygame.SRCALPHA)
        top_bar.fill(HUD_BG)
        screen.blit(top_bar, (0, 0))

        mode_label = {
            "white": "Vit",
            "white_grid": "Vit + rutnät",
            "gray": "Grå",
            "black": "Svart",
            "checker": "Rutmönster",
            "checker_anim": "Rutmönster (video)",
            "bubbles": "Bubblor",
        }.get(mode, mode)
        summary = self.runtime.memory.summary()
        header = (
            f"AI-träning • {mode_label} • "
            f"{summary['positive_count']} pos / {summary['negative_count']} neg"
        )
        screen.blit(self.small.render(header, True, WHITE), (12, 8))

        # Status indicator
        if self.awaiting_click:
            status_text = "KLICKA var du träffade"
            status_color = YELLOW
        else:
            status_text = "Skjut..."
            status_color = GREEN

        status_surf = self.small.render(status_text, True, status_color)
        screen.blit(status_surf, (sw - status_surf.get_width() - 16, 8))

        # Bottom status message
        if self.status_message:
            bot_bar = pygame.Surface((sw, 28), pygame.SRCALPHA)
            bot_bar.fill(HUD_BG)
            screen.blit(bot_bar, (0, sh - 28))
            screen.blit(self.tiny.render(self.status_message, True, WHITE), (12, sh - 24))

        # Help (bottom right)
        help_text = "TAB=bakgrund  SPACE=manuellt skott  R=nollställ  ESC=tillbaka"
        help_surf = self.tiny.render(help_text, True, SOFT_WHITE if not is_dark else (140, 140, 140))
        screen.blit(help_surf, (sw - help_surf.get_width() - 12, sh - 48))
