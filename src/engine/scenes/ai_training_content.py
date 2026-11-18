"""Test content for the AI training scene (AITrainingScene).

Owns everything that is *projected content* in the in-game AI test scene and
is independent of the detection/round state machine:

- hole profile: which synthetic hole kind/size the scene schedules
  ("co2" real-stamp holes scaled 0.2-0.5, "airgun" real stamps at 1.0, or
  "legacy" = the old procedural kinds);
- extra background modes (image slideshow, dark scene, scrolling landscape,
  moving targets), pre-rendered once per viewport size;
- shot reactions (hit marker / small explosion, score text, screen shake);
- the "freeze at shot" option and a content clock that honours it.

Shot reactions are drawn as projected content BEFORE the hole overlay is
composited, so a (multiplicative) synthetic hole darkens them exactly as a
physical hole darkens whatever the projector paints over it. Screen shake
shifts the projected content only; holes stay at their physical position.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import pygame

from src.engine.synthetic.synthetic_hole_overlay import (
    REAL_STAMP_AR15_KIND,
    REAL_STAMP_KIND,
    REAL_STAMP_LIBRARIES,
    STAMP_NOMINAL_RADIUS_CAM_PX,
    real_stamps_available,
)

# co2 = real CO2 AR15 stamps at true size (2026-11-18); co2_shrunk = the earlier proxy
# (airgun stamps shrunk); airgun = real airgun stamps at true size.
HOLE_PROFILES = ("co2", "airgun", "co2_shrunk", "legacy")
HOLE_PROFILE_SCALE = {"co2": (1.0, 1.0), "co2_shrunk": (0.2, 0.5), "airgun": (1.0, 1.0)}
# PROJECTED holes must look like real holes to the CAMERA.  Projector + camera
# re-imaging loses ~6x of the stamp's darkening, and CO2-size stamps (scale
# 0.2-0.5) are below one projector pixel, so they cannot be shown at all.
# Measured 2026-11-18 (gray 200 board, median stamp, camera depth after sigma 1):
#   scale 1.0: gain 1 -> 0.04, 2 -> 0.13, 3 -> 0.23 (real airgun hole: 0.23)
#   scale 0.6: gain 2.5 -> 0.06, 4 -> 0.17, 6 -> 0.22
#   scale 0.35: gain 1..4 -> 0.007-0.02 (not projectable)
# Target camera depth: airgun 0.17-0.32 (real IQR), CO2 0.05-0.17 (camera-space
# model of real stamps at scale 0.2-0.5).  So projected profiles use
# (footprint scale, optical gain) instead of the physical scale; CO2 holes are
# drawn ~2x larger than real (smallest projectable footprint) with matched contrast.
# Real AR15 stamps are used AS THEY ARE (scale 1.0, own library): their camera core area
# (~22 px, depth 0.05-0.19) is about half the airgun's (40 px, 0.28), i.e. comparable to
# the old shrunk proxy (scale ~0.6), so the same optical gain range applies.
# Projector check 2026-11-18 (evaluation_runs/stamp_projection_20261118, gain 3, white):
# airgun camera core 39.5 px / depth 0.20 vs real 36.5 / 0.21 (matches); AR15 core 24.5 vs
# 22 (size matches) but depth 0.056 vs real 0.095 -> AR15 needs ~5x gain (small holes lose
# more contrast in projector+camera re-imaging).
# Both projected ~8-11 % too large in camera area (and by eye: the stamps already contain the
# camera blur, the projector adds ~1.2 mm/px) -> footprint scale 0.95-1.0 (random; operator
# 2026-11-18). Rotation is random 0-360 deg per hole (SyntheticHoleOverlay.add_hole) plus a
# random mirror; weak AR15 stamps (pellet barely through the paper, low gas) are kept on purpose.
PROJECTED_PROFILE = {"co2": dict(scale=(0.95, 1.0), gain=(4.5, 6.0), kind=REAL_STAMP_AR15_KIND),
                     "co2_shrunk": dict(scale=(0.55, 0.65), gain=(2.5, 4.0), kind=REAL_STAMP_KIND),
                     "airgun": dict(scale=(0.95, 1.0), gain=(2.5, 3.5), kind=REAL_STAMP_KIND)}
EFFECT_LEVELS = ("off", "marker", "marker+score", "all")
NEW_MODE_NAMES = ("images", "dark_scene", "scroll_landscape", "moving_targets")
NEW_MODE_LABELS = {
    "images": "Bildspel",
    "dark_scene": "Mörk scen",
    "scroll_landscape": "Rullande landskap",
    "moving_targets": "Rörliga mål",
}
LEGACY_KINDS = ["clean_hole", "torn_hole", "ragged_hole", "dent_ring", "weak_indent"]
LEGACY_WEIGHTS = [66, 16, 10, 6, 2]
IMAGE_DIR = Path("assets/images")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class AITrainingContent:
    def __init__(self, image_dir: Path | str = IMAGE_DIR, *, hole_profile: str = "co2") -> None:
        self.hole_profile = hole_profile if hole_profile in HOLE_PROFILES else "co2"
        self.effects_level = "off"
        self.freeze_at_shot = True

        self.clock = 0.0
        self._last_tick: float | None = None
        self._cache: dict[tuple, Any] = {}
        self._cam_per_screen: float | None = None

        self.image_paths = sorted(
            p for p in Path(image_dir).glob("*") if p.suffix.lower() in IMAGE_SUFFIXES
        ) if Path(image_dir).is_dir() else []
        self.image_index = 0
        self._image_round_key: float | None = None

        self._markers: list[dict[str, Any]] = []
        self.score = 0
        self._shake_until = 0.0
        self._shake_start = 0.0

    # ------------------------------------------------------------------
    # Settings / toggles
    # ------------------------------------------------------------------
    def cycle_hole_profile(self) -> str:
        i = HOLE_PROFILES.index(self.hole_profile) if self.hole_profile in HOLE_PROFILES else 0
        self.hole_profile = HOLE_PROFILES[(i + 1) % len(HOLE_PROFILES)]
        return self.hole_profile

    def cycle_effects(self) -> str:
        i = EFFECT_LEVELS.index(self.effects_level) if self.effects_level in EFFECT_LEVELS else 0
        self.effects_level = EFFECT_LEVELS[(i + 1) % len(EFFECT_LEVELS)]
        if self.effects_level == "off":
            self._markers.clear()
            self._shake_until = 0.0
        return self.effects_level

    def toggle_freeze(self) -> bool:
        self.freeze_at_shot = not self.freeze_at_shot
        return self.freeze_at_shot

    def effective_hole_profile(self) -> str:
        profile = self.hole_profile
        if profile == "co2" and not real_stamps_available(REAL_STAMP_LIBRARIES[REAL_STAMP_AR15_KIND]):
            profile = "co2_shrunk"   # AR15 library missing -> earlier proxy
        if profile != "legacy" and not real_stamps_available():
            return "legacy"
        return profile

    def status_text(self) -> str:
        profile = self.effective_hole_profile()
        if profile != self.hole_profile:
            profile = f"{self.hole_profile}->{profile} (stämplar saknas)"
        freeze = "på" if self.freeze_at_shot else "av"
        return f"Hål: {profile} | Effekter: {self.effects_level} | Frys vid skott: {freeze}"

    # ------------------------------------------------------------------
    # Hole spec
    # ------------------------------------------------------------------
    def camera_px_per_screen_px(self) -> float:
        """Isotropic camera-px per screen-px from scanport/viewport settings.

        Geometric mean of the width and height ratios (the scanport is not
        aspect-equal to the viewport because of camera perspective).
        Falls back to 2.0 when settings are unavailable.
        """
        if self._cam_per_screen is None:
            value = 2.0
            try:
                from src.engine.settings import load_scanport_rect, load_viewport_rect
                scan = load_scanport_rect()
                vp = load_viewport_rect()
                if scan is not None and vp.w > 0 and vp.h > 0 and scan.w > 0 and scan.h > 0:
                    value = math.sqrt((scan.w / vp.w) * (scan.h / vp.h))
            except Exception:
                pass
            self._cam_per_screen = max(0.25, min(8.0, float(value)))
        return self._cam_per_screen

    def choose_hole(self, rng: random.Random | Any = random) -> tuple[str, float]:
        """Return (kind, radius_px) for the next synthetic hole (screen px)."""
        profile = self.effective_hole_profile()
        if profile == "legacy":
            kind = rng.choices(LEGACY_KINDS, weights=LEGACY_WEIGHTS, k=1)[0]
            return kind, rng.uniform(1.7, 2.9)
        proj = PROJECTED_PROFILE[profile]
        scale = rng.uniform(*proj["scale"])
        self.last_gain = rng.uniform(*proj["gain"])
        # real_stamp radius = stamp scale expressed in screen px (scale 1 = true camera size).
        return proj["kind"], STAMP_NOMINAL_RADIUS_CAM_PX * scale / self.camera_px_per_screen_px()

    def hole_strength(self, rng: random.Random | Any = random) -> float:
        """Strength for the hole spec: optical gain for real stamps, legacy range otherwise."""
        if self.effective_hole_profile() == "legacy":
            return rng.uniform(0.90, 1.18)
        return float(getattr(self, "last_gain", 1.0))

    # ------------------------------------------------------------------
    # Clock / freeze
    # ------------------------------------------------------------------
    def is_frozen(self, scene_frozen: bool) -> bool:
        return bool(scene_frozen) and self.freeze_at_shot

    def tick(self, scene_frozen: bool, now: float | None = None) -> None:
        """Advance the content clock once per rendered frame (wall time)."""
        now = time.monotonic() if now is None else now
        if self._last_tick is not None and not self.is_frozen(scene_frozen):
            self.clock += max(0.0, min(0.1, now - self._last_tick))
        self._last_tick = now

    # ------------------------------------------------------------------
    # Background modes
    # ------------------------------------------------------------------
    def render_background(self, scene: Any, screen: pygame.Surface, mode: str, vp: pygame.Rect) -> bool:
        if mode == "images":
            self._maybe_advance_image(scene)
            self._draw_images(screen, vp)
        elif mode == "dark_scene":
            screen.blit(self._dark_scene(vp.size), vp.topleft)
        elif mode == "scroll_landscape":
            self._draw_scroll_landscape(screen, vp)
        elif mode == "moving_targets":
            self._draw_moving_targets(screen, vp)
        else:
            return False
        return True

    @staticmethod
    def _scene_idle(scene: Any) -> bool:
        return not (
            getattr(scene, "synthetic_trigger_pending", False)
            or getattr(scene, "synthetic_reveal_pending", False)
            or getattr(scene, "awaiting_click", False)
            or getattr(scene, "_reviewing", False)
            or getattr(scene, "single_synth_round_active", False)
            or getattr(scene, "_pending_click_phase", None) is not None
            or getattr(scene, "auto_phase", "idle") not in ("idle", "waiting_next")
        )

    def _maybe_advance_image(self, scene: Any) -> None:
        """Change picture only between rounds, never during a shot."""
        if len(self.image_paths) < 2:
            return
        key = float(getattr(scene, "_last_peak_ts", 0.0) or 0.0)
        if self._image_round_key is None:
            self._image_round_key = key
            return
        if key == self._image_round_key or not self._scene_idle(scene):
            return
        now = time.time()
        if key > 0.0 and now - key < 1.0:
            return
        self._image_round_key = key
        self.image_index = (self.image_index + 1) % len(self.image_paths)
        # Let the camera settle on the new picture before the next trigger.
        try:
            scene.background_settle_until_ts = max(
                float(getattr(scene, "background_settle_until_ts", 0.0)),
                now + float(getattr(scene, "background_settle_delay_s", 0.25)),
            )
        except Exception:
            pass

    def _draw_images(self, screen: pygame.Surface, vp: pygame.Rect) -> None:
        if not self.image_paths:
            screen.blit(self._dark_scene(vp.size), vp.topleft)
            return
        path = self.image_paths[self.image_index % len(self.image_paths)]
        key = ("image", str(path), vp.size)
        surf = self._cache.get(key)
        if surf is None:
            try:
                img = pygame.image.load(str(path))
                surf = pygame.transform.smoothscale(img.convert() if pygame.display.get_surface() else img, vp.size)
            except Exception:
                surf = pygame.Surface(vp.size)
                surf.fill((128, 128, 128))
            self._cache[key] = surf
        screen.blit(surf, vp.topleft)

    def _dark_scene(self, size: tuple[int, int]) -> pygame.Surface:
        key = ("dark_scene", size)
        surf = self._cache.get(key)
        if surf is not None:
            return surf
        w, h = size
        rng = random.Random(7)
        surf = pygame.Surface(size)
        for y in range(h):  # vertical gradient 40 -> 80
            g = int(_lerp(40, 80, y / max(1, h - 1)))
            pygame.draw.line(surf, (g, g, g + 4), (0, y), (w, y))
        pygame.draw.circle(surf, (120, 120, 112), (int(w * 0.82), int(h * 0.16)), max(8, h // 14))  # moon
        x = 0
        while x < w:  # dark skyline with a few lit windows
            bw = rng.randint(w // 16, w // 7)
            bh = rng.randint(h // 5, h // 2)
            rect = pygame.Rect(x, h - bh, bw, bh)
            shade = rng.randint(22, 34)
            pygame.draw.rect(surf, (shade, shade, shade + 2), rect)
            for _ in range(rng.randint(1, 5)):
                wx = rect.x + rng.randint(4, max(5, bw - 12))
                wy = rect.y + rng.randint(6, max(7, bh - 14))
                lit = rng.randint(85, 130)
                pygame.draw.rect(surf, (lit, lit, lit - 10), (wx, wy, 6, 8))
            x += bw + rng.randint(0, 12)
        for _ in range(6):  # foreground crates / bright debris
            cx, cy = rng.randint(0, w), rng.randint(int(h * 0.75), h)
            v = rng.choice((30, 95, 110))
            pygame.draw.rect(surf, (v, v, v), (cx, cy, rng.randint(20, 60), rng.randint(14, 36)))
        self._cache[key] = surf
        return surf

    def _landscape_layers(self, size: tuple[int, int]) -> tuple[pygame.Surface, pygame.Surface]:
        key = ("landscape", size)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        w, h = size
        tw = max(2, w * 2)  # tileable period
        rng = random.Random(11)
        far = pygame.Surface((tw, h))
        for y in range(h):
            t = y / max(1, h - 1)
            pygame.draw.line(far, (int(_lerp(120, 205, t)), int(_lerp(170, 220, t)), int(_lerp(230, 235, t))), (0, y), (tw, y))
        for _ in range(5):  # clouds
            cx, cy = rng.randint(0, tw), rng.randint(10, max(11, h // 4))
            for k in range(4):
                pygame.draw.circle(far, (238, 240, 244), ((cx + k * 22) % tw, cy + (k % 2) * 6), 18)

        def ridge(base: float, amp: float, freqs: tuple[int, ...]) -> list[tuple[int, int]]:
            phases = [rng.uniform(0, 2 * math.pi) for _ in freqs]
            pts = [(0, h)]
            for x in range(0, tw + 1, 4):
                y = base
                for f, p in zip(freqs, phases):
                    y += amp / f * math.sin(2 * math.pi * f * x / tw + p)
                pts.append((x, int(y)))
            pts.append((tw, h))
            return pts

        pygame.draw.polygon(far, (110, 125, 150), ridge(h * 0.50, h * 0.12, (2, 5, 9)))
        pygame.draw.polygon(far, (90, 130, 95), ridge(h * 0.62, h * 0.08, (3, 7)))

        near = pygame.Surface((tw, h))
        colorkey = (255, 0, 255)
        near.fill(colorkey)
        near.set_colorkey(colorkey)
        hill = ridge(h * 0.78, h * 0.06, (4, 10))
        pygame.draw.polygon(near, (70, 120, 55), hill)
        for _ in range(28):  # trees and huts on the near hill
            x = rng.randint(0, tw - 1)
            base_y = h * 0.78 - 4
            if rng.random() < 0.8:
                th = rng.randint(h // 14, h // 7)
                pygame.draw.rect(near, (80, 55, 35), (x - 2, int(base_y - th * 0.3), 4, int(th * 0.4)))
                pygame.draw.polygon(near, (35, 85, 40), [(x, int(base_y - th)), (x - th // 3, int(base_y - th * 0.25)), (x + th // 3, int(base_y - th * 0.25))])
            else:
                bw = rng.randint(24, 44)
                pygame.draw.rect(near, (150, 70, 55), (x, int(base_y - bw * 0.6), bw, int(bw * 0.6)))
                pygame.draw.polygon(near, (90, 60, 50), [(x - 3, int(base_y - bw * 0.6)), (x + bw // 2, int(base_y - bw)), (x + bw + 3, int(base_y - bw * 0.6))])
        pygame.draw.rect(near, (120, 105, 80), (0, int(h * 0.90), tw, h - int(h * 0.90)))  # road
        for x in range(0, tw, 60):
            pygame.draw.rect(near, (215, 210, 190), (x, int(h * 0.945), 30, 4))
        cached = (far.convert() if pygame.display.get_surface() else far, near)
        self._cache[key] = cached
        return cached

    def _draw_scroll_landscape(self, screen: pygame.Surface, vp: pygame.Rect) -> None:
        far, near = self._landscape_layers(vp.size)
        tw = far.get_width()
        old_clip = screen.get_clip()
        screen.set_clip(vp.clip(screen.get_rect()))
        for layer, speed in ((far, 25.0), (near, 70.0)):
            off = int(self.clock * speed) % tw
            screen.blit(layer, (vp.x - off, vp.y))
            screen.blit(layer, (vp.x - off + tw, vp.y))
        screen.set_clip(old_clip)

    def _target_sprites(self, vp_h: int) -> list[pygame.Surface]:
        key = ("targets", vp_h)
        sprites = self._cache.get(key)
        if sprites is not None:
            return sprites
        colorkey = (255, 0, 255)
        sprites = []
        s = max(30, vp_h // 6)
        # silhouette (figure target)
        sil = pygame.Surface((s, int(s * 1.5)))
        sil.fill(colorkey)
        pygame.draw.circle(sil, (45, 60, 45), (s // 2, s // 4), s // 5)
        pygame.draw.polygon(sil, (45, 60, 45), [(s // 6, int(s * 1.5)), (s // 4, s // 2), (3 * s // 4, s // 2), (5 * s // 6, int(s * 1.5))])
        sprites.append(sil)
        # round bullseye
        bull = pygame.Surface((s, s))
        bull.fill(colorkey)
        for i, col in enumerate(((240, 240, 240), (20, 20, 20), (240, 240, 240), (200, 40, 40))):
            pygame.draw.circle(bull, col, (s // 2, s // 2), max(2, s // 2 - i * s // 9))
        sprites.append(bull)
        # running animal / vehicle-like blob
        deer = pygame.Surface((int(s * 1.4), s))
        deer.fill(colorkey)
        pygame.draw.ellipse(deer, (130, 90, 50), (s // 5, s // 4, s, s // 2))
        pygame.draw.circle(deer, (130, 90, 50), (int(s * 1.2), s // 4), s // 7)
        for lx in (s // 4, s // 2, int(s * 0.8), int(s * 1.05)):
            pygame.draw.line(deer, (90, 60, 35), (lx, int(s * 0.7)), (lx, s - 1), 3)
        sprites.append(deer)
        for spr in sprites:
            spr.set_colorkey(colorkey)
        self._cache[key] = sprites
        return sprites

    def _range_backdrop(self, size: tuple[int, int]) -> pygame.Surface:
        key = ("range", size)
        surf = self._cache.get(key)
        if surf is None:
            w, h = size
            surf = pygame.Surface(size)
            surf.fill((170, 190, 150))
            pygame.draw.rect(surf, (150, 140, 120), (0, 0, w, h // 5))  # backstop
            for i, y in enumerate((0.35, 0.58, 0.82)):
                pygame.draw.line(surf, (125, 140, 110), (0, int(h * y)), (w, int(h * y)), 2)
            self._cache[key] = surf
        return surf

    def _draw_moving_targets(self, screen: pygame.Surface, vp: pygame.Rect) -> None:
        screen.blit(self._range_backdrop(vp.size), vp.topleft)
        sprites = self._target_sprites(vp.h)
        lanes = ((0.22, 60.0, 0, 0.0), (0.45, -85.0, 1, 0.3), (0.62, 45.0, 2, 0.6), (0.30, -35.0, 1, 0.8))
        old_clip = screen.get_clip()
        screen.set_clip(vp.clip(screen.get_rect()))
        for lane_y, speed, idx, phase in lanes:
            spr = sprites[idx]
            span = vp.w + spr.get_width()
            pos = (phase * span + self.clock * abs(speed)) % span
            x = vp.x - spr.get_width() + pos if speed > 0 else vp.right - pos
            y = vp.y + int(lane_y * vp.h) + int(6 * math.sin(self.clock * 2.0 + phase * 6))
            screen.blit(spr, (int(x), y))
        screen.set_clip(old_clip)

    # ------------------------------------------------------------------
    # Shot reactions
    # ------------------------------------------------------------------
    def on_hole_revealed(self, x: int, y: int, now: float | None = None) -> None:
        if self.effects_level == "off":
            return
        now = time.time() if now is None else now
        if random.random() < 0.5:
            mx, my = x, y  # at the hit
        else:
            a = random.uniform(0, 2 * math.pi)
            d = random.uniform(10, 40)
            mx, my = int(x + math.cos(a) * d), int(y + math.sin(a) * d)
        self._markers.append({
            "x": mx, "y": my, "start": now,
            "dur": random.uniform(0.3, 0.6),
            "style": random.choice(("hitmarker", "explosion")),
        })
        if self.effects_level in ("marker+score", "all"):
            self.score += random.randint(1, 10) * 10
        if self.effects_level == "all":
            self._shake_start = now
            self._shake_until = now + random.uniform(0.15, 0.30)

    def draw_projected_effects(self, screen: pygame.Surface, vp: pygame.Rect, font: pygame.font.Font | None,
                               now: float | None = None) -> None:
        """Draw reactions as projected content (call before the hole overlay)."""
        if self.effects_level == "off":
            return
        now = time.time() if now is None else now
        if self._shake_until > now:
            amp = 7.0 * (self._shake_until - now) / max(1e-3, self._shake_until - self._shake_start)
            dx, dy = int(random.uniform(-amp, amp)), int(random.uniform(-amp, amp))
            area = vp.clip(screen.get_rect())
            if area.w > 0 and area.h > 0 and (dx or dy):
                screen.subsurface(area).scroll(dx, dy)
        alive = []
        for m in self._markers:
            t = (now - m["start"]) / m["dur"]
            if t >= 1.0:
                continue
            alive.append(m)
            x, y = m["x"], m["y"]
            if m["style"] == "hitmarker":
                g, L = 5, 12
                for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    pygame.draw.line(screen, (255, 255, 255), (x + sx * g, y + sy * g), (x + sx * (g + L), y + sy * (g + L)), 3)
            else:
                r = int(6 + 26 * t)
                pygame.draw.circle(screen, (255, int(200 - 120 * t), 40), (x, y), r)
                pygame.draw.circle(screen, (255, 245, 180), (x, y), max(2, int(r * 0.55 * (1.0 - t))))
        self._markers = alive
        if self.effects_level in ("marker+score", "all") and font is not None:
            surf = font.render(f"POÄNG {self.score:05d}", True, (255, 230, 90), (20, 20, 20))
            screen.blit(surf, (vp.right - surf.get_width() - 10, vp.bottom - surf.get_height() - 10))
