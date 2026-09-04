from __future__ import annotations

"""
Pop the Balloon - two-player shooting-range game.

Drop-in game module for jwestholm/skjutbana (dev branch).

Integration:
    create_game(game_root, viewport) -> PopTheBalloonGame

Input:
    Uses src.engine.input.hit_input.hit_input, so simulated mouse hits and
    camera hits travel through the same path.  OverlayScene already converts
    left mouse clicks into HitEvents.

Rules implemented:
- Player 1 always shoots, then Player 2. No alternating starting player.
- 30 blue + 30 red balloons exist from the start. Popped = gone permanently.
- One shot opportunity per player per level, max 30 levels.
- Every turn: PLAYER X -> 3 -> 2 -> 1 -> NU!, then the 20 second timer starts.
- World is frozen during result overlays and countdowns.
- 3 misses/timeouts = elimination, but both players always finish the level.
- If P1 reaches zero balloons, P2 still gets the turn in that same level.
- If both reach zero on the same level, score decides (tie remains tie).
- If both survive 30 levels, score decides.
- Hits pass through overlapping yellow power-ups and affect balloons behind.
- Hitting at least one own balloon makes the direct shot valid even if an
  overlapping opponent balloon is also popped.
- Hitting only opponent balloon(s) costs one miss.
- Dynamite can pop both colors; if it destroys any opponent balloon it costs
  one miss maximum for that shot.
"""

from dataclasses import dataclass
import math
import random
import time
from typing import Optional

import pygame

from src.engine.input.hit_input import HitEvent, hit_input


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

TOTAL_BALLOONS_PER_PLAYER = 30
MAX_LEVELS = 30
MAX_MISSES = 3
SHOT_TIME_SECONDS = 20.0
COUNTDOWN_STEP_SECONDS = 0.58
PLAYER_INTRO_SECONDS = 0.72
NOW_SECONDS = 0.52
RESULT_SECONDS = 1.35
FREEZE_SECONDS = 4.0

MAX_POWERUPS_ON_FIELD = 5
NORMAL_BALLOON_RADIUS = 24.0
POWERUP_RADIUS = 23.0

BASE_SPEED_MIN = 28.0
BASE_SPEED_MAX = 46.0
LEVEL_SPEED_GAIN = 0.018  # +1.8 % / level, while timer stays fixed at 20 sec.

LIGHTNING_RADIUS = 155.0
DYNAMITE_RADIUS = 175.0

SIZE_STEP = 0.10
SIZE_MIN = 0.60
SIZE_MAX = 1.40
SPEED_STEP = 0.10
SPEED_MIN = 0.60
SPEED_MAX = 1.60
CHAOS_STEP = 0.35
CHAOS_MIN = 1.0
CHAOS_MAX = 3.0

TIME_BONUS_PER_SECOND = 5


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

BG = (14, 18, 25)
GRID = (30, 38, 50)
PANEL = (19, 25, 35)
PANEL_2 = (28, 36, 49)
TEXT = (244, 247, 251)
MUTED = (160, 171, 189)
P1_COLOR = (57, 151, 255)
P2_COLOR = (255, 76, 96)
YELLOW = (255, 214, 64)
GOOD = (95, 220, 137)
BAD = (255, 65, 58)
ORANGE = (255, 180, 65)
ICE = (115, 215, 255)
BLACK = (18, 18, 20)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class PlayerMods:
    size: float = 1.0
    speed: float = 1.0
    chaos: float = 1.0
    direction: Optional[str] = None  # left/right/up/down
    freeze_armed: bool = False


@dataclass
class Balloon:
    kind: str  # "player" or "power"
    x: float
    y: float
    vx: float
    vy: float
    base_radius: float
    owner: int = 0
    power: str = ""
    alive: bool = True
    chaos_phase: float = 0.0
    chaos_switch_at: float = 0.0

    def radius(self, mods: dict[int, PlayerMods]) -> float:
        if self.kind == "player":
            return self.base_radius * mods[self.owner].size
        return self.base_radius


@dataclass
class OverlayMessage:
    title: str
    subtitle: str = ""
    color: tuple[int, int, int] = TEXT


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class PopTheBalloonGame:
    def __init__(self, game_root: str, viewport: pygame.Rect) -> None:
        del game_root  # No external assets required.
        self.viewport = viewport.copy()

        self.font_huge: pygame.font.Font | None = None
        self.font_big: pygame.font.Font | None = None
        self.font_medium: pygame.font.Font | None = None
        self.font_small: pygame.font.Font | None = None

        self.rng = random.Random()

        self.balloons: list[Balloon] = []
        self.mods = {1: PlayerMods(), 2: PlayerMods()}
        self.score = {1: 0, 2: 0}
        self.misses = {1: 0, 2: 0}
        self.remaining = {1: TOTAL_BALLOONS_PER_PLAYER, 2: TOTAL_BALLOONS_PER_PLAYER}

        self.level = 1
        self.player = 1

        # phase: intro -> count3 -> count2 -> count1 -> now -> live -> result -> game_over
        self.phase = "intro"
        self.phase_until = 0.0
        self.turn_started_at = 0.0
        self.freeze_until = 0.0
        self.overlay = OverlayMessage("")

        self.game_over_title = ""
        self.game_over_reason = ""

        self._subscribed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        self.font_huge = pygame.font.Font(None, 118)
        self.font_big = pygame.font.Font(None, 62)
        self.font_medium = pygame.font.Font(None, 34)
        self.font_small = pygame.font.Font(None, 24)

        self._spawn_initial_balloons()
        self._subscribe()
        self._start_turn(player=1)

    def on_exit(self) -> None:
        self._unsubscribe()

    def _subscribe(self) -> None:
        if not self._subscribed:
            hit_input.subscribe(self._on_hit)
            self._subscribed = True

    def _unsubscribe(self) -> None:
        if self._subscribed:
            hit_input.unsubscribe(self._on_hit)
            self._subscribed = False

    # ------------------------------------------------------------------
    # Setup / spawn
    # ------------------------------------------------------------------

    def _spawn_initial_balloons(self) -> None:
        self.balloons.clear()
        for owner in (1, 2):
            for _ in range(TOTAL_BALLOONS_PER_PLAYER):
                self.balloons.append(self._make_player_balloon(owner))

    def _safe_spawn_xy(self, radius: float) -> tuple[float, float]:
        # Keep HUD area free. The GameScene viewport is already clipped.
        margin = radius + 8
        top = 152 + radius
        left = margin
        right = max(left + 1, self.viewport.w - margin)
        bottom = max(top + 1, self.viewport.h - margin)

        # Try to avoid exact pile-ups, but overlap is allowed and desired.
        for _ in range(50):
            x = self.rng.uniform(left, right)
            y = self.rng.uniform(top, bottom)
            if not any(
                b.alive
                and math.hypot(b.x - x, b.y - y) < (b.base_radius + radius) * 0.55
                for b in self.balloons
            ):
                return x, y
        return self.rng.uniform(left, right), self.rng.uniform(top, bottom)

    def _random_velocity(self) -> tuple[float, float]:
        speed = self.rng.uniform(BASE_SPEED_MIN, BASE_SPEED_MAX)
        angle = self.rng.uniform(0.0, math.tau)
        return math.cos(angle) * speed, math.sin(angle) * speed

    def _make_player_balloon(self, owner: int) -> Balloon:
        x, y = self._safe_spawn_xy(NORMAL_BALLOON_RADIUS)
        vx, vy = self._random_velocity()
        now = time.monotonic()
        return Balloon(
            kind="player",
            owner=owner,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            base_radius=NORMAL_BALLOON_RADIUS,
            chaos_phase=self.rng.uniform(0.0, math.tau),
            chaos_switch_at=now + self.rng.uniform(0.45, 1.2),
        )

    def _powerup_count(self) -> int:
        return sum(1 for b in self.balloons if b.alive and b.kind == "power")

    def _spawn_powerup_for_turn(self) -> None:
        if self._powerup_count() >= MAX_POWERUPS_ON_FIELD:
            return

        # Weighted enough to keep point balloons visible but leave room for gameplay powers.
        choices = [
            "score100", "score100",
            "score250",
            "score500",
            "big",
            "small",
            "slow",
            "fast",
            "dir_left",
            "dir_right",
            "dir_up",
            "dir_down",
            "chaos",
            "freeze",
            "lightning",
            "bomb",
            "random",
        ]
        power = self.rng.choice(choices)
        x, y = self._safe_spawn_xy(POWERUP_RADIUS)
        vx, vy = self._random_velocity()
        now = time.monotonic()
        self.balloons.append(
            Balloon(
                kind="power",
                power=power,
                x=x,
                y=y,
                vx=vx * 0.85,
                vy=vy * 0.85,
                base_radius=POWERUP_RADIUS,
                chaos_phase=self.rng.uniform(0.0, math.tau),
                chaos_switch_at=now + self.rng.uniform(0.8, 1.5),
            )
        )

    # ------------------------------------------------------------------
    # Turn flow
    # ------------------------------------------------------------------

    def _start_turn(self, player: int) -> None:
        self.player = player
        self._spawn_powerup_for_turn()
        self.phase = "intro"
        self.phase_until = time.monotonic() + PLAYER_INTRO_SECONDS
        self.overlay = OverlayMessage(f"SPELARE {player}", "", self._player_color(player))

    def _advance_countdown(self, now: float) -> None:
        if self.phase == "intro":
            self.phase = "count3"
            self.phase_until = now + COUNTDOWN_STEP_SECONDS
            self.overlay = OverlayMessage("3", "", self._player_color(self.player))
        elif self.phase == "count3":
            self.phase = "count2"
            self.phase_until = now + COUNTDOWN_STEP_SECONDS
            self.overlay = OverlayMessage("2", "", self._player_color(self.player))
        elif self.phase == "count2":
            self.phase = "count1"
            self.phase_until = now + COUNTDOWN_STEP_SECONDS
            self.overlay = OverlayMessage("1", "", self._player_color(self.player))
        elif self.phase == "count1":
            self.phase = "now"
            self.phase_until = now + NOW_SECONDS
            self.overlay = OverlayMessage("NU!", "", self._player_color(self.player))
        elif self.phase == "now":
            self.phase = "live"
            self.turn_started_at = now
            self.overlay = OverlayMessage("")

            if self.mods[self.player].freeze_armed:
                self.mods[self.player].freeze_armed = False
                self.freeze_until = now + FREEZE_SECONDS
            else:
                self.freeze_until = 0.0

    def _finish_turn_with_result(self, title: str, subtitle: str, color=TEXT) -> None:
        self.phase = "result"
        self.phase_until = time.monotonic() + RESULT_SECONDS
        self.overlay = OverlayMessage(title, subtitle, color)

    def _after_result(self) -> None:
        if self.player == 1:
            self._start_turn(2)
            return

        # Both players have now completed this level.
        if self._resolve_end_conditions():
            return

        self.level += 1
        self._start_turn(1)

    def _resolve_end_conditions(self) -> bool:
        p1_done = self.remaining[1] <= 0
        p2_done = self.remaining[2] <= 0
        p1_out = self.misses[1] >= MAX_MISSES
        p2_out = self.misses[2] >= MAX_MISSES

        # Balloon objective is primary.
        if p1_done or p2_done:
            if p1_done and p2_done:
                self._end_by_score("Båda poppade sin sista ballong på samma nivå.")
            elif p1_done:
                self._end_game("SPELARE 1 VINNER!", "Alla 30 blå ballonger är poppade.")
            else:
                self._end_game("SPELARE 2 VINNER!", "Alla 30 röda ballonger är poppade.")
            return True

        if p1_out or p2_out:
            if p1_out and p2_out:
                self._end_by_score("Båda nådde 3 missar på samma nivå.")
            elif p1_out:
                self._end_game("SPELARE 2 VINNER!", "Spelare 1 fick 3 missar.")
            else:
                self._end_game("SPELARE 1 VINNER!", "Spelare 2 fick 3 missar.")
            return True

        if self.level >= MAX_LEVELS:
            self._end_by_score("30 nivåer spelade.")
            return True

        return False

    def _end_by_score(self, reason: str) -> None:
        if self.score[1] > self.score[2]:
            self._end_game("SPELARE 1 VINNER!", reason + " Högst poäng avgör.")
        elif self.score[2] > self.score[1]:
            self._end_game("SPELARE 2 VINNER!", reason + " Högst poäng avgör.")
        else:
            self._end_game("OAVGJORT!", reason + " Poängen är också lika.")

    def _end_game(self, title: str, reason: str) -> None:
        self.phase = "game_over"
        self.game_over_title = title
        self.game_over_reason = reason
        self.overlay = OverlayMessage("")

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event):
        # OverlayScene already converts left mouse clicks to HitEvents.
        # R is intentionally useful while prototyping on a PC.
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self._reset_game()
        return None

    def _reset_game(self) -> None:
        self.mods = {1: PlayerMods(), 2: PlayerMods()}
        self.score = {1: 0, 2: 0}
        self.misses = {1: 0, 2: 0}
        self.remaining = {1: TOTAL_BALLOONS_PER_PLAYER, 2: TOTAL_BALLOONS_PER_PLAYER}
        self.level = 1
        self.player = 1
        self.freeze_until = 0.0
        self.game_over_title = ""
        self.game_over_reason = ""
        self._spawn_initial_balloons()
        self._start_turn(1)

    def _on_hit(self, hit: HitEvent) -> None:
        if self.phase != "live":
            return

        # HitEvent.game_x/game_y are viewport-local in the current engine.
        x = float(hit.game_x)
        y = float(hit.game_y)

        # One accepted hit = the player's one shot.
        self.phase = "processing"

        hit_balloons = [
            b for b in self.balloons
            if b.alive and self._contains(b, x, y)
        ]

        # Important: yellow balloons do NOT block the shot.
        power_hits = [b for b in hit_balloons if b.kind == "power"]
        normal_hits = [b for b in hit_balloons if b.kind == "player"]

        power_messages: list[str] = []
        for power_balloon in power_hits:
            power_messages.append(self._apply_powerup(power_balloon))

        # Powerups may have popped normal balloons, so filter again.
        normal_hits = [b for b in normal_hits if b.alive]
        own_hits = [b for b in normal_hits if b.owner == self.player]
        enemy_hits = [b for b in normal_hits if b.owner != self.player]

        if own_hits:
            for b in own_hits:
                self._pop_player_balloon(b, score_owner=self.player)
            # Direct overlap with the opponent is NOT a miss if at least one
            # own balloon was hit by the shot.
            for b in enemy_hits:
                self._pop_player_balloon(b, score_owner=0)

            time_left = max(0.0, SHOT_TIME_SECONDS - (time.monotonic() - self.turn_started_at))
            bonus = int(time_left) * TIME_BONUS_PER_SECOND
            self.score[self.player] += bonus

            details = [f"{len(own_hits)} egna", f"tidsbonus +{bonus}"]
            if enemy_hits:
                details.append(f"{len(enemy_hits)} motståndarballong råkade följa med")
            if power_messages:
                details.extend(power_messages)

            count = len(own_hits) + len(enemy_hits) + len(power_hits)
            title = f"POP x{count}!" if count > 1 else "POP!"
            self._finish_turn_with_result(title, " • ".join(details), GOOD)
            return

        # Hitting one or more powerups is a valid shot, even with no own balloon.
        if power_hits:
            self._finish_turn_with_result(
                "POWER UP!",
                " • ".join(power_messages),
                YELLOW,
            )
            return

        if enemy_hits:
            for b in enemy_hits:
                self._pop_player_balloon(b, score_owner=0)
            self._add_miss()
            self._finish_turn_with_result(
                "FEL FÄRG!",
                f"{len(enemy_hits)} motståndarballong poppad • MISS {self.misses[self.player]}/{MAX_MISSES}",
                BAD,
            )
            return

        self._add_miss()
        self._finish_turn_with_result(
            "MISS!",
            f"MISS {self.misses[self.player]}/{MAX_MISSES}",
            BAD,
        )

    def _contains(self, balloon: Balloon, x: float, y: float) -> bool:
        r = balloon.radius(self.mods)
        # Slightly taller than wide: balloon-like hitbox.
        rx = r
        ry = r * 1.22
        dx = (x - balloon.x) / max(1.0, rx)
        dy = (y - balloon.y) / max(1.0, ry)
        return dx * dx + dy * dy <= 1.0

    def _add_miss(self) -> None:
        self.misses[self.player] = min(MAX_MISSES, self.misses[self.player] + 1)

    # ------------------------------------------------------------------
    # Powerups
    # ------------------------------------------------------------------

    def _apply_powerup(self, b: Balloon) -> str:
        power = b.power
        px, py = b.x, b.y
        b.alive = False

        me = self.player
        other = 2 if me == 1 else 1

        if power == "score100":
            self.score[me] += 100
            return "+100 POÄNG"

        if power == "score250":
            self.score[me] += 250
            return "+250 POÄNG"

        if power == "score500":
            self.score[me] += 500
            return "+500 POÄNG"

        if power == "big":
            self.mods[me].size = min(SIZE_MAX, self.mods[me].size + SIZE_STEP)
            return "BIG +1"

        if power == "small":
            self.mods[other].size = max(SIZE_MIN, self.mods[other].size - SIZE_STEP)
            return "MOTSTÅNDAREN MINDRE"

        if power == "slow":
            self.mods[me].speed = max(SPEED_MIN, self.mods[me].speed - SPEED_STEP)
            return "DINA LÅNGSAMMARE"

        if power == "fast":
            self.mods[other].speed = min(SPEED_MAX, self.mods[other].speed + SPEED_STEP)
            return "MOTSTÅNDAREN SNABBARE"

        if power.startswith("dir_"):
            direction = power.split("_", 1)[1]
            self.mods[me].direction = direction
            label = {
                "left": "← VÄNSTER",
                "right": "→ HÖGER",
                "up": "↑ UPP",
                "down": "↓ NER",
            }[direction]
            return f"RIKTNING {label}"

        if power == "chaos":
            self.mods[other].chaos = min(CHAOS_MAX, self.mods[other].chaos + CHAOS_STEP)
            return "MOTSTÅNDAREN: CHAOS"

        if power == "freeze":
            self.mods[me].freeze_armed = True
            return "FREEZE READY"

        if power == "lightning":
            targets = self._nearby_player_balloons(px, py, me, LIGHTNING_RADIUS)
            for target in targets:
                self._pop_player_balloon(target, score_owner=me)
            return f"BLIXT: {len(targets)} EGNA POP"

        if power == "bomb":
            targets = [
                x for x in self.balloons
                if x.alive
                and x.kind == "player"
                and math.hypot(x.x - px, x.y - py) <= DYNAMITE_RADIUS
            ]
            own = 0
            enemy = 0
            for target in targets:
                if target.owner == me:
                    own += 1
                    self._pop_player_balloon(target, score_owner=me)
                else:
                    enemy += 1
                    self._pop_player_balloon(target, score_owner=0)

            if enemy:
                # Max one life/miss penalty for the entire explosion.
                self._add_miss()
            return f"DYNAMIT: {own} EGNA / {enemy} MOTSTÅNDARE" + (" / 1 MISS" if enemy else "")

        if power == "random":
            return self._random_power_effect(me, other)

        return power.upper()

    def _random_power_effect(self, me: int, other: int) -> str:
        outcomes = [
            "bigger",
            "slower",
            "enemy_fast",
            "enemy_small",
            "heal_miss",
            "take_miss",
            "mega_pop",
            "chaos_self",
            "reset_bad",
        ]
        result = self.rng.choice(outcomes)

        if result == "bigger":
            self.mods[me].size = min(SIZE_MAX, self.mods[me].size + 0.20)
            return "RANDOM: JACKPOT +20% STORLEK"

        if result == "slower":
            self.mods[me].speed = max(SPEED_MIN, self.mods[me].speed - 0.20)
            return "RANDOM: -20% FART"

        if result == "enemy_fast":
            self.mods[other].speed = min(SPEED_MAX, self.mods[other].speed + 0.20)
            return "RANDOM: MOTSTÅNDAREN +20% FART"

        if result == "enemy_small":
            self.mods[other].size = max(SIZE_MIN, self.mods[other].size - 0.20)
            return "RANDOM: MOTSTÅNDAREN -20% STORLEK"

        if result == "heal_miss":
            self.misses[me] = max(0, self.misses[me] - 1)
            return "RANDOM: EN MISS ÅTERSTÄLLD"

        if result == "take_miss":
            self._add_miss()
            return "RANDOM: BACKFIRE - 1 MISS"

        if result == "mega_pop":
            own = [b for b in self.balloons if b.alive and b.kind == "player" and b.owner == me]
            self.rng.shuffle(own)
            own = own[: min(5, len(own))]
            for target in own:
                self._pop_player_balloon(target, score_owner=me)
            return f"RANDOM: MEGA POP x{len(own)}"

        if result == "chaos_self":
            self.mods[me].chaos = min(CHAOS_MAX, self.mods[me].chaos + 0.70)
            return "RANDOM: DINA BALLONGER CHAOS"

        # reset_bad: deliberately useful comeback result.
        self.mods[me].speed = min(self.mods[me].speed, 1.0)
        self.mods[me].size = max(self.mods[me].size, 1.0)
        self.mods[me].chaos = 1.0
        return "RANDOM: NEGATIVA MODS RESET"

    def _nearby_player_balloons(
        self,
        x: float,
        y: float,
        owner: int,
        radius: float,
    ) -> list[Balloon]:
        return [
            b for b in self.balloons
            if b.alive
            and b.kind == "player"
            and b.owner == owner
            and math.hypot(b.x - x, b.y - y) <= radius
        ]

    def _pop_player_balloon(self, b: Balloon, score_owner: int) -> None:
        if not b.alive or b.kind != "player":
            return
        b.alive = False
        self.remaining[b.owner] = max(0, self.remaining[b.owner] - 1)
        if score_owner == b.owner:
            self.score[score_owner] += 100

    # ------------------------------------------------------------------
    # Update / movement
    # ------------------------------------------------------------------

    def update(self, dt: float):
        now = time.monotonic()

        if self.phase in ("intro", "count3", "count2", "count1", "now") and now >= self.phase_until:
            self._advance_countdown(now)
            return None

        if self.phase == "result" and now >= self.phase_until:
            self._after_result()
            return None

        if self.phase == "live":
            elapsed = now - self.turn_started_at
            if elapsed >= SHOT_TIME_SECONDS:
                self.phase = "processing"
                self._add_miss()
                self._finish_turn_with_result(
                    "TIME OUT!",
                    f"MISS {self.misses[self.player]}/{MAX_MISSES}",
                    BAD,
                )
                return None

            self._update_balloons(dt, now)

        # During countdown/result/game-over the world intentionally does not move.
        return None

    def _update_balloons(self, dt: float, now: float) -> None:
        level_mul = 1.0 + (self.level - 1) * LEVEL_SPEED_GAIN

        for b in self.balloons:
            if not b.alive:
                continue

            if now < self.freeze_until:
                # Freeze means the whole playfield is frozen for the player who
                # earned it. It begins at NU!, while the 20 s timer keeps running.
                continue

            speed_mul = level_mul
            if b.kind == "player":
                mod = self.mods[b.owner]
                speed_mul *= mod.speed

                # Clear, persistent direction power.
                push = 36.0 * dt
                if mod.direction == "right":
                    b.vx += push
                elif mod.direction == "left":
                    b.vx -= push
                elif mod.direction == "up":
                    b.vy -= push
                elif mod.direction == "down":
                    b.vy += push

                # CHAOS: looping orbital force + frequent direction changes.
                if mod.chaos > 1.0:
                    b.chaos_phase += dt * (2.1 + mod.chaos * 1.0)
                    loop_force = (mod.chaos - 1.0) * 58.0
                    b.vx += math.cos(b.chaos_phase) * loop_force * dt
                    b.vy += math.sin(b.chaos_phase) * loop_force * dt

                    if now >= b.chaos_switch_at:
                        turn = (mod.chaos - 1.0) * 38.0
                        b.vx += self.rng.uniform(-turn, turn)
                        b.vy += self.rng.uniform(-turn, turn)

                        # Sometimes make a very visible hard turn.
                        if self.rng.random() < 0.42:
                            old_vx = b.vx
                            b.vx = -b.vy
                            b.vy = old_vx * self.rng.choice((-1.0, 1.0))

                        b.chaos_switch_at = now + self.rng.uniform(0.22, 0.60) / mod.chaos

            else:
                speed_mul *= 0.90

            max_speed = 78.0
            b.vx = max(-max_speed, min(max_speed, b.vx))
            b.vy = max(-max_speed, min(max_speed, b.vy))

            b.x += b.vx * speed_mul * dt
            b.y += b.vy * speed_mul * dt

            self._bounce_inside_playfield(b)

    def _bounce_inside_playfield(self, b: Balloon) -> None:
        r = b.radius(self.mods)
        top = 152 + r
        left = r
        right = self.viewport.w - r
        bottom = self.viewport.h - r

        if b.x <= left:
            b.x = left
            b.vx = abs(b.vx)
        elif b.x >= right:
            b.x = right
            b.vx = -abs(b.vx)

        if b.y <= top:
            b.y = top
            b.vy = abs(b.vy)
        elif b.y >= bottom:
            b.y = bottom
            b.vy = -abs(b.vy)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, screen: pygame.Surface) -> None:
        # The screen is clipped to viewport by GameScene, but coordinates here
        # are viewport-local. Render onto a local surface to keep that contract simple.
        surface = pygame.Surface((self.viewport.w, self.viewport.h))
        surface.fill(BG)

        self._draw_grid(surface)
        self._draw_balloons(surface)
        self._draw_hud(surface)

        if self.phase in ("intro", "count3", "count2", "count1", "now", "result"):
            self._draw_center_overlay(surface)

        if self.phase == "game_over":
            self._draw_game_over(surface)

        screen.blit(surface, self.viewport.topleft)

    def _draw_grid(self, surface: pygame.Surface) -> None:
        step = 42
        for x in range(0, surface.get_width(), step):
            pygame.draw.line(surface, GRID, (x, 0), (x, surface.get_height()), 1)
        for y in range(0, surface.get_height(), step):
            pygame.draw.line(surface, GRID, (0, y), (surface.get_width(), y), 1)

    def _draw_balloons(self, surface: pygame.Surface) -> None:
        for b in self.balloons:
            if not b.alive:
                continue
            self._draw_balloon(surface, b)

    def _draw_balloon(self, surface: pygame.Surface, b: Balloon) -> None:
        r = b.radius(self.mods)
        rx = max(7, int(r))
        ry = max(9, int(r * 1.22))
        rect = pygame.Rect(int(b.x - rx), int(b.y - ry), rx * 2, ry * 2)

        if b.kind == "player":
            color = self._player_color(b.owner)
            pygame.draw.ellipse(surface, color, rect)
            pygame.draw.ellipse(surface, (235, 240, 250), rect, 2)

            # highlight
            highlight = pygame.Rect(
                int(b.x - rx * 0.45),
                int(b.y - ry * 0.55),
                max(3, int(rx * 0.35)),
                max(5, int(ry * 0.45)),
            )
            pygame.draw.ellipse(surface, (255, 255, 255), highlight)

            knot = [
                (int(b.x - 5), int(b.y + ry - 1)),
                (int(b.x + 5), int(b.y + ry - 1)),
                (int(b.x), int(b.y + ry + 8)),
            ]
            pygame.draw.polygon(surface, color, knot)
            return

        # Power balloon
        pygame.draw.ellipse(surface, YELLOW, rect)
        pygame.draw.ellipse(surface, (255, 255, 240), rect, 2)
        label = self._power_label(b.power)
        font = self.font_small if len(label) <= 3 else pygame.font.Font(None, 18)
        txt = font.render(label, True, BLACK)
        surface.blit(txt, txt.get_rect(center=(int(b.x), int(b.y))))

    def _power_label(self, power: str) -> str:
        return {
            "score100": "+100",
            "score250": "+250",
            "score500": "+500",
            "big": "+",
            "small": "−",
            "slow": "SLOW",
            "fast": "FAST",
            "dir_left": "←",
            "dir_right": "→",
            "dir_up": "↑",
            "dir_down": "↓",
            "chaos": "C",
            "freeze": "ICE",
            "lightning": "⚡",
            "bomb": "B",
            "random": "?",
        }.get(power, "?")

    def _draw_hud(self, surface: pygame.Surface) -> None:
        hud_h = 142
        pygame.draw.rect(surface, PANEL, (0, 0, surface.get_width(), hud_h))

        third = surface.get_width() // 3
        self._draw_player_panel(surface, 1, pygame.Rect(8, 8, third - 16, hud_h - 16))
        self._draw_center_panel(surface, pygame.Rect(third + 5, 8, third - 10, hud_h - 16))
        self._draw_player_panel(
            surface,
            2,
            pygame.Rect(third * 2 + 8, 8, surface.get_width() - third * 2 - 16, hud_h - 16),
        )

    def _draw_player_panel(self, surface: pygame.Surface, player: int, rect: pygame.Rect) -> None:
        active = player == self.player and self.phase != "game_over"
        pygame.draw.rect(surface, PANEL_2, rect, border_radius=10)
        if active:
            pygame.draw.rect(surface, self._player_color(player), rect, 3, border_radius=10)

        x = rect.x + 12
        y = rect.y + 8

        name = self.font_medium.render(f"SPELARE {player}", True, self._player_color(player))
        surface.blit(name, (x, y))
        y += 31

        score = self.font_medium.render(f"{self.score[player]} p", True, TEXT)
        surface.blit(score, (x, y))
        y += 28

        rem = self.font_small.render(
            f"Ballonger: {self.remaining[player]}/{TOTAL_BALLOONS_PER_PLAYER}",
            True,
            TEXT,
        )
        surface.blit(rem, (x, y))
        y += 22

        miss_icons = "X" * self.misses[player] + "O" * (MAX_MISSES - self.misses[player])
        miss = self.font_small.render(f"Missar: {miss_icons}", True, BAD if self.misses[player] else MUTED)
        surface.blit(miss, (x, y))
        y += 21

        mods = self.mods[player]
        direction = {
            None: "-",
            "left": "←",
            "right": "→",
            "up": "↑",
            "down": "↓",
        }[mods.direction]
        size_lvl = round((mods.size - 1.0) / SIZE_STEP)
        speed_lvl = round((mods.speed - 1.0) / SPEED_STEP)
        chaos_lvl = max(0, round((mods.chaos - 1.0) / CHAOS_STEP))
        mod_line = f"Storlek {size_lvl:+d}  Fart {speed_lvl:+d}  Dir {direction}  Kaos {chaos_lvl}"
        mod_surf = pygame.font.Font(None, 19).render(mod_line, True, MUTED)
        surface.blit(mod_surf, (x, y))

        if mods.freeze_armed:
            ice = pygame.font.Font(None, 19).render("ICE READY", True, ICE)
            surface.blit(ice, (rect.right - ice.get_width() - 10, rect.bottom - 20))

    def _draw_center_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, (34, 43, 57), rect, border_radius=10)

        title = self.font_medium.render(
            f"SPELARE {self.player} SKJUTER" if self.phase == "live" else f"SPELARE {self.player}",
            True,
            self._player_color(self.player),
        )
        surface.blit(title, title.get_rect(center=(rect.centerx, rect.y + 28)))

        level = self.font_small.render(f"NIVÅ {self.level}/{MAX_LEVELS}", True, YELLOW)
        surface.blit(level, level.get_rect(center=(rect.centerx, rect.y + 58)))

        if self.phase == "live":
            elapsed = time.monotonic() - self.turn_started_at
            left = max(0.0, SHOT_TIME_SECONDS - elapsed)
            timer_color = BAD if left <= 5 else ORANGE if left <= 10 else TEXT
            timer = self.font_big.render(str(int(math.ceil(left))), True, timer_color)
            surface.blit(timer, timer.get_rect(center=(rect.centerx, rect.y + 96)))
        else:
            timer = self.font_big.render("20", True, MUTED)
            surface.blit(timer, timer.get_rect(center=(rect.centerx, rect.y + 96)))

    def _draw_center_overlay(self, surface: pygame.Surface) -> None:
        if not self.overlay.title:
            return

        shade = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 78))
        surface.blit(shade, (0, 0))

        center_y = max(230, surface.get_height() // 2)
        title = self.font_huge.render(self.overlay.title, True, self.overlay.color)
        surface.blit(title, title.get_rect(center=(surface.get_width() // 2, center_y)))

        if self.overlay.subtitle:
            # Basic wrapping for long power-up/result descriptions.
            self._draw_centered_wrapped(
                surface,
                self.overlay.subtitle,
                y=center_y + 76,
                max_width=int(surface.get_width() * 0.84),
                color=TEXT,
            )

    def _draw_centered_wrapped(
        self,
        surface: pygame.Surface,
        text: str,
        y: int,
        max_width: int,
        color,
    ) -> None:
        words = text.split()
        lines: list[str] = []
        line = ""
        for word in words:
            candidate = (line + " " + word).strip()
            if self.font_medium.size(candidate)[0] <= max_width:
                line = candidate
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)

        for i, line in enumerate(lines[:3]):
            rendered = self.font_medium.render(line, True, color)
            surface.blit(
                rendered,
                rendered.get_rect(center=(surface.get_width() // 2, y + i * 34)),
            )

    def _draw_game_over(self, surface: pygame.Surface) -> None:
        shade = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 220))
        surface.blit(shade, (0, 0))

        y = surface.get_height() // 2 - 70
        title = self.font_huge.render(self.game_over_title, True, YELLOW)
        surface.blit(title, title.get_rect(center=(surface.get_width() // 2, y)))

        reason = self.font_medium.render(self.game_over_reason, True, TEXT)
        surface.blit(reason, reason.get_rect(center=(surface.get_width() // 2, y + 70)))

        score = self.font_big.render(
            f"{self.score[1]}  -  {self.score[2]}",
            True,
            TEXT,
        )
        surface.blit(score, score.get_rect(center=(surface.get_width() // 2, y + 130)))

        hint = self.font_small.render("R = spela igen    ESC = tillbaka", True, MUTED)
        surface.blit(hint, hint.get_rect(center=(surface.get_width() // 2, y + 185)))

    def _player_color(self, player: int) -> tuple[int, int, int]:
        return P1_COLOR if player == 1 else P2_COLOR


def create_game(game_root: str, viewport: pygame.Rect):
    return PopTheBalloonGame(game_root=game_root, viewport=viewport)
