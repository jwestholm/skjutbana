"""Pure arcade rules; no physical hit detection, GL, or player locomotion.

A death requires an enemy's ACTUAL fired shot, unobstructed line of sight,
and a deterministic chance roll. No global countdown can kill the player.
"""
from dataclasses import dataclass
import math
import random
from combat_tactics import difficulty_at


@dataclass
class ArcadeRun:
    alive: bool = True
    distance_score: float = 0.0
    deaths: int = 0
    killed_by: int | None = None
    shots: int = 0
    kills: int = 0
    incoming_shots: int = 0
    last_incoming_event: str = ""
    _seed: int = 1944

    def __post_init__(self):
        self._rng = random.Random(self._seed)

    def reset(self):
        self.alive = True
        self.distance_score = 0.0
        self.deaths = 0
        self.killed_by = None
        self.shots = 0
        self.kills = 0
        self.incoming_shots = 0
        self.last_incoming_event = ""
        self._rng.seed(self._seed)

    def step(self, dt, distance, threat=None):
        """Record progress only. Never kill from elapsed time."""
        if not (math.isfinite(dt) and dt >= 0 and math.isfinite(distance) and distance >= 0):
            raise ValueError('invalid simulation time or distance')
        if self.alive:
            self.distance_score = max(self.distance_score, distance)
        return False

    def player_shot(self, result):
        if not self.alive:
            return
        self.shots += 1
        if result is not None:
            self.kills += 1

    def enemy_shot(self, enemy_id, *, clear_sight, distance_metres, exposure=1.0,
                   allow_damage=True, progress_metres=0.0):
        """A fired enemy round may miss. Occluded shots can NEVER damage us.

        Deliberately arcade-like probabilities, deterministic for regression.
        These are not real-world marksmanship/ballistics.
        """
        if not self.alive:
            return False
        if not clear_sight:
            self.last_incoming_event = "blocked"
            return False
        self.incoming_shots += 1
        if not allow_damage:
            self.last_incoming_event = "warning"
            return False
        d = max(0.0, float(distance_metres))
        p = min(.42, max(.10, .31 - .007 * d)) * max(0.0, min(1.0, exposure))
        p *= difficulty_at(progress_metres).accuracy_scale
        if self._rng.random() >= p:
            self.last_incoming_event = "miss"
            return False
        self.last_incoming_event = "hit"
        self.alive = False
        self.deaths += 1
        self.killed_by = int(enemy_id)
        return True

    @property
    def score_metres(self):
        return int(self.distance_score)

    @property
    def score(self):
        return self.score_metres + 100 * self.kills

    def status(self):
        if not self.alive:
            return f'DÖD VID {self.score_metres} m | {self.kills} FIENDER | {self.shots} SKOTT | POÄNG {self.score} | R = OMSTART'
        return f'{self.score_metres} m | {self.kills} FIENDER | {self.shots} SKOTT'
