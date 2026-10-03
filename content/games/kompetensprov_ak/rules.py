from __future__ import annotations

"""Pure rules/model helpers for Kompetensprov AK.

Authoritative BAS references used by the simulator:
- Försvarsmakten, H Skjutb AK 2021, Delmoment 12 Kompetensprov Bas.
- Försvarsmakten, H Skjutb AK 2021, 1.3.1 scoring for Helfigur 2020.
- Försvarsmakten, Reglemente Skjutning Automatkarbin 2021, AK4B/Ptr 10
  red-dot trajectory table (30 m +2 cm, 50 m +5 cm).

BAS Hv is a selectable Home Guard variant: same principle, 30 m and Pk 1.3.
The fixed 45 s cease-fire window is a simulator rule requested for this game;
BAS itself measures time from start signal to the last shot.
"""

from dataclasses import dataclass
from typing import Iterable

SIMULATOR_WINDOW_S = 45.0
COUNTED_HITS = 9
TARGET_REAL_HEIGHT_CM = 180.0
BULLET_DIAMETER_CM = 0.762  # 7.62 mm / Ptr 10

ZONE_POINTS: dict[str, int] = {
    "H": 1,
    "A": 5,
    "X": 2,
    "B": 4,
    "C": 3,
    "D": 3,
}

# Only distances used by V1 are encoded. Values are official approximate
# AK4B / Ptr 10 / red-dot trajectory heights relative to sight line.
AK4B_BANHEIGHT_CM: dict[float, float] = {
    30.0: 2.0,
    50.0: 5.0,
}


@dataclass(frozen=True, slots=True)
class CompetitionProfile:
    profile_id: str
    title: str
    subtitle: str
    distance_m: float
    min_pk: float
    source_note: str


PROFILES: tuple[CompetitionProfile, ...] = (
    CompetitionProfile(
        profile_id="bas_hv",
        title="KOMPETENSPROV BAS Hv",
        subtitle="Hemvärnsvariant • 30 m • Pk ≥ 1,30",
        distance_m=30.0,
        min_pk=1.30,
        source_note="BAS-princip + Hemvärnets 30 m / Pk 1,3-variant",
    ),
    CompetitionProfile(
        profile_id="bas",
        title="KOMPETENSPROV BAS",
        subtitle="Försvarsmakten • 50 m • Pk ≥ 1,00",
        distance_m=50.0,
        min_pk=1.00,
        source_note="H Skjutb AK 2021, Delmoment 12",
    ),
)


@dataclass(frozen=True, slots=True)
class ShotRecord:
    shot_number: int
    shot_id: int | None
    time_s: float
    raw_x: float
    raw_y: float
    scored_x: float
    scored_y: float
    norm_x: float
    norm_y: float
    zone: str | None
    points: int
    expected_posture: str
    detection_mode: str
    ballistic_cm: float
    camera_x: float | None = None
    camera_y: float | None = None
    source: str = ""
    hit_timestamp: float | None = None
    target_rect: tuple[int, int, int, int] | None = None

    @property
    def is_hit(self) -> bool:
        return self.zone in ZONE_POINTS and self.points > 0


@dataclass(frozen=True, slots=True)
class CompetitionResult:
    total_shots: int
    total_hits: int
    counted_shot_numbers: tuple[int, ...]
    counted_points: int
    last_shot_time_s: float | None
    pk: float
    passed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ManualMark:
    """Human-verified physical impact marked in the post-fire review."""
    mark_number: int
    raw_x: float
    raw_y: float
    scored_x: float
    scored_y: float
    norm_x: float
    norm_y: float
    zone: str | None
    points: int

    @property
    def is_hit(self) -> bool:
        return self.zone in ZONE_POINTS and self.points > 0


def expected_posture(shot_number: int) -> str:
    """Expected posture according to the BAS sequence.

    1-3 prone, 4-6 kneeling/sitting, 7+ standing. Improvement shots are
    therefore automatically standing, matching the rule that improvements are
    made after the reload in standing position.
    """
    n = int(shot_number)
    if n <= 3:
        return "LIGGANDE"
    if n <= 6:
        return "KNÄSTÅENDE/SITTANDE"
    return "STÅENDE"


def ak4b_banheight_cm(distance_m: float) -> float:
    """Return official table value for the supported competition distances."""
    d = float(distance_m)
    if d in AK4B_BANHEIGHT_CM:
        return float(AK4B_BANHEIGHT_CM[d])
    # Defensive interpolation if a future profile is inserted between 30/50 m.
    keys = sorted(AK4B_BANHEIGHT_CM)
    if d <= keys[0]:
        return float(AK4B_BANHEIGHT_CM[keys[0]])
    if d >= keys[-1]:
        return float(AK4B_BANHEIGHT_CM[keys[-1]])
    for lo, hi in zip(keys, keys[1:]):
        if lo <= d <= hi:
            t = (d - lo) / (hi - lo)
            return (1.0 - t) * AK4B_BANHEIGHT_CM[lo] + t * AK4B_BANHEIGHT_CM[hi]
    return 0.0


def counted_hits(shots: Iterable[ShotRecord], limit: int = COUNTED_HITS) -> tuple[ShotRecord, ...]:
    """Return the nine best hits; deterministic tie break uses earlier shot."""
    hits = [s for s in shots if s.is_hit]
    hits.sort(key=lambda s: (-int(s.points), int(s.shot_number)))
    return tuple(hits[: max(0, int(limit))])


def evaluate_result(shots: Iterable[ShotRecord], profile: CompetitionProfile) -> CompetitionResult:
    records = tuple(shots)
    hits = tuple(s for s in records if s.is_hit)
    selected = counted_hits(records)
    points = sum(int(s.points) for s in selected)
    last_time = max((float(s.time_s) for s in records), default=None)

    if last_time is None or last_time <= 0.0:
        pk = 0.0
    else:
        pk = round(points / last_time, 2)

    if len(hits) < COUNTED_HITS:
        passed = False
        reason = f"Endast {len(hits)} träffar – minst {COUNTED_HITS} krävs"
    elif pk < float(profile.min_pk):
        passed = False
        reason = f"Pk {pk:.2f} < krav {profile.min_pk:.2f}"
    else:
        passed = True
        reason = f"{len(hits)} träffar och Pk {pk:.2f}"

    return CompetitionResult(
        total_shots=len(records),
        total_hits=len(hits),
        counted_shot_numbers=tuple(s.shot_number for s in selected),
        counted_points=points,
        last_shot_time_s=last_time,
        pk=pk,
        passed=passed,
        reason=reason,
    )


def counted_manual_marks(
    marks: Iterable[ManualMark],
    limit: int = COUNTED_HITS,
) -> tuple[ManualMark, ...]:
    """Return the nine highest-scoring human-verified marks."""
    hits = [m for m in marks if m.is_hit]
    hits.sort(key=lambda m: (-int(m.points), int(m.mark_number)))
    return tuple(hits[: max(0, int(limit))])


def evaluate_manual_result(
    marks: Iterable[ManualMark],
    *,
    total_shots: int,
    last_shot_time_s: float | None,
    profile: CompetitionProfile,
) -> CompetitionResult:
    """Score manual verification without overwriting automatic detector truth.

    The denominator is still the real timestamp of the final registered shot.
    Manual review only replaces the hit/zone assessment for the displayed
    verified result.
    """
    records = tuple(marks)
    hits = tuple(m for m in records if m.is_hit)
    selected = counted_manual_marks(records)
    points = sum(int(m.points) for m in selected)

    if last_shot_time_s is None or float(last_shot_time_s) <= 0.0:
        pk = 0.0
    else:
        pk = round(points / float(last_shot_time_s), 2)

    if len(hits) < COUNTED_HITS:
        passed = False
        reason = f"Endast {len(hits)} verifierade träffar – minst {COUNTED_HITS} krävs"
    elif pk < float(profile.min_pk):
        passed = False
        reason = f"Pk {pk:.2f} < krav {profile.min_pk:.2f}"
    else:
        passed = True
        reason = f"{len(hits)} verifierade träffar och Pk {pk:.2f}"

    return CompetitionResult(
        total_shots=int(total_shots),
        total_hits=len(hits),
        counted_shot_numbers=tuple(m.mark_number for m in selected),
        counted_points=points,
        last_shot_time_s=None if last_shot_time_s is None else float(last_shot_time_s),
        pk=pk,
        passed=passed,
        reason=reason,
    )
