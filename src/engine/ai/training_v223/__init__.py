"""V2.23 unified training/model pipeline.

This package is intentionally shadow/offline first.  It captures the same
candidate groups used by the live shooting pipeline, compiles them with older
V2.16/V2.20 packs, trains candidate-ranking challengers, and maintains a
research champion registry without changing live hit authority.
"""

from .schema import SCHEMA_VERSION, FEATURE_NAMES, ShotTrainingRecord, CandidateTrainingRow
from .model import RankModelV223

__all__ = [
    "SCHEMA_VERSION",
    "FEATURE_NAMES",
    "ShotTrainingRecord",
    "CandidateTrainingRow",
    "RankModelV223",
]
