"""V2.23 unified training/model pipeline.

V2.23.2 adds full PRE/POST framepacks for new F2/manual GT shots, offline
V2.21.5 dense proposal expansion, a real centre-biased training sampler, a
reference-baseline fallback, and a fresh-F2 domain gate for research champion
promotion. The frozen V2.22 live hit authority is unchanged.
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
