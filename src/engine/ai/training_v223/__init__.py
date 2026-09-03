"""V2.23 unified training/model pipeline.

V2.23.3 keeps the V2.23.2 full-frame/dense proposal foundation and adds
GT-free rich PRE/POST evidence, a pairwise learned candidate reducer, and a
two-stage reducer→final-ranker research cascade. The frozen V2.22 live hit
authority is unchanged.
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
