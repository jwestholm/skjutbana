"""V2.23 unified training/model pipeline.

V2.23.5 keeps the V2.23.2 dense high-recall teacher and V2.23.3 numeric
foundation, but trains on candidate-centred patches from the same registered,
photometrically compensated physical evidence maps that generated the dense
pool. Labels are tightened (<=6px positive, 6..42px neutral) and iterative
model-hard-negative mining is used before final ranking. Fresh-domain sessions
remain excluded from model selection and live authority is unchanged.
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
