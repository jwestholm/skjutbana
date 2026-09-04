"""V2.23 unified training/model pipeline.

V2.23.6 advances the master plan from global dense-candidate ranking to a
registered-evidence direct heatmap localizer. The V2.23.2 dense pool remains a
high-recall teacher/diagnostic/fallback, but the new learner predicts spatial
NEW-hole likelihood directly over the GT-free physical evidence maps. Fresh
session discipline remains intact and live authority is unchanged.
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
