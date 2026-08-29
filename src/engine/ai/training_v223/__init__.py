"""V2.23 unified training/model pipeline.

V2.23.1 keeps the pipeline shadow/offline first, loads canonical V2.16/V2.20
candidate packs, captures V2.8 high-recall hypothesis pools for new F2/manual
shots, and support-gates research champion promotion. Live hit authority is
unchanged.
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
