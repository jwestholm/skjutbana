"""V2.23 unified training/model pipeline.

V2.23.4 keeps the V2.23.2 full-frame/dense proposal foundation and the
V2.23.3 rich numeric evidence, but replaces the weak tabular first-stage
reducer with a learned candidate-centred PRE/POST patch model.  Fresh-domain
sessions remain excluded from model selection and live authority is unchanged.
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
