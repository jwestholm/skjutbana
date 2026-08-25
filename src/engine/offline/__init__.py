"""Offline detector replay and multi-source evidence tools (Detector V2.12)."""

from .shot_case import GroundTruth, ShotCase
from .evidence import EvidenceBundle, EvidenceConfig, EvidenceOverlay

__all__ = [
    "GroundTruth",
    "ShotCase",
    "EvidenceBundle",
    "EvidenceConfig",
    "EvidenceOverlay",
]
