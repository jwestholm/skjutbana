"""V2.22 fast shot resolver.

This module is intentionally independent from pygame/OpenCV.  It fuses evidence
that has already been produced by the camera detector, SimpleAIMemory and future
experts (for example the physical dense ranker) and selects one *existing*
candidate position.  It never interpolates between two different holes.

The resolver is designed for the synchronous emission path, so its work is
bounded and uses a small spatial hash for candidate clustering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Point = Tuple[float, float]
Candidate = Dict[str, Any]

SCHEMA_VERSION = "2.22"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if not math.isfinite(result):
            return default
        return result
    except Exception:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = _clamp01(pct) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


@dataclass
class ResolverObservation:
    camera_x: float
    camera_y: float
    source: str
    source_score: Optional[float] = None
    detector_score: Optional[float] = None
    simple_ai_score: Optional[float] = None
    persistence: Optional[float] = None
    existed_before: Optional[float] = None
    source_support: Optional[float] = None
    expert_weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def point(self) -> Point:
        return (self.camera_x, self.camera_y)


@dataclass
class ResolverCluster:
    cluster_id: int
    observations: List[ResolverObservation] = field(default_factory=list)
    camera_x: float = 0.0
    camera_y: float = 0.0
    score: float = 0.0
    game_prior: float = 0.0
    source_names: Tuple[str, ...] = ()
    evidence: Dict[str, float] = field(default_factory=dict)


@dataclass
class ResolverDecision:
    shot_id: Optional[int]
    camera_x: float
    camera_y: float
    confidence: float
    confidence_calibrated: bool
    score: float
    margin: float
    cluster_id: int
    cluster_count: int
    resolver_ms: float
    reason: str
    source_names: Tuple[str, ...]
    evidence: Dict[str, float]
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "shot_id": self.shot_id,
            "camera_x": float(self.camera_x),
            "camera_y": float(self.camera_y),
            "confidence": float(self.confidence),
            "confidence_calibrated": bool(self.confidence_calibrated),
            "score": float(self.score),
            "margin": float(self.margin),
            "cluster_id": int(self.cluster_id),
            "cluster_count": int(self.cluster_count),
            "resolver_ms": float(self.resolver_ms),
            "reason": str(self.reason),
            "source_names": list(self.source_names),
            "evidence": dict(self.evidence),
            "alternatives": list(self.alternatives),
        }


class ShotResolverV222:
    """Fast, discrete evidence fusion for one audio shot.

    The resolver deliberately does not own any image model.  Heavy or expensive
    experts should run elsewhere (ideally in parallel after the audio peak) and
    publish a short list of scored camera-space votes before emission.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "cluster_radius_px": 18.0,
        "max_camera_candidates": 256,
        "max_ranked_candidates": 256,
        "max_external_votes_per_source": 96,
        "detector_score_scale": 15.0,
        "camera_track_sigma_px": 26.0,
        "game_prior_default_radius_px": 48.0,
        "game_prior_max_weight": 0.06,
        # Base weights. Active weights are renormalised per cluster.
        "weight_camera_track": 0.20,
        "weight_detector": 0.18,
        "weight_simple_ai": 0.18,
        "weight_persistence": 0.12,
        "weight_novelty": 0.13,
        "weight_source_support": 0.05,
        "weight_external_expert": 0.35,
        "penalty_existed_before": 0.22,
        # Confidence is a resolver score, NOT a calibrated probability yet.
        "confidence_quality_weight": 0.60,
        "confidence_margin_weight": 0.30,
        "confidence_support_weight": 0.10,
        "confidence_margin_full_scale": 0.20,
    }

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(dict(config))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def resolve(
        self,
        default_xy: Point,
        camera_candidates: Sequence[Candidate],
        ranked_candidates: Sequence[Candidate] = (),
        external_votes: Optional[Mapping[str, Sequence[Candidate]]] = None,
        game_context: Optional[Mapping[str, Any]] = None,
        *,
        trust_percent: float = 100.0,
        mode: str = "advisory",
        shot_id: Optional[int] = None,
    ) -> ResolverDecision:
        started = time.perf_counter()
        observations = self._build_observations(
            default_xy=default_xy,
            camera_candidates=camera_candidates,
            ranked_candidates=ranked_candidates,
            external_votes=external_votes or {},
        )
        clusters = self._cluster_observations(observations)
        if not clusters:
            elapsed = (time.perf_counter() - started) * 1000.0
            return ResolverDecision(
                shot_id=shot_id,
                camera_x=float(default_xy[0]),
                camera_y=float(default_xy[1]),
                confidence=0.0,
                confidence_calibrated=False,
                score=0.0,
                margin=0.0,
                cluster_id=0,
                cluster_count=0,
                resolver_ms=elapsed,
                reason="no_candidates",
                source_names=("camera_track",),
                evidence={},
                alternatives=[],
            )

        trust = _clamp01(_safe_float(trust_percent, 100.0) / 100.0)
        mode_name = str(mode or "advisory").strip().lower()
        ai_evidence_scale = trust if mode_name == "blended" else 1.0

        for cluster in clusters:
            self._score_cluster(
                cluster,
                default_xy=default_xy,
                game_context=game_context,
                ai_evidence_scale=ai_evidence_scale,
            )

        clusters.sort(key=lambda c: (c.score, len(c.source_names)), reverse=True)
        best = clusters[0]
        second_score = clusters[1].score if len(clusters) > 1 else 0.0
        margin = max(0.0, best.score - second_score)
        margin_scale = max(1e-6, _safe_float(self.config.get("confidence_margin_full_scale", 0.20), 0.20))
        margin_conf = _clamp01(margin / margin_scale)
        support_conf = _clamp01((len(best.source_names) - 1) / 3.0)

        q_w = _safe_float(self.config.get("confidence_quality_weight", 0.60), 0.60)
        m_w = _safe_float(self.config.get("confidence_margin_weight", 0.30), 0.30)
        s_w = _safe_float(self.config.get("confidence_support_weight", 0.10), 0.10)
        denom = max(1e-9, q_w + m_w + s_w)
        confidence = _clamp01((q_w * best.score + m_w * margin_conf + s_w * support_conf) / denom)

        alternatives: List[Dict[str, Any]] = []
        for cluster in clusters[:5]:
            alternatives.append({
                "cluster_id": int(cluster.cluster_id),
                "camera_x": float(cluster.camera_x),
                "camera_y": float(cluster.camera_y),
                "score": float(cluster.score),
                "sources": list(cluster.source_names),
            })

        elapsed = (time.perf_counter() - started) * 1000.0
        return ResolverDecision(
            shot_id=shot_id,
            camera_x=float(best.camera_x),
            camera_y=float(best.camera_y),
            confidence=float(confidence),
            confidence_calibrated=False,
            score=float(best.score),
            margin=float(margin),
            cluster_id=int(best.cluster_id),
            cluster_count=len(clusters),
            resolver_ms=float(elapsed),
            reason="resolved",
            source_names=best.source_names,
            evidence=dict(best.evidence),
            alternatives=alternatives,
        )

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------
    def _build_observations(
        self,
        *,
        default_xy: Point,
        camera_candidates: Sequence[Candidate],
        ranked_candidates: Sequence[Candidate],
        external_votes: Mapping[str, Sequence[Candidate]],
    ) -> List[ResolverObservation]:
        observations: List[ResolverObservation] = [
            ResolverObservation(
                camera_x=float(default_xy[0]),
                camera_y=float(default_xy[1]),
                source="camera_track",
                source_score=1.0,
                detector_score=1.0,
                meta={"kind": "emitted_track"},
            )
        ]

        ranked_limit = max(1, int(self.config.get("max_ranked_candidates", 256)))
        rank_lookup: Dict[Tuple[int, int], Candidate] = {}
        for candidate in list(ranked_candidates)[:ranked_limit]:
            x = _safe_float(candidate.get("camera_x", 0.0))
            y = _safe_float(candidate.get("camera_y", 0.0))
            rank_lookup[(int(round(x * 2.0)), int(round(y * 2.0)))] = candidate

        camera_limit = max(1, int(self.config.get("max_camera_candidates", 256)))
        for candidate in list(camera_candidates)[:camera_limit]:
            x = _safe_float(candidate.get("camera_x", 0.0))
            y = _safe_float(candidate.get("camera_y", 0.0))
            key = (int(round(x * 2.0)), int(round(y * 2.0)))
            ranked = rank_lookup.get(key, {})
            source = str(candidate.get("source", candidate.get("proposal_source", "camera_detector")) or "camera_detector")
            observations.append(
                ResolverObservation(
                    camera_x=x,
                    camera_y=y,
                    source=source,
                    source_score=self._first_optional(candidate, "combined_score", "score"),
                    detector_score=self._first_optional(candidate, "score", "detector_score"),
                    simple_ai_score=self._first_optional(ranked, "ai_score", default=self._first_optional(candidate, "ai_score")),
                    persistence=self._first_optional(candidate, "persistence"),
                    existed_before=self._first_optional(candidate, "existed_before"),
                    source_support=self._first_optional(candidate, "dense_source_support", "source_support"),
                    meta={"kind": "camera_candidate", "rank": ranked.get("rank")},
                )
            )

        max_votes = max(1, int(self.config.get("max_external_votes_per_source", 96)))
        for source_name, source_votes in external_votes.items():
            source = str(source_name or "external")
            votes = list(source_votes)
            votes.sort(key=lambda item: _safe_float(item.get("score", item.get("confidence", 0.0))), reverse=True)
            for vote in votes[:max_votes]:
                x = _safe_float(vote.get("camera_x", vote.get("x", 0.0)))
                y = _safe_float(vote.get("camera_y", vote.get("y", 0.0)))
                score = _clamp01(_safe_float(vote.get("score", vote.get("confidence", 0.0))))
                observations.append(
                    ResolverObservation(
                        camera_x=x,
                        camera_y=y,
                        source=source,
                        source_score=score,
                        expert_weight=max(0.0, _safe_float(vote.get("expert_weight", 1.0), 1.0)),
                        meta={"kind": "external_vote"},
                    )
                )
        return observations

    @staticmethod
    def _first_optional(mapping: Mapping[str, Any], *keys: str, default: Optional[float] = None) -> Optional[float]:
        for key in keys:
            if key in mapping and mapping.get(key) is not None:
                return _safe_float(mapping.get(key))
        return default

    # ------------------------------------------------------------------
    # Spatial clustering
    # ------------------------------------------------------------------
    def _cluster_observations(self, observations: Sequence[ResolverObservation]) -> List[ResolverCluster]:
        radius = max(1.0, _safe_float(self.config.get("cluster_radius_px", 18.0), 18.0))
        cell_size = radius
        grid: Dict[Tuple[int, int], List[int]] = {}
        clusters: List[ResolverCluster] = []

        for obs in observations:
            cell = (int(math.floor(obs.camera_x / cell_size)), int(math.floor(obs.camera_y / cell_size)))
            best_cluster_idx: Optional[int] = None
            best_dist = float("inf")
            for gx in range(cell[0] - 1, cell[0] + 2):
                for gy in range(cell[1] - 1, cell[1] + 2):
                    for cluster_idx in grid.get((gx, gy), ()):
                        cluster = clusters[cluster_idx]
                        dist = math.hypot(obs.camera_x - cluster.camera_x, obs.camera_y - cluster.camera_y)
                        if dist <= radius and dist < best_dist:
                            best_dist = dist
                            best_cluster_idx = cluster_idx

            if best_cluster_idx is None:
                cluster_idx = len(clusters)
                cluster = ResolverCluster(
                    cluster_id=cluster_idx + 1,
                    observations=[obs],
                    camera_x=obs.camera_x,
                    camera_y=obs.camera_y,
                )
                clusters.append(cluster)
                grid.setdefault(cell, []).append(cluster_idx)
            else:
                cluster = clusters[best_cluster_idx]
                cluster.observations.append(obs)
                # The lookup centroid is allowed to move for clustering only.
                # Final emission is a medoid / real candidate, never this centroid.
                count = len(cluster.observations)
                cluster.camera_x += (obs.camera_x - cluster.camera_x) / count
                cluster.camera_y += (obs.camera_y - cluster.camera_y) / count

        # Replace working centroids with a real observation coordinate (medoid).
        for cluster in clusters:
            cx, cy = cluster.camera_x, cluster.camera_y
            medoid = min(
                cluster.observations,
                key=lambda obs: (math.hypot(obs.camera_x - cx, obs.camera_y - cy), -self._member_priority(obs)),
            )
            cluster.camera_x = medoid.camera_x
            cluster.camera_y = medoid.camera_y
        return clusters

    @staticmethod
    def _member_priority(obs: ResolverObservation) -> float:
        values = [v for v in (obs.source_score, obs.simple_ai_score, obs.persistence) if v is not None]
        if obs.source == "camera_track":
            values.append(1.0)
        return max(values) if values else 0.0

    # ------------------------------------------------------------------
    # Evidence scoring
    # ------------------------------------------------------------------
    def _score_cluster(
        self,
        cluster: ResolverCluster,
        *,
        default_xy: Point,
        game_context: Optional[Mapping[str, Any]],
        ai_evidence_scale: float,
    ) -> None:
        observations = cluster.observations
        source_names = tuple(sorted({str(obs.source) for obs in observations}))
        cluster.source_names = source_names

        default_dist = min(
            math.hypot(obs.camera_x - default_xy[0], obs.camera_y - default_xy[1])
            for obs in observations
        )
        sigma = max(1.0, _safe_float(self.config.get("camera_track_sigma_px", 26.0), 26.0))
        camera_track = math.exp(-0.5 * (default_dist / sigma) ** 2)

        detector_values = [
            _safe_float(obs.detector_score)
            for obs in observations
            if obs.detector_score is not None and obs.source != "camera_track"
        ]
        detector_scale = max(1e-6, _safe_float(self.config.get("detector_score_scale", 15.0), 15.0))
        detector = _clamp01(max(detector_values) / detector_scale) if detector_values else None

        ai_values = [_clamp01(_safe_float(obs.simple_ai_score)) for obs in observations if obs.simple_ai_score is not None]
        simple_ai = max(ai_values) if ai_values else None

        persistence_values = [_clamp01(_safe_float(obs.persistence)) for obs in observations if obs.persistence is not None]
        persistence = max(persistence_values) if persistence_values else None

        existed_values = [_clamp01(_safe_float(obs.existed_before)) for obs in observations if obs.existed_before is not None]
        novelty = (1.0 - sum(existed_values) / len(existed_values)) if existed_values else None

        explicit_support = [_clamp01(_safe_float(obs.source_support) / 4.0) for obs in observations if obs.source_support is not None]
        source_support = max(explicit_support) if explicit_support else _clamp01((len(source_names) - 1) / 3.0)

        external_values: List[Tuple[float, float]] = []
        for obs in observations:
            if obs.meta.get("kind") != "external_vote" or obs.source_score is None:
                continue
            external_values.append((_clamp01(_safe_float(obs.source_score)), max(0.0, obs.expert_weight)))
        external = None
        external_weight_multiplier = 1.0
        if external_values:
            weighted_sum = sum(score * weight for score, weight in external_values)
            total_weight = sum(weight for _score, weight in external_values)
            if total_weight > 1e-9:
                external = _clamp01(weighted_sum / total_weight)
                external_weight_multiplier = min(2.0, max(0.25, total_weight / len(external_values)))

        game_prior = self._game_prior_for_point((cluster.camera_x, cluster.camera_y), game_context)
        cluster.game_prior = game_prior

        evidence: Dict[str, float] = {
            "camera_track": float(camera_track),
            "source_support": float(source_support),
        }
        weighted: List[Tuple[str, float, float]] = [
            ("camera_track", camera_track, _safe_float(self.config.get("weight_camera_track", 0.20), 0.20)),
            ("source_support", source_support, _safe_float(self.config.get("weight_source_support", 0.05), 0.05)),
        ]
        if detector is not None:
            evidence["detector"] = float(detector)
            weighted.append(("detector", detector, _safe_float(self.config.get("weight_detector", 0.18), 0.18)))
        if simple_ai is not None and ai_evidence_scale > 0.0:
            evidence["simple_ai"] = float(simple_ai)
            weighted.append((
                "simple_ai",
                simple_ai,
                _safe_float(self.config.get("weight_simple_ai", 0.18), 0.18) * _clamp01(ai_evidence_scale),
            ))
        if persistence is not None:
            evidence["persistence"] = float(persistence)
            weighted.append(("persistence", persistence, _safe_float(self.config.get("weight_persistence", 0.12), 0.12)))
        if novelty is not None:
            evidence["novelty"] = float(novelty)
            weighted.append(("novelty", novelty, _safe_float(self.config.get("weight_novelty", 0.13), 0.13)))
        if external is not None and ai_evidence_scale > 0.0:
            evidence["external_expert"] = float(external)
            weighted.append((
                "external_expert",
                external,
                _safe_float(self.config.get("weight_external_expert", 0.35), 0.35)
                * external_weight_multiplier
                * _clamp01(ai_evidence_scale),
            ))
        if game_prior > 0.0:
            evidence["game_prior"] = float(game_prior)
            weighted.append((
                "game_prior",
                game_prior,
                _safe_float(self.config.get("game_prior_max_weight", 0.06), 0.06),
            ))

        total_weight = sum(max(0.0, weight) for _name, _value, weight in weighted)
        if total_weight <= 1e-9:
            cluster.score = 0.0
        else:
            base_score = sum(value * max(0.0, weight) for _name, value, weight in weighted) / total_weight
            existed_mean = (sum(existed_values) / len(existed_values)) if existed_values else 0.0
            existed_penalty = _safe_float(self.config.get("penalty_existed_before", 0.22), 0.22) * existed_mean
            evidence["existed_before_penalty"] = float(existed_penalty)
            cluster.score = _clamp01(base_score - existed_penalty)
        cluster.evidence = evidence

    def _game_prior_for_point(self, point: Point, game_context: Optional[Mapping[str, Any]]) -> float:
        if not game_context:
            return 0.0
        raw_priors: List[Mapping[str, Any]] = []
        for key in ("priors", "hotspots", "targets"):
            value = game_context.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                raw_priors.extend(item for item in value if isinstance(item, Mapping))

        best = 0.0
        default_radius = max(1.0, _safe_float(self.config.get("game_prior_default_radius_px", 48.0), 48.0))
        for prior in raw_priors:
            x = _safe_float(prior.get("camera_x", prior.get("x", 0.0)))
            y = _safe_float(prior.get("camera_y", prior.get("y", 0.0)))
            radius = max(1.0, _safe_float(prior.get("radius_px", default_radius), default_radius))
            score = _clamp01(_safe_float(prior.get("score", prior.get("weight", 1.0)), 1.0))
            dist = math.hypot(point[0] - x, point[1] - y)
            local = score * math.exp(-0.5 * (dist / radius) ** 2)
            best = max(best, local)
        return _clamp01(best)


__all__ = [
    "SCHEMA_VERSION",
    "ResolverDecision",
    "ShotResolverV222",
]
