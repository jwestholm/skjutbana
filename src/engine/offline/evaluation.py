"""Versioned observation scorecards; never invokes or changes detector policy."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from .metrics import DEFAULT_RADII, nearest_distance, within

VERSION = "1.1"
STAGES = ("raw", "filtered", "retained", "confirmed", "selected", "emitted")
LOSSES = ("before_candidates", "filtering", "retention", "confirmation", "selection", "emission_localization")
MODES = ("offline_candidate", "live_path_replay", "physical_validation")


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def provenance(root, paths, dataset_id, runtime):
    """Runtime describes the observation producer, not today's local settings."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    status = git("status", "--porcelain", "--untracked-files=all")
    files = [{"path": str(p), "sha256": digest(p)} for p in sorted(set(map(Path, paths)))]
    # Include untracked source bytes: a dirty flag/commit alone cannot freeze code.
    names = sorted(set(git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()))
    source = [{"path": n, "sha256": digest(Path(root) / n)} for n in names
              if (Path(root) / n).is_file() and (n.endswith(".py") or n.endswith(".md"))]
    return {"git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
            "working_tree_dirty": bool(status), "git_status": status,
            "source_manifest": source, "source_sha256": hashlib.sha256(encoded(source).encode()).hexdigest(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version(),
            "dataset_id": dataset_id, "input_files": files,
            "dataset_sha256": hashlib.sha256(encoded(files).encode()).hexdigest(),
            "producer_runtime": runtime, "tool": "offline.evaluation", "tool_version": VERSION,
            "schema_version": VERSION, "tolerances_camera_px": list(DEFAULT_RADII)}


def points(value):
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Stage must be null or a complete list of camera-coordinate observations")
    for p in value:
        if not isinstance(p, dict) or any(isinstance(p.get(k), bool) or not isinstance(p.get(k), (int, float))
                or not math.isfinite(p[k]) for k in ("camera_x", "camera_y")):
            raise ValueError("Invalid/nonfinite camera coordinates")
    return value


def _ground_truth_info(gt):
    if gt is None:
        return None
    quality = str(gt.get("quality", "precise"))
    if quality not in {"precise", "approximate", "unknown"}:
        raise ValueError("ground_truth quality must be precise, approximate, or unknown")
    radius = gt.get("uncertainty_radius_px")
    if quality == "approximate":
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius < 0:
            raise ValueError("approximate ground truth requires finite uncertainty_radius_px")
    elif radius is not None and (isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius < 0):
        raise ValueError("invalid ground truth uncertainty_radius_px")
    return {"quality": quality, "uncertainty_radius_px": None if radius is None else float(radius)}


def _interpret(distance, threshold, gt_info):
    if distance is None or gt_info is None:
        return None
    if distance <= threshold:
        return "within_threshold"
    if gt_info["quality"] == "approximate" and distance <= float(gt_info["uncertainty_radius_px"]):
        return "uncertain_within_label_uncertainty"
    if gt_info["quality"] == "unknown":
        return "uncertain_label_quality_unrecorded"
    return "outside_label_uncertainty" if gt_info["quality"] == "approximate" else "outside_threshold"


def _stage_distance(obs, xy):
    if obs is None or xy is None:
        return None
    distance = nearest_distance(obs, xy)
    return float("inf") if isinstance(obs, list) and not obs else distance


def evaluate(shots, *, mode, top_k=10):
    if mode not in MODES or type(top_k) is not int or top_k < 1:
        raise ValueError("Invalid mode or Top-K")
    seen = set()
    for shot in shots:
        key = (shot["session_id"], shot["shot_id"])
        if key in seen:
            raise ValueError(f"Duplicate shot identity: {key}")
        seen.add(key)
        if shot.get("coordinate_space") != "camera":
            raise ValueError("Only canonical full-camera pixel coordinates are accepted")
        gt = shot.get("ground_truth")
        if gt is not None:
            points([gt])
            _ground_truth_info(gt)
        for stage in (*STAGES, "saved_pool", "ranked"):
            points(shot.get(stage))
        if shot.get("selected") is not None and len(shot["selected"]) > 1:
            raise ValueError("Selected must contain at most one winner")
        if shot.get("rescue_used") is not None and type(shot["rescue_used"]) is not bool:
            raise ValueError("rescue_used must be boolean or null")
        latency = shot.get("latency_ms")
        if latency is not None and (type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0):
            raise ValueError("Invalid latency")
        if latency is not None and shot.get("emitted") == []:
            raise ValueError("Emission latency requires an emission; use null for shots with no emission")
    reports = {}
    for radius in DEFAULT_RADII:
        details = []
        names = (*STAGES, "saved_pool", "top_1", "top_3", "top_k")
        for shot in shots:
            gt = shot.get("ground_truth")
            xy = (gt["camera_x"], gt["camera_y"]) if gt else None
            gt_info = _ground_truth_info(gt)
            observations = {s: shot.get(s) for s in (*STAGES, "saved_pool")}
            ranked = shot.get("ranked")
            observations.update({name: None if ranked is None else ranked[:k]
                                 for name, k in (("top_1", 1), ("top_3", 3), ("top_k", top_k))})
            distances_by_stage = {s: _stage_distance(obs, xy)
                                  for s, obs in observations.items()}
            hits = {s: None if distances_by_stage[s] is None else within(distances_by_stage[s], radius)
                    for s, obs in observations.items()}
            interpretations = {s: _interpret(distances_by_stage[s], radius, gt_info)
                               for s in observations}
            # Only a fully observed successful prefix permits first-loss attribution.
            failure = "not_evaluated"
            for stage, loss in zip(STAGES, LOSSES):
                if hits[stage] is None:
                    break
                if not hits[stage]:
                    if interpretations[stage] in {"uncertain_within_label_uncertainty", "uncertain_label_quality_unrecorded"}:
                        failure = "uncertain_label"
                        break
                    failure = loss
                    break
            else:
                failure = "success"
            emissions = shot.get("emitted")
            distances = None if emissions is None or xy is None else [nearest_distance([p], xy) for p in emissions]
            correct = None if distances is None else sum(within(d, radius) for d in distances)
            emission_interpretations = None if distances is None else [_interpret(d, radius, gt_info) for d in distances]
            uncertain_emissions = None if emission_interpretations is None else sum(v in {"uncertain_within_label_uncertainty", "uncertain_label_quality_unrecorded"} for v in emission_interpretations)
            definitive_false = None if distances is None else len(distances) - correct - (uncertain_emissions or 0)
            details.append({"session_id": shot["session_id"], "shot_id": shot["shot_id"], "ground_truth_quality": None if gt_info is None else gt_info["quality"], "ground_truth_uncertainty_radius_px": None if gt_info is None else gt_info["uncertainty_radius_px"], "distances_px": distances_by_stage, "interpretations": interpretations, "emission_interpretations": emission_interpretations, "hits": hits,
                            "first_loss": failure,
                            "false_emissions": definitive_false,
                            "uncertain_emissions": uncertain_emissions,
                            "duplicates": None if correct is None else max(0, correct - 1)})
        stages = {}
        for name in names:
            observed = [d["hits"][name] for d in details if d["hits"][name] is not None]
            uncertain = sum(d["interpretations"].get(name) in {"uncertain_within_label_uncertainty", "uncertain_label_quality_unrecorded"} for d in details)
            stages[name] = {"correct": sum(observed), "evaluated": len(observed),
                            "unavailable": len(shots) - len(observed),
                            "accuracy_percent": 100 * sum(observed) / len(observed) if observed else None}
            if uncertain:
                stages[name]["uncertain"] = uncertain
        reports[str(int(radius))] = {"stages": stages, "first_loss_counts": dict(Counter(d["first_loss"] for d in details)),
            **{key: {"count": sum(d[key] for d in details if d[key] is not None),
                     "evaluated_shots": sum(d[key] is not None for d in details)} for key in ("false_emissions", "uncertain_emissions", "duplicates")},
            "shots": details}
    latencies = sorted(s["latency_ms"] for s in shots if s.get("latency_ms") is not None)
    rescue = [s["rescue_used"] for s in shots if s.get("rescue_used") is not None]
    return {"schema_version": VERSION, "mode": mode, "total_shots": len(shots),
            "labelled_shots": sum(s.get("ground_truth") is not None for s in shots),
            "source_kinds": dict(Counter(s.get("source_kind", "unknown") for s in shots)),
            "top_k": top_k, "tolerances_camera_px": reports,
            "rescue": {"used": sum(rescue), "evaluated": len(rescue)},
            "latency_ms": {"evaluated": len(latencies), "mean": mean(latencies) if latencies else None,
                           "median": median(latencies) if latencies else None,
                           "p95": latencies[math.ceil(.95 * len(latencies)) - 1] if latencies else None}}


def summary(report):
    lines = [f"Mode: {report['mode']} | TOTAL SHOTS: {report['total_shots']}",
             f"Sources: {report['source_kinds']}"]
    for radius, score in report["tolerances_camera_px"].items():
        lines.append(f"\nCamera tolerance <= {radius} px (Euclidean)")
        for stage, metric in score["stages"].items():
            lines.append(f"  {stage:12s}: {metric['correct']}/{metric['evaluated']} evaluated; {metric['unavailable']} unavailable")
        lines.append(f"  First loss: {score['first_loss_counts']}")
        lines.append(f"  False emissions: {score['false_emissions']}; duplicates: {score['duplicates']}")
    lines.extend([f"Rescue: {report['rescue']}", f"Latency ms: {report['latency_ms']}"])
    return "\n".join(lines)
