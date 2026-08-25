from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .shot_case import GroundTruth, ShotCase


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Intentionally permissive: the shooting PC has data from several generations.
# We do not rename or mutate source files; ambiguous groups are reported.
ROLE_RE = re.compile(
    r"(?i)(?:^|[_\-.])"
    r"(pre|before|reference|ref|without(?:[_\-]?hole)?|post|after|with(?:[_\-]?hole)?|diff|delta)"
    r"(?:[_\-.]?(\d+))?(?=$|[_\-.])"
)

PRE_ROLES = {"pre", "before", "reference", "ref", "without", "without_hole", "without-hole"}
POST_ROLES = {"post", "after", "with", "with_hole", "with-hole"}
IGNORE_ROLES = {"diff", "delta"}


@dataclass
class ArchiveSummary:
    root: str
    image_files: int = 0
    candidate_groups: int = 0
    paired_shots: int = 0
    labelled_shots: int = 0
    unlabelled_shots: int = 0
    ambiguous_groups: int = 0
    ignored_images: int = 0
    unreadable_images: int = 0
    shape_mismatch_shots: int = 0
    background_types: Counter[str] = field(default_factory=Counter)
    role_counts: Counter[str] = field(default_factory=Counter)
    extension_counts: Counter[str] = field(default_factory=Counter)
    standalone_labelled_images: int = 0
    examples_ambiguous: list[str] = field(default_factory=list)
    examples_unclassified: list[str] = field(default_factory=list)
    examples_standalone_labelled: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["background_types"] = dict(self.background_types)
        result["role_counts"] = dict(self.role_counts)
        result["extension_counts"] = dict(self.extension_counts)
        return result


def _normalise_role(role: str) -> str:
    value = role.lower().replace("-", "_")
    if value.startswith("without"):
        return "without_hole"
    if value.startswith("with"):
        return "with_hole"
    return value


def classify_filename(path: Path) -> tuple[str | None, str]:
    """Return (role, canonical-shot-key).

    Examples:
      shot_0042_pre.png      -> (pre, shot_0042)
      shot_0042_post_03.png  -> (post, shot_0042)
      0042_before.jpg        -> (before, 0042)
    """

    stem = path.stem
    match = ROLE_RE.search(stem)
    if not match:
        return None, stem.lower()
    role = _normalise_role(match.group(1))
    # Remove role and an optional immediately-associated frame index.
    key = (stem[: match.start()] + stem[match.end() :]).strip("_.-")
    key = re.sub(r"[_\-.]{2,}", "_", key).lower()
    return role, key or stem.lower()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _dig(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def extract_ground_truth(metadata: dict[str, Any]) -> GroundTruth | None:
    direct = GroundTruth.from_value(metadata.get("ground_truth") or metadata.get("gt"))
    if direct:
        return direct

    xy_keys = (
        "gt_xy",
        "ground_truth_xy",
        "camera_xy",
        "hole_xy",
        "center_xy",
    )
    for key in xy_keys:
        gt = GroundTruth.from_value(metadata.get(key))
        if gt:
            return gt

    pairs = (
        ("gt_camera_x", "gt_camera_y"),
        ("ground_truth_camera_x", "ground_truth_camera_y"),
        ("camera_x", "camera_y"),
        ("hole_x", "hole_y"),
        ("x", "y"),
    )
    for x_key, y_key in pairs:
        if metadata.get(x_key) is not None and metadata.get(y_key) is not None:
            try:
                return GroundTruth(float(metadata[x_key]), float(metadata[y_key]))
            except Exception:
                pass

    nested_paths = (
        (("synthetic_hole", "camera_x"), ("synthetic_hole", "camera_y")),
        (("hole", "camera_x"), ("hole", "camera_y")),
    )
    for x_path, y_path in nested_paths:
        x = _dig(metadata, x_path)
        y = _dig(metadata, y_path)
        if x is not None and y is not None:
            try:
                return GroundTruth(float(x), float(y))
            except Exception:
                pass
    return None


def _metadata_candidates(group_key: str, images: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for image in images:
        for candidate in (
            image.with_suffix(".json"),
            image.parent / f"{group_key}.json",
            image.parent / "metadata.json",
        ):
            if candidate not in seen and candidate.exists():
                seen.add(candidate)
                result.append(candidate)
    return result


def _merge_metadata(paths: Iterable[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    sources: list[str] = []
    for path in paths:
        payload = _load_json(path)
        if not payload:
            continue
        # Earlier / more specific keys win; metadata source list remains explicit.
        for key, value in payload.items():
            merged.setdefault(key, value)
        sources.append(str(path))
    if sources:
        merged["_metadata_sources"] = sources
    return merged


def _read_gray(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image if isinstance(image, np.ndarray) and image.size else None


def background_stats(path: Path) -> dict[str, Any]:
    gray = _read_gray(path)
    if gray is None:
        return {"readable": False, "background_type": "unreadable"}
    # Downsample for cheap archive-scale inspection.
    h, w = gray.shape[:2]
    scale = min(1.0, 512.0 / float(max(h, w)))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    p10, p50, p90 = (float(v) for v in np.percentile(gray, [10, 50, 90]))
    edge = cv2.Laplacian(gray, cv2.CV_32F)
    edge_energy = float(np.mean(np.abs(edge)))

    if mean >= 185.0 and std <= 45.0:
        kind = "mostly_white"
    elif mean <= 70.0:
        kind = "dark"
    elif std >= 58.0 or edge_energy >= 20.0:
        kind = "textured"
    else:
        kind = "mid_tone"
    return {
        "readable": True,
        "width": int(w),
        "height": int(h),
        "mean": round(mean, 3),
        "std": round(std, 3),
        "p10": round(p10, 3),
        "p50": round(p50, 3),
        "p90": round(p90, 3),
        "edge_energy": round(edge_energy, 3),
        "background_type": kind,
    }


def discover_shot_cases(
    root: Path,
    *,
    inspect_images: bool = True,
    limit: int | None = None,
) -> tuple[list[ShotCase], ArchiveSummary]:
    root = Path(root).expanduser().resolve()
    summary = ArchiveSummary(root=str(root))
    groups: dict[tuple[Path, str], dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))

    image_paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    summary.image_files = len(image_paths)

    for path in image_paths:
        summary.extension_counts[path.suffix.lower()] += 1
        role, key = classify_filename(path)
        if role is None:
            # Unknown images are not silently treated as replay pairs.  Still
            # recognise labelled standalone hole patches: they are valuable for
            # the later Hole-AI even though they cannot drive before/after replay.
            sidecar = path.with_suffix(".json")
            sidecar_meta = _load_json(sidecar) if sidecar.exists() else {}
            if extract_ground_truth(sidecar_meta) is not None or sidecar_meta.get("image_type") is not None:
                summary.standalone_labelled_images += 1
                if len(summary.examples_standalone_labelled) < 20:
                    summary.examples_standalone_labelled.append(str(path.relative_to(root)))
            elif len(summary.examples_unclassified) < 20:
                summary.examples_unclassified.append(str(path.relative_to(root)))
            summary.ignored_images += 1
            continue
        summary.role_counts[role] += 1
        if role in IGNORE_ROLES:
            summary.ignored_images += 1
            continue
        groups[(path.parent, key)][role].append(path)

    summary.candidate_groups = len(groups)
    cases: list[ShotCase] = []

    for (parent, key), role_map in sorted(groups.items(), key=lambda item: (str(item[0][0]), item[0][1])):
        pre: list[Path] = []
        post: list[Path] = []
        for role, paths in role_map.items():
            if role in PRE_ROLES:
                pre.extend(paths)
            elif role in POST_ROLES:
                post.extend(paths)

        pre = sorted(set(pre))
        post = sorted(set(post))
        if not pre or not post:
            summary.ambiguous_groups += 1
            if len(summary.examples_ambiguous) < 20:
                summary.examples_ambiguous.append(str(parent / key))
            continue

        all_images = pre + post
        metadata = _merge_metadata(_metadata_candidates(key, all_images))
        gt = extract_ground_truth(metadata)

        image_meta: dict[str, Any] = {}
        shape_ok = True
        if inspect_images:
            first_pre = _read_gray(pre[0])
            first_post = _read_gray(post[0])
            if first_pre is None or first_post is None:
                summary.unreadable_images += int(first_pre is None) + int(first_post is None)
                image_meta = {"readable": False}
            else:
                shape_ok = first_pre.shape == first_post.shape
                image_meta = background_stats(pre[0])
                image_meta["shape_match"] = bool(shape_ok)
                if not shape_ok:
                    summary.shape_mismatch_shots += 1
                kind = str(image_meta.get("background_type", "unknown"))
                summary.background_types[kind] += 1

        relative = lambda path: path.relative_to(root).as_posix()  # noqa: E731
        case = ShotCase(
            shot_id=f"{parent.relative_to(root).as_posix()}::{key}" if parent != root else key,
            session_id=(root.name if parent == root else parent.relative_to(root).as_posix()),
            pre_images=tuple(relative(path) for path in pre),
            post_images=tuple(relative(path) for path in post),
            ground_truth=gt,
            metadata={
                **metadata,
                "archive": {
                    "canonical_key": key,
                    "pre_count": len(pre),
                    "post_count": len(post),
                    "image": image_meta,
                    "shape_ok": bool(shape_ok),
                },
            },
        )
        cases.append(case)
        summary.paired_shots += 1
        if gt is None:
            summary.unlabelled_shots += 1
        else:
            summary.labelled_shots += 1

        if limit is not None and len(cases) >= max(0, int(limit)):
            break

    return cases, summary
