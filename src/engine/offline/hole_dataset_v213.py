from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class HoleAsset:
    image_path: Path
    metadata_path: Path | None
    kind: str  # synthetic | real
    session_id: str
    background_mode: str
    image_type: str
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def stem(self) -> str:
        return self.image_path.stem

    def to_dict(self, root: Path | None = None) -> dict[str, Any]:
        def rel(path: Path | None) -> str | None:
            if path is None:
                return None
            if root is not None:
                try:
                    return str(path.resolve().relative_to(root.resolve()))
                except Exception:
                    pass
            return str(path)

        return {
            "image": rel(self.image_path),
            "metadata": rel(self.metadata_path),
            "kind": self.kind,
            "session_id": self.session_id,
            "background_mode": self.background_mode,
            "image_type": self.image_type,
        }


@dataclass
class HoleArchiveSummary:
    root: str
    synthetic_png: int = 0
    real_png: int = 0
    synthetic_json: int = 0
    real_json: int = 0
    paired_synthetic: int = 0
    paired_real: int = 0
    invalid_json: int = 0
    missing_sidecar: int = 0
    unreadable_images: int = 0
    shape_counts: Counter[str] = field(default_factory=Counter)
    synthetic_backgrounds: Counter[str] = field(default_factory=Counter)
    real_backgrounds: Counter[str] = field(default_factory=Counter)
    synthetic_sessions: Counter[str] = field(default_factory=Counter)
    real_sessions: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        for key in (
            "shape_counts",
            "synthetic_backgrounds",
            "real_backgrounds",
            "synthetic_sessions",
            "real_sessions",
        ):
            result[key] = dict(result[key])
        return result


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[HoleAsset, ...]
    validation: tuple[HoleAsset, ...]
    test: tuple[HoleAsset, ...]
    background_holdout: tuple[HoleAsset, ...]
    real_holdout: tuple[HoleAsset, ...]
    session_assignment: dict[str, str]
    holdout_backgrounds: tuple[str, ...]

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
            "background_holdout": len(self.background_holdout),
            "real_holdout": len(self.real_holdout),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts(),
            "session_assignment": dict(sorted(self.session_assignment.items())),
            "holdout_backgrounds": list(self.holdout_backgrounds),
            "backgrounds": {
                name: dict(Counter(asset.background_mode for asset in values))
                for name, values in (
                    ("train", self.train),
                    ("validation", self.validation),
                    ("test", self.test),
                    ("background_holdout", self.background_holdout),
                    ("real_holdout", self.real_holdout),
                )
            },
            "sessions": {
                name: dict(Counter(asset.session_id for asset in values))
                for name, values in (
                    ("train", self.train),
                    ("validation", self.validation),
                    ("test", self.test),
                    ("background_holdout", self.background_holdout),
                    ("real_holdout", self.real_holdout),
                )
            },
        }


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _kind_from_name(path: Path, metadata: dict[str, Any]) -> str | None:
    stem = path.stem.lower()
    image_type = str(metadata.get("image_type", "")).strip().lower()
    if stem.startswith("synt_") or image_type in {"synt", "synthetic"}:
        return "synthetic"
    if stem.startswith("hole_") or image_type in {"hole", "real", "physical"}:
        return "real"
    return None


def discover_hole_assets(
    root: Path,
    *,
    inspect_images: bool = True,
) -> tuple[list[HoleAsset], HoleArchiveSummary]:
    """Discover the existing ``content/ai/holes`` bank without modifying it.

    V2.13 deliberately ignores ``shot_diag``.  ``holes`` contains raw 128x128
    camera patches and sidecar metadata; ``shot_diag`` is human diagnostics with
    drawn annotations and therefore must not silently enter model training.
    """

    root = Path(root).expanduser().resolve()
    summary = HoleArchiveSummary(root=str(root))
    assets: list[HoleAsset] = []

    images = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and (path.stem.lower().startswith("synt_") or path.stem.lower().startswith("hole_"))
    )
    json_files = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
        and (path.stem.lower().startswith("synt_") or path.stem.lower().startswith("hole_"))
    )

    summary.synthetic_png = sum(path.stem.lower().startswith("synt_") for path in images)
    summary.real_png = sum(path.stem.lower().startswith("hole_") for path in images)
    summary.synthetic_json = sum(path.stem.lower().startswith("synt_") for path in json_files)
    summary.real_json = sum(path.stem.lower().startswith("hole_") for path in json_files)

    for image_path in images:
        sidecar = image_path.with_suffix(".json")
        if not sidecar.exists():
            summary.missing_sidecar += 1
            continue
        metadata = _load_json(sidecar)
        if metadata is None:
            summary.invalid_json += 1
            continue
        kind = _kind_from_name(image_path, metadata)
        if kind is None:
            continue
        session_id = str(metadata.get("session_id") or "unknown")
        background = str(metadata.get("background_mode") or "unknown")
        image_type = str(metadata.get("image_type") or ("synt" if kind == "synthetic" else "hole"))

        if inspect_images:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None or not image.size:
                summary.unreadable_images += 1
                continue
            h, w = image.shape[:2]
            summary.shape_counts[f"{w}x{h}"] += 1

        asset = HoleAsset(
            image_path=image_path,
            metadata_path=sidecar,
            kind=kind,
            session_id=session_id,
            background_mode=background,
            image_type=image_type,
            metadata=metadata,
        )
        assets.append(asset)
        if kind == "synthetic":
            summary.paired_synthetic += 1
            summary.synthetic_backgrounds[background] += 1
            summary.synthetic_sessions[session_id] += 1
        else:
            summary.paired_real += 1
            summary.real_backgrounds[background] += 1
            summary.real_sessions[session_id] += 1

    return assets, summary


def _stable_tiebreak(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _assign_sessions_balanced(
    assets: Sequence[HoleAsset],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, str]:
    counts = Counter(asset.session_id for asset in assets)
    sessions = list(counts)
    if not sessions:
        return {}

    if len(sessions) == 1:
        return {sessions[0]: "train"}
    if len(sessions) == 2:
        ordered = sorted(sessions, key=lambda s: (-counts[s], _stable_tiebreak(s, seed)))
        return {ordered[0]: "train", ordered[1]: "test"}

    train_fraction = float(np.clip(train_fraction, 0.50, 0.90))
    validation_fraction = float(np.clip(validation_fraction, 0.05, 0.30))
    test_fraction = max(0.05, 1.0 - train_fraction - validation_fraction)
    total_fraction = train_fraction + validation_fraction + test_fraction
    fractions = {
        "train": train_fraction / total_fraction,
        "validation": validation_fraction / total_fraction,
        "test": test_fraction / total_fraction,
    }
    total = float(sum(counts.values()))
    targets = {name: max(1.0, total * fraction) for name, fraction in fractions.items()}
    assigned_counts = {name: 0 for name in targets}
    assignment: dict[str, str] = {}

    # Seed each split with one session.  Largest first, but use deterministic
    # tie-breaking so reruns are reproducible.
    ordered = sorted(sessions, key=lambda s: (-counts[s], _stable_tiebreak(s, seed)))
    seed_order = ["train", "validation", "test"]
    for session, split_name in zip(ordered[:3], seed_order):
        assignment[session] = split_name
        assigned_counts[split_name] += counts[session]

    # Greedily put remaining whole sessions into the split furthest below its
    # target.  No image from one physical session can leak into another split.
    for session in ordered[3:]:
        def need(split_name: str) -> tuple[float, int]:
            ratio = assigned_counts[split_name] / targets[split_name]
            return ratio, _stable_tiebreak(f"{session}:{split_name}", seed)

        split_name = min(targets, key=need)
        assignment[session] = split_name
        assigned_counts[split_name] += counts[session]

    return assignment


def build_dataset_split(
    assets: Sequence[HoleAsset],
    *,
    holdout_backgrounds: Iterable[str] = ("black", "checker", "gray", "bubbles"),
    seed: int = 21301,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> DatasetSplit:
    holdout = tuple(sorted({str(value) for value in holdout_backgrounds if str(value)}))
    holdout_set = set(holdout)

    real = [asset for asset in assets if asset.kind == "real"]
    synthetic = [asset for asset in assets if asset.kind == "synthetic"]
    background_holdout = [asset for asset in synthetic if asset.background_mode in holdout_set]
    eligible = [asset for asset in synthetic if asset.background_mode not in holdout_set]

    assignment = _assign_sessions_balanced(
        eligible,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    by_split: dict[str, list[HoleAsset]] = defaultdict(list)
    for asset in eligible:
        by_split[assignment.get(asset.session_id, "train")].append(asset)

    return DatasetSplit(
        train=tuple(sorted(by_split.get("train", []), key=lambda a: a.image_path.name)),
        validation=tuple(sorted(by_split.get("validation", []), key=lambda a: a.image_path.name)),
        test=tuple(sorted(by_split.get("test", []), key=lambda a: a.image_path.name)),
        background_holdout=tuple(sorted(background_holdout, key=lambda a: a.image_path.name)),
        real_holdout=tuple(sorted(real, key=lambda a: a.image_path.name)),
        session_assignment=assignment,
        holdout_backgrounds=holdout,
    )


def read_gray(asset: HoleAsset) -> np.ndarray:
    image = cv2.imread(str(asset.image_path), cv2.IMREAD_GRAYSCALE)
    if image is None or not image.size:
        raise ValueError(f"Could not read {asset.image_path}")
    return image


def image_center_xy(image: np.ndarray) -> tuple[float, float]:
    h, w = image.shape[:2]
    return (0.5 * (w - 1), 0.5 * (h - 1))


def _sample_offset_in_disk(rng: np.random.Generator, radius: float, minimum: float = 0.0) -> tuple[float, float]:
    radius = max(float(radius), 0.0)
    minimum = max(0.0, min(float(minimum), radius))
    if radius <= 1e-9:
        return 0.0, 0.0
    # Uniform by area in the annulus.
    r2 = rng.uniform(minimum * minimum, radius * radius)
    r = math.sqrt(r2)
    angle = rng.uniform(0.0, 2.0 * math.pi)
    return r * math.cos(angle), r * math.sin(angle)


def _candidate_center_is_safe(
    center_xy: tuple[float, float],
    shape: tuple[int, int],
    crop_size: int,
) -> bool:
    h, w = shape[:2]
    half = crop_size / 2.0
    x, y = center_xy
    return half <= x <= (w - 1) - half and half <= y <= (h - 1) - half


def sample_candidate_center(
    image: np.ndarray,
    *,
    rng: np.random.Generator,
    label: int,
    crop_size: int,
    positive_jitter_px: float,
    negative_min_px: float,
    negative_max_px: float,
    max_tries: int = 100,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ``(candidate_xy, hole_minus_candidate_xy)``.

    The source hole bank stores the physical/synthetic hole at the centre of its
    128x128 crop.  We *never* feed that source crop straight to the model.  The
    model sees a smaller crop centred on a sampled candidate.  Therefore the
    hole moves within the model input and the candidate centre has the same
    semantic meaning for positive and negative samples.
    """

    gt_x, gt_y = image_center_xy(image)
    for _ in range(max_tries):
        if int(label) == 1:
            dx, dy = _sample_offset_in_disk(rng, positive_jitter_px)
        else:
            dx, dy = _sample_offset_in_disk(rng, negative_max_px, negative_min_px)
        candidate = (gt_x + dx, gt_y + dy)
        if _candidate_center_is_safe(candidate, image.shape, crop_size):
            # Target refinement is from candidate to true hole.
            return candidate, (gt_x - candidate[0], gt_y - candidate[1])
    raise ValueError(
        f"Could not sample a safe {'positive' if label else 'negative'} candidate "
        f"for image {image.shape}, crop_size={crop_size}"
    )


def crop_candidate(image: np.ndarray, center_xy: tuple[float, float], crop_size: int) -> np.ndarray:
    crop_size = int(crop_size)
    if crop_size < 8:
        raise ValueError("crop_size must be >=8")
    patch = cv2.getRectSubPix(image, (crop_size, crop_size), (float(center_xy[0]), float(center_xy[1])))
    if patch is None or patch.shape[:2] != (crop_size, crop_size):
        raise ValueError("Could not extract candidate-centred crop")
    return patch


def center_contrast_score(patch: np.ndarray) -> float:
    """Simple non-learning baseline: dark centre versus surrounding annulus."""
    gray = patch.astype(np.float32)
    h, w = gray.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = 0.5 * (w - 1), 0.5 * (h - 1)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inner_r = max(2.0, min(h, w) * 0.08)
    ring_inner = max(inner_r + 1.0, min(h, w) * 0.14)
    ring_outer = max(ring_inner + 1.0, min(h, w) * 0.30)
    inner = gray[rr <= inner_r]
    ring = gray[(rr >= ring_inner) & (rr <= ring_outer)]
    if inner.size == 0 or ring.size == 0:
        return 0.0
    # Positive means the candidate centre is darker than its local surround.
    denom = max(8.0, float(np.std(ring)))
    return float((np.mean(ring) - np.mean(inner)) / denom)


def iter_assets_limited(assets: Sequence[HoleAsset], max_assets: int | None, seed: int) -> tuple[HoleAsset, ...]:
    if max_assets is None or int(max_assets) <= 0 or len(assets) <= int(max_assets):
        return tuple(assets)
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(assets))
    rng.shuffle(indices)
    selected = sorted(indices[: int(max_assets)])
    return tuple(assets[int(index)] for index in selected)
