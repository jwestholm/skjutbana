from __future__ import annotations

"""Media-bank indexing for V2.19 offline scenario generation.

The bank deliberately stores *source-level* provenance.  A video clip is one
asset and therefore one split unit; frames from the same clip may never leak
between training and holdout merely because they were sampled at different
frame indices.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".gif"}
DEFAULT_MANIFEST = Path("content/ai/media_bank_v219/media_manifest.jsonl")
DEFAULT_SUMMARY = Path("content/ai/media_bank_v219/media_summary.json")


@dataclass(frozen=True)
class MediaAssetV219:
    media_id: str
    path: str
    kind: str  # image | video
    split: str
    category: str = "unknown"
    family_id: str = ""
    width: int = 0
    height: int = 0
    frame_count: int = 1
    fps: float = 0.0
    license: str = "unknown"
    source_url: str = ""
    content_sha256: str = ""
    perceptual_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "MediaAssetV219":
        return cls(
            media_id=str(row.get("media_id") or row.get("id") or ""),
            path=str(row.get("path") or ""),
            kind=str(row.get("kind") or "image"),
            split=str(row.get("split") or "train"),
            category=str(row.get("category") or "unknown"),
            family_id=str(row.get("family_id") or row.get("media_id") or ""),
            width=int(row.get("width") or 0),
            height=int(row.get("height") or 0),
            frame_count=max(1, int(row.get("frame_count") or 1)),
            fps=float(row.get("fps") or 0.0),
            license=str(row.get("license") or "unknown"),
            source_url=str(row.get("source_url") or ""),
            content_sha256=str(row.get("content_sha256") or ""),
            perceptual_hash=str(row.get("perceptual_hash") or ""),
            metadata=dict(row.get("metadata") or {}),
        )


def _stable_fraction(value: str, salt: str) -> float:
    digest = hashlib.sha256((salt + "\n" + value).encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big", signed=False)
    return integer / float(2**64 - 1)


def split_for_family(
    family_id: str,
    *,
    salt: str = "skjutbana-v219-media-split-v1",
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
) -> str:
    """Assign a whole source/family to one deterministic split."""
    p = _stable_fraction(str(family_id), salt)
    train_fraction = min(max(float(train_fraction), 0.0), 1.0)
    validation_fraction = min(max(float(validation_fraction), 0.0), 1.0 - train_fraction)
    if p < train_fraction:
        return "train"
    if p < train_fraction + validation_fraction:
        return "validation"
    return "holdout"


def _read_sidecar(path: Path) -> dict[str, Any]:
    candidates = [
        path.with_suffix(path.suffix + ".json"),
        path.with_suffix(".json"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate != path:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
    return {}


def _category_from_path(path: Path, kind: str) -> str:
    text = " ".join(part.lower() for part in path.parts[-4:])
    tests: list[tuple[str, tuple[str, ...]]] = [
        ("game", ("game", "spel", "sprite", "hud", "enemy", "target")),
        ("painting", ("painting", "art", "tavla", "poster", "canvas")),
        ("people", ("people", "person", "portrait", "face", "human")),
        ("nature", ("nature", "forest", "wood", "tree", "landscape", "grass")),
        ("urban", ("city", "urban", "street", "building", "indoor", "room")),
        ("pattern", ("checker", "grid", "pattern", "stripe", "texture")),
        ("text_ui", ("text", "ui", "menu", "screen", "terminal")),
    ]
    for category, words in tests:
        if any(word in text for word in words):
            return category
    if kind == "video":
        return "video"
    return "photo_or_image"


def _probe(path: Path, kind: str) -> tuple[int, int, int, float]:
    if kind == "image":
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return 0, 0, 1, 0.0
        h, w = image.shape[:2]
        return int(w), int(h), 1, 0.0

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0, 1, 0.0
    try:
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
        frame_count = max(1, int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return width, height, frame_count, fps
    finally:
        cap.release()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()
    except Exception:
        return ""


def _representative_gray(path: Path, kind: str) -> Any:
    if kind == "image":
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, count // 2))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    finally:
        cap.release()


def _dhash(gray: Any) -> str:
    if gray is None or not getattr(gray, "size", 0):
        return ""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def hamming_hash(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 999
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except Exception:
        return 999


def _media_id(path: Path, family_id: str) -> str:
    digest = hashlib.sha1((str(path) + "\n" + family_id).encode("utf-8")).hexdigest()[:16]
    return f"media_{digest}"


def index_media_roots(
    roots: Sequence[Path | str],
    *,
    repo_root: Path | None = None,
    split_salt: str = "skjutbana-v219-media-split-v1",
    include_hidden: bool = False,
) -> list[MediaAssetV219]:
    repo_root = Path(repo_root).resolve() if repo_root is not None else None
    seen: set[Path] = set()
    assets: list[MediaAssetV219] = []

    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        files: Iterable[Path] = [root] if root.is_file() else root.rglob("*")
        for path in files:
            if not path.is_file():
                continue
            if not include_hidden and any(part.startswith(".") for part in path.parts):
                continue
            suffix = path.suffix.lower()
            if suffix not in IMAGE_EXTENSIONS and suffix not in VIDEO_EXTENSIONS:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            kind = "image" if suffix in IMAGE_EXTENSIONS else "video"
            sidecar = _read_sidecar(path)
            family_id = str(sidecar.get("family_id") or sidecar.get("source_family") or resolved)
            split = str(sidecar.get("split") or split_for_family(family_id, salt=split_salt))
            if split not in {"train", "validation", "holdout"}:
                split = split_for_family(family_id, salt=split_salt)
            category = str(sidecar.get("category") or sidecar.get("media_category") or _category_from_path(path, kind))
            width, height, frame_count, fps = _probe(path, kind)
            content_sha256 = _sha256_file(path)
            perceptual_hash = _dhash(_representative_gray(path, kind))
            stored_path = str(path)
            if repo_root is not None:
                try:
                    stored_path = str(resolved.relative_to(repo_root))
                except Exception:
                    stored_path = str(resolved)
            metadata = {k: v for k, v in sidecar.items() if k not in {
                "family_id", "source_family", "split", "category", "media_category", "license", "source_url"
            }}
            assets.append(
                MediaAssetV219(
                    media_id=_media_id(resolved, family_id),
                    path=stored_path,
                    kind=kind,
                    split=split,
                    category=category,
                    family_id=family_id,
                    width=width,
                    height=height,
                    frame_count=frame_count,
                    fps=fps,
                    license=str(sidecar.get("license") or "unknown"),
                    source_url=str(sidecar.get("source_url") or ""),
                    content_sha256=content_sha256,
                    perceptual_hash=perceptual_hash,
                    metadata=metadata,
                )
            )
    return sorted(assets, key=lambda a: (a.split, a.category, a.path))


def write_media_manifest(path: Path, assets: Sequence[MediaAssetV219]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for asset in assets:
            handle.write(json.dumps(asset.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def read_media_manifest(path: Path = DEFAULT_MANIFEST) -> list[MediaAssetV219]:
    path = Path(path)
    if not path.exists():
        return []
    result: list[MediaAssetV219] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            result.append(MediaAssetV219.from_dict(row))
    return result


def summarise_media(assets: Sequence[MediaAssetV219]) -> dict[str, Any]:
    by_split: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    unknown_license = 0
    for asset in assets:
        by_split[asset.split] = by_split.get(asset.split, 0) + 1
        by_category[asset.category] = by_category.get(asset.category, 0) + 1
        by_kind[asset.kind] = by_kind.get(asset.kind, 0) + 1
        if not asset.license or asset.license == "unknown":
            unknown_license += 1
    return {
        "schema_version": "2.19",
        "assets": len(assets),
        "by_split": dict(sorted(by_split.items())),
        "by_category": dict(sorted(by_category.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "unknown_license": unknown_license,
        "families": len({asset.family_id for asset in assets}),
    }


def audit_media(assets: Sequence[MediaAssetV219], *, near_duplicate_hamming: int = 3) -> dict[str, Any]:
    exact_groups: dict[str, list[MediaAssetV219]] = {}
    for asset in assets:
        if asset.content_sha256:
            exact_groups.setdefault(asset.content_sha256, []).append(asset)
    exact_cross_split = []
    for digest, group in exact_groups.items():
        splits = sorted({asset.split for asset in group})
        if len(splits) > 1:
            exact_cross_split.append({"sha256": digest, "splits": splits, "paths": [asset.path for asset in group]})

    # BK-tree over 64-bit dHash values.  This keeps near-duplicate auditing
    # practical for banks with thousands/tens of thousands of media sources
    # without an O(n^2) all-pairs scan.
    class _BKNode:
        __slots__ = ("value", "assets", "children")
        def __init__(self, value: int, asset: MediaAssetV219):
            self.value = value
            self.assets = [asset]
            self.children: dict[int, "_BKNode"] = {}

    def _dist(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    root = None
    near_cross_split: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    threshold = max(0, int(near_duplicate_hamming))
    for asset in (a for a in assets if a.perceptual_hash):
        try:
            value = int(asset.perceptual_hash, 16)
        except Exception:
            continue
        if root is not None:
            stack = [root]
            while stack:
                node = stack.pop()
                d = _dist(value, node.value)
                if d <= threshold:
                    for other in node.assets:
                        if other.split == asset.split:
                            continue
                        pair = tuple(sorted((other.path, asset.path)))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        near_cross_split.append({
                            "distance": d, "left": other.path, "left_split": other.split,
                            "right": asset.path, "right_split": asset.split,
                        })
                        if len(near_cross_split) >= 500:
                            break
                if len(near_cross_split) >= 500:
                    break
                low, high = d - threshold, d + threshold
                for edge, child in node.children.items():
                    if low <= edge <= high:
                        stack.append(child)
        if root is None:
            root = _BKNode(value, asset)
        else:
            node = root
            while True:
                d = _dist(value, node.value)
                if d == 0:
                    node.assets.append(asset)
                    break
                child = node.children.get(d)
                if child is None:
                    node.children[d] = _BKNode(value, asset)
                    break
                node = child

    return {
        "schema_version": "2.19",
        "assets": len(assets),
        "exact_cross_split_duplicates": exact_cross_split,
        "near_cross_split_duplicates": near_cross_split,
        "near_duplicate_check_skipped": False,
        "unknown_license_paths": [asset.path for asset in assets if not asset.license or asset.license == "unknown"],
        "ok_for_frozen_holdout": not exact_cross_split and not near_cross_split,
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def default_media_roots(repo_root: Path = Path(".")) -> list[Path]:
    roots = [repo_root / "content" / "ai" / "media_bank", repo_root / "assets"]
    return [path for path in roots if path.exists()]
