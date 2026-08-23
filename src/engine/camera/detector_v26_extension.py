from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from src.engine.camera.detector_v24_extension import _ensure_diag_record, _patch_prior


CONFIG_PATH = Path("content/ai/detector_v26.json")

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    # V2.5 data: v24/v25 BEST/EVER ~=81%, but only ~59% survived to the
    # evaluation snapshot. V2.6 therefore preserves spatially distinct
    # hypotheses over the complete pending-shot window without requiring a
    # candidate to repeat on several camera frames.
    "shot_vault_enabled": True,
    "shot_vault_cell_px": 10.0,
    "shot_vault_dedupe_radius_px": 4.5,
    "shot_vault_macro_cell_px": 120.0,
    "shot_vault_max_cells": 900,
    "shot_vault_carried_slots": 420,
    "shot_vault_output_limit": 680,
    "shot_vault_max_age_s": 2.2,
    "shot_vault_keep_shots": 6,
    # A carried point is evidence from an earlier *real camera frame*. Do not
    # reward it just because it was carried; the ranker may use hits/source
    # metadata separately.
    "shot_vault_score_decay": 0.98,
    # Keep diagnostics cheap but complete.
    "diagnostics_enabled": True,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value_f = float(value)
        return value_f if math.isfinite(value_f) else float(default)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _nearest(candidates: list[dict[str, Any]], gx: float, gy: float) -> float | None:
    if not candidates:
        return None
    return min(
        math.hypot(
            _safe_float(candidate.get("camera_x")) - gx,
            _safe_float(candidate.get("camera_y")) - gy,
        )
        for candidate in candidates
    )


def _candidate_quality(candidate: dict[str, Any]) -> float:
    """Only used to choose between hypotheses already in the SAME 10 px cell.

    It deliberately does not globally sort the shot. V2.5 showed that detector
    strength/patch priors are not trustworthy enough to decide which region of
    the image is the hole, but they are safe tie breakers between nearly
    coincident points.
    """
    score = max(0.0, _safe_float(candidate.get("score")))
    patch = max(0.0, min(1.0, _safe_float(candidate.get("v24_patch_prior", _patch_prior(candidate)))))
    v1 = 1.0 if _safe_float(candidate.get("detector_v1")) > 0.5 else 0.0
    v2 = 1.0 if _safe_float(candidate.get("detector_v2")) > 0.5 else 0.0
    agreement = 1.0 if _safe_float(candidate.get("detector_agreement")) > 0.5 else 0.0
    tile = 1.0 if _safe_float(candidate.get("v24_tile_probe")) > 0.5 else 0.0
    return (
        0.18 * min(score / 15.0, 2.0)
        + 0.18 * patch
        + 0.18 * v1
        + 0.14 * v2
        + 0.20 * agreement
        + 0.12 * tile
    )


class V26Config:
    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self.path = Path(path)
        self.values = dict(DEFAULT_CONFIG)
        self._mtime: float | None = None
        self._last_check = 0.0
        self.reload(force=True)

    def reload(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_check < 1.0:
            return
        self._last_check = now
        try:
            mtime = self.path.stat().st_mtime
        except Exception:
            mtime = None
        if not force and mtime == self._mtime:
            return
        self._mtime = mtime

        values = dict(DEFAULT_CONFIG)
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    values.update(loaded)
        except Exception as exc:
            print(f"[DETECTOR-V2.6] config load failed, defaults kept: {exc}")
        self.values = values

    def snapshot(self) -> dict[str, Any]:
        self.reload()
        return dict(self.values)


class ShotCandidateVault:
    """Bounded per-shot spatial union of candidate observations.

    V2.4/V2.5 required repeated evidence before carrying a point. Real data
    showed that the true candidate is often visible in only one detector frame.
    The vault instead quantises the camera plane into small cells and keeps one
    representative hypothesis per cell for the duration of the shot. This
    preserves recall while bounding memory and downstream candidate count.
    """

    def __init__(self) -> None:
        self.shots: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _cell(candidate: dict[str, Any], cell_px: float) -> tuple[int, int]:
        return (
            int(math.floor(_safe_float(candidate.get("camera_x")) / cell_px)),
            int(math.floor(_safe_float(candidate.get("camera_y")) / cell_px)),
        )

    def reset(self) -> None:
        self.shots.clear()

    def _prune(self, now: float, cfg: dict[str, Any], active_shot_id: int) -> None:
        max_age = max(0.5, _safe_float(cfg.get("shot_vault_max_age_s", 2.2), 2.2))
        keep_shots = max(2, _safe_int(cfg.get("shot_vault_keep_shots", 6), 6))

        for shot_id, state in list(self.shots.items()):
            if shot_id == active_shot_id:
                continue
            if now - _safe_float(state.get("last_ts"), now) > max_age:
                self.shots.pop(shot_id, None)

        if len(self.shots) > keep_shots:
            ordered = sorted(
                self.shots.items(),
                key=lambda item: _safe_float(item[1].get("last_ts")),
                reverse=True,
            )
            keep = {shot_id for shot_id, _state in ordered[:keep_shots]}
            keep.add(active_shot_id)
            for shot_id in list(self.shots):
                if shot_id not in keep:
                    self.shots.pop(shot_id, None)

    def observe(
        self,
        shot_id: int,
        candidates: list[dict[str, Any]],
        frame_ts: float,
        cfg: dict[str, Any],
    ) -> dict[str, Any] | None:
        if shot_id <= 0:
            return None
        self._prune(frame_ts, cfg, shot_id)

        state = self.shots.setdefault(
            shot_id,
            {
                "shot_id": shot_id,
                "first_ts": frame_ts,
                "last_ts": frame_ts,
                "frame_index": 0,
                "frames_with_candidates": 0,
                "cells": {},
                "observations": 0,
            },
        )
        state["last_ts"] = frame_ts
        state["frame_index"] = int(state.get("frame_index", 0)) + 1
        frame_index = int(state["frame_index"])
        cell_px = max(4.0, _safe_float(cfg.get("shot_vault_cell_px", 10.0), 10.0))
        max_cells = max(50, _safe_int(cfg.get("shot_vault_max_cells", 900), 900))

        # Never recursively re-observe carried candidates from this or older
        # persistence systems. A vault hit must correspond to a detector
        # observation on a new camera frame.
        observations = [
            dict(candidate)
            for candidate in candidates
            if _safe_float(candidate.get("v26_vault_carried")) <= 0.5
            and _safe_float(candidate.get("shot_accumulator_carried")) <= 0.5
            and _safe_float(candidate.get("candidate_bank_carried", candidate.get("v2_bank_carried", 0.0))) <= 0.5
        ]
        if observations:
            state["frames_with_candidates"] = int(state.get("frames_with_candidates", 0)) + 1

        # Only one observation per spatial cell can count in one frame.
        by_cell: dict[tuple[int, int], dict[str, Any]] = {}
        for candidate in observations:
            key = self._cell(candidate, cell_px)
            previous = by_cell.get(key)
            if previous is None or _candidate_quality(candidate) > _candidate_quality(previous):
                by_cell[key] = candidate

        state["observations"] = int(state.get("observations", 0)) + len(by_cell)
        cells: dict[tuple[int, int], dict[str, Any]] = state["cells"]
        for key, candidate in by_cell.items():
            quality = _candidate_quality(candidate)
            entry = cells.get(key)
            if entry is None:
                cells[key] = {
                    "candidate": dict(candidate),
                    "quality": quality,
                    "hits": 1,
                    "first_frame": frame_index,
                    "last_frame": frame_index,
                    "first_ts": frame_ts,
                    "last_ts": frame_ts,
                    "v1_hits": int(_safe_float(candidate.get("detector_v1")) > 0.5),
                    "v2_hits": int(_safe_float(candidate.get("detector_v2")) > 0.5),
                    "tile_hits": int(_safe_float(candidate.get("v24_tile_probe")) > 0.5),
                    "agreement_hits": int(_safe_float(candidate.get("detector_agreement")) > 0.5),
                }
                continue

            # Same cell, new camera frame. Hits are useful metadata even though
            # V2.6 does NOT require repeated hits to carry the point.
            if int(entry.get("last_frame", -1)) != frame_index:
                entry["hits"] = int(entry.get("hits", 1)) + 1
                entry["last_frame"] = frame_index
                entry["last_ts"] = frame_ts
                entry["v1_hits"] = int(entry.get("v1_hits", 0)) + int(
                    _safe_float(candidate.get("detector_v1")) > 0.5
                )
                entry["v2_hits"] = int(entry.get("v2_hits", 0)) + int(
                    _safe_float(candidate.get("detector_v2")) > 0.5
                )
                entry["tile_hits"] = int(entry.get("tile_hits", 0)) + int(
                    _safe_float(candidate.get("v24_tile_probe")) > 0.5
                )
                entry["agreement_hits"] = int(entry.get("agreement_hits", 0)) + int(
                    _safe_float(candidate.get("detector_agreement")) > 0.5
                )

            # Because all members of the cell are within about 14 px of each
            # other, choosing the strongest representative does not sacrifice
            # 42 px recall but yields a cleaner descriptor for ranking.
            if quality > _safe_float(entry.get("quality")):
                entry["candidate"] = dict(candidate)
                entry["quality"] = quality

        if len(cells) > max_cells:
            # Spatial coverage first: newest/repeated/stronger entries survive,
            # but only after each macro area has had a chance to contribute.
            chosen = self._select_entries(state, max_cells, cfg)
            keep_ids = {id(entry) for entry in chosen}
            cells = {
                key: entry
                for key, entry in cells.items()
                if id(entry) in keep_ids
            }
            state["cells"] = cells

        return state

    def _select_entries(
        self,
        state: dict[str, Any],
        limit: int,
        cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        entries = list(state.get("cells", {}).values())
        if len(entries) <= limit:
            return entries

        macro_px = max(40.0, _safe_float(cfg.get("shot_vault_macro_cell_px", 120.0), 120.0))
        buckets: dict[tuple[int, int], deque[dict[str, Any]]] = {}
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            candidate = entry.get("candidate", {})
            key = (
                int(math.floor(_safe_float(candidate.get("camera_x")) / macro_px)),
                int(math.floor(_safe_float(candidate.get("camera_y")) / macro_px)),
            )
            grouped[key].append(entry)

        for key, values in grouped.items():
            values.sort(
                key=lambda entry: (
                    min(3, int(entry.get("hits", 1))),
                    int(entry.get("last_frame", 0)),
                    _safe_float(entry.get("quality")),
                ),
                reverse=True,
            )
            buckets[key] = deque(values)

        selected: list[dict[str, Any]] = []
        keys = list(buckets)
        # Round-robin prevents a textured corner from consuming the whole vault.
        while keys and len(selected) < limit:
            next_keys: list[tuple[int, int]] = []
            for key in keys:
                bucket = buckets[key]
                if bucket and len(selected) < limit:
                    selected.append(bucket.popleft())
                if bucket:
                    next_keys.append(key)
            keys = next_keys
        return selected

    def build_output(
        self,
        shot_id: int,
        current: list[dict[str, Any]],
        frame_ts: float,
        cfg: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        state = self.shots.get(shot_id)
        if not isinstance(state, dict):
            return [dict(candidate) for candidate in current], {
                "cells": 0,
                "carried": 0,
                "frames": 0,
            }

        cells: dict[tuple[int, int], dict[str, Any]] = state.get("cells", {})
        cell_px = max(4.0, _safe_float(cfg.get("shot_vault_cell_px", 10.0), 10.0))
        carry_slots = max(0, _safe_int(cfg.get("shot_vault_carried_slots", 420), 420))
        output_limit = max(len(current), _safe_int(cfg.get("shot_vault_output_limit", 680), 680))
        dedupe_radius = max(1.0, _safe_float(cfg.get("shot_vault_dedupe_radius_px", 4.5), 4.5))
        score_decay = max(0.5, min(1.0, _safe_float(cfg.get("shot_vault_score_decay", 0.98), 0.98)))

        output = [dict(candidate) for candidate in current]

        # Annotate current candidates with vault history.
        for candidate in output:
            key = self._cell(candidate, cell_px)
            entry = cells.get(key)
            if not isinstance(entry, dict):
                continue
            candidate["v26_vault_hits"] = float(entry.get("hits", 1))
            candidate["v26_vault_seen_frames"] = float(entry.get("hits", 1))
            candidate["v26_vault_carried"] = 0.0
            candidate["v26_vault_age_s"] = 0.0

        current_positions = [
            (_safe_float(candidate.get("camera_x")), _safe_float(candidate.get("camera_y")))
            for candidate in output
        ]
        # Select exactly the carry budget with macro-cell round-robin.  Do not
        # globally re-sort afterwards: that would let one recently textured
        # region consume the whole carry budget and defeat spatial diversity.
        options = self._select_entries(state, min(len(cells), carry_slots), cfg)

        carried = 0
        for entry in options:
            if carried >= carry_slots or len(output) >= output_limit:
                break
            best = dict(entry.get("candidate", {}))
            if not best:
                continue
            cx = _safe_float(best.get("camera_x"))
            cy = _safe_float(best.get("camera_y"))
            if any(math.hypot(px - cx, py - cy) < dedupe_radius for px, py in current_positions):
                continue

            age = max(0.0, frame_ts - _safe_float(entry.get("last_ts"), frame_ts))
            best["v26_vault_carried"] = 1.0
            best["v26_vault_hits"] = float(entry.get("hits", 1))
            best["v26_vault_seen_frames"] = float(entry.get("hits", 1))
            best["v26_vault_age_s"] = float(age)
            best["v26_vault_first_frame"] = float(entry.get("first_frame", 1))
            best["v26_vault_last_frame"] = float(entry.get("last_frame", 1))
            best["v26_vault_quality"] = float(entry.get("quality", 0.0))
            best["score"] = score_decay * _safe_float(best.get("score"), 0.0)
            output.append(best)
            current_positions.append((cx, cy))
            carried += 1

        return output, {
            "cells": len(cells),
            "carried": carried,
            "frames": int(state.get("frame_index", 0)),
            "frames_with_candidates": int(state.get("frames_with_candidates", 0)),
            "observations": int(state.get("observations", 0)),
            "cells_hits_ge_2": sum(1 for entry in cells.values() if int(entry.get("hits", 1)) >= 2),
            "cells_hits_ge_3": sum(1 for entry in cells.values() if int(entry.get("hits", 1)) >= 3),
        }

    def snapshot_candidates(self, shot_id: int) -> list[dict[str, Any]]:
        state = self.shots.get(shot_id)
        if not isinstance(state, dict):
            return []
        return [dict(entry.get("candidate", {})) for entry in state.get("cells", {}).values() if entry.get("candidate")]



def patch_candidate_generator_v26(cls: type) -> None:
    if bool(getattr(cls, "_detector_v26_extension_installed", False)):
        return

    original_init = cls.__init__
    original_generate = cls.generate
    original_record = cls.record_funnel_evaluation
    original_empty = cls.record_empty_evaluation
    original_reset = cls.reset_runtime_state

    def init_wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._v26_config = V26Config()
        self._v26_vault = ShotCandidateVault()

    def generate_wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_generate(self, *args, **kwargs)
        try:
            cfg = self._v26_config.snapshot()
            if not bool(cfg.get("enabled", True)) or not bool(cfg.get("shot_vault_enabled", True)):
                return result

            telemetry = dict(getattr(result, "telemetry", {}) or {})
            shot_id = _safe_int(telemetry.get("shot_id"), 0)
            if shot_id <= 0:
                return result
            frame_ts = _safe_float(telemetry.get("frame_ts", kwargs.get("frame_ts", time.time())), time.time())
            current = [dict(candidate) for candidate in list(result.candidates)]
            self._v26_vault.observe(shot_id, current, frame_ts, cfg)
            output, vault_stats = self._v26_vault.build_output(shot_id, current, frame_ts, cfg)

            telemetry.update({
                "schema_version_v26": "2.6",
                "v26_vault_cells": vault_stats.get("cells", 0),
                "v26_vault_carried": vault_stats.get("carried", 0),
                "v26_vault_output_count": len(output),
                "v26_vault_frames": vault_stats.get("frames", 0),
            })

            scanner = kwargs.get("scanner")
            gt = getattr(scanner, "_detector_v2_ground_truth", None) if scanner is not None else None
            if isinstance(gt, dict) and _safe_int(gt.get("shot_id"), 0) == shot_id:
                record = getattr(self, "_diagnostics", {}).get(shot_id)
                if not isinstance(record, dict):
                    record = _ensure_diag_record(self, scanner, shot_id)
                if isinstance(record, dict):
                    gx = _safe_float(gt.get("camera_x"))
                    gy = _safe_float(gt.get("camera_y"))
                    v26 = record.setdefault("v26", {})
                    vault_candidates = self._v26_vault.snapshot_candidates(shot_id)
                    vault_nearest = _nearest(vault_candidates, gx, gy)
                    final_nearest = _nearest(output, gx, gy)
                    old_vault = v26.get("vault_best_nearest_px")
                    old_final = v26.get("final_best_nearest_px")
                    if vault_nearest is not None:
                        v26["vault_best_nearest_px"] = (
                            vault_nearest if old_vault is None else min(_safe_float(old_vault, 1e9), vault_nearest)
                        )
                    if final_nearest is not None:
                        v26["final_best_nearest_px"] = (
                            final_nearest if old_final is None else min(_safe_float(old_final, 1e9), final_nearest)
                        )
                    v26["max_vault_cells"] = max(_safe_int(v26.get("max_vault_cells")), _safe_int(vault_stats.get("cells")))
                    v26["max_vault_carried"] = max(_safe_int(v26.get("max_vault_carried")), _safe_int(vault_stats.get("carried")))
                    v26["max_output_count"] = max(_safe_int(v26.get("max_output_count")), len(output))
                    v26["frames_seen_by_vault"] = max(_safe_int(v26.get("frames_seen_by_vault")), _safe_int(vault_stats.get("frames")))

            return type(result)(candidates=output, telemetry=telemetry)
        except Exception as exc:
            scanner = kwargs.get("scanner")
            if bool(getattr(scanner, "shot_diag_enabled", False)):
                print(f"[DETECTOR-V2.6] vault fallback: {exc}")
            return result

    def _record_v26(self: Any, scanner: Any, gt_xy: Any, raw_hotspots: Any) -> None:
        if scanner is None:
            return
        gt = getattr(scanner, "_detector_v2_ground_truth", None)
        if not isinstance(gt, dict):
            return
        shot_id = _safe_int(gt.get("shot_id"), 0)
        if shot_id <= 0:
            return
        record = getattr(self, "_diagnostics", {}).get(shot_id)
        if not isinstance(record, dict):
            record = _ensure_diag_record(self, scanner, shot_id)
        if not isinstance(record, dict):
            return

        if gt_xy is None:
            gt_xy = (_safe_float(gt.get("camera_x")), _safe_float(gt.get("camera_y")))
        gx, gy = float(gt_xy[0]), float(gt_xy[1])
        raw = [dict(candidate) for candidate in list(raw_hotspots or [])]
        carried = [candidate for candidate in raw if _safe_float(candidate.get("v26_vault_carried")) > 0.5]
        funnel = record.setdefault("evaluation_funnel", {})
        funnel["raw_v26_vault_carried_count"] = len(carried)
        funnel["raw_v26_vault_carried_nearest_px"] = _nearest(carried, gx, gy)
        funnel["raw_v26_final_count"] = len(raw)
        funnel["raw_v26_final_nearest_px"] = _nearest(raw, gx, gy)

        state = self._v26_vault.shots.get(shot_id, {})
        cells = state.get("cells", {}) if isinstance(state, dict) else {}
        vault_candidates = self._v26_vault.snapshot_candidates(shot_id)
        funnel["v26_vault_summary"] = {
            "cells": len(cells) if isinstance(cells, dict) else 0,
            "frames": _safe_int(state.get("frame_index")) if isinstance(state, dict) else 0,
            "frames_with_candidates": _safe_int(state.get("frames_with_candidates")) if isinstance(state, dict) else 0,
            "observations": _safe_int(state.get("observations")) if isinstance(state, dict) else 0,
            "cells_hits_ge_2": sum(1 for entry in cells.values() if _safe_int(entry.get("hits"), 1) >= 2) if isinstance(cells, dict) else 0,
            "cells_hits_ge_3": sum(1 for entry in cells.values() if _safe_int(entry.get("hits"), 1) >= 3) if isinstance(cells, dict) else 0,
            "gt_nearest_px": _nearest(vault_candidates, gx, gy),
        }

        # V5 runtime diagnostics are attached here because this is the exact
        # synthetic shot/funnel record written to detector JSONL.
        try:
            from src.engine.ai.runtime import get_ai_runtime

            runtime = get_ai_runtime()
            v5 = getattr(runtime, "_v26_last_v5_diagnostic", None)
            if isinstance(v5, dict):
                funnel["v26_ranker_v5"] = dict(v5)
        except Exception:
            pass

    def record_wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        original_record(self, *args, **kwargs)
        try:
            _record_v26(self, kwargs.get("scanner"), kwargs.get("gt_xy"), kwargs.get("raw_hotspots", []))
        except Exception:
            pass

    def empty_wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        original_empty(self, *args, **kwargs)
        try:
            _record_v26(self, kwargs.get("scanner"), kwargs.get("gt_xy"), [])
        except Exception:
            pass

    def reset_wrapped(self: Any) -> None:
        original_reset(self)
        try:
            self._v26_vault.reset()
        except Exception:
            pass

    cls.__init__ = init_wrapped
    cls.generate = generate_wrapped
    cls.record_funnel_evaluation = record_wrapped
    cls.record_empty_evaluation = empty_wrapped
    cls.reset_runtime_state = reset_wrapped
    cls._detector_v26_extension_installed = True


def apply_detector_v26_extension() -> None:
    from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2

    patch_candidate_generator_v26(CandidateGeneratorV2)
    print("[DETECTOR-V2.6] shot candidate vault installed")


__all__ = [
    "ShotCandidateVault",
    "V26Config",
    "apply_detector_v26_extension",
    "patch_candidate_generator_v26",
]
