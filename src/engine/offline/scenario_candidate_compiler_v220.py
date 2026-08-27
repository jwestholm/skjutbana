from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .candidate_pack_v216 import CandidateCaptureConfigV216, CandidateShadowRecorderV216
from .evidence import EvidenceConfig, build_evidence, extract_overlay_candidates, merge_candidate_sources
from .live_detector_replay import LiveHybridReplayDetector
from .scenario_generator_v220 import OfflineScenarioGeneratorV220


class ScenarioCandidateCompilerV220:
    def __init__(
        self,
        *,
        generator: OfflineScenarioGeneratorV220,
        output_root: Path = Path("content/ai/candidate_synthetic_v220"),
        candidate_limit: int = 384,
        include_overlay: bool = True,
        save_full_frames: bool = False,
    ) -> None:
        self.generator = generator
        self.output_root = Path(output_root)
        self.candidate_limit = max(16, int(candidate_limit))
        self.include_overlay = bool(include_overlay)
        self.detector = LiveHybridReplayDetector()
        self.evidence_config = EvidenceConfig.from_file()
        base = CandidateCaptureConfigV216.load()
        self.capture_config = replace(
            base,
            data_root=str(self.output_root),
            max_candidates=self.candidate_limit,
            include_raw_extras=True,
            save_gt_patches=True,
            save_full_frames=bool(save_full_frames),
            max_post_frames=max(1, int(self.generator.profile.post_frames)),
        )

    @staticmethod
    def _median_pre(frames: list[np.ndarray]) -> np.ndarray:
        return np.median(np.stack(frames, axis=0).astype(np.float32), axis=0).astype(np.uint8)

    def compile(self, *, first_seed: int, count: int, split: str, session_id: str | None = None) -> dict[str, Any]:
        count = max(1, int(count))
        session_id = session_id or f"v220_{split}_{int(first_seed)}_{count}_{int(time.time())}"
        recorder = CandidateShadowRecorderV216(
            self.capture_config,
            background=f"v220_media_{split}",
            benchmark_seed=int(first_seed),
            sampling_mode="v220_generated_world",
            session_id=session_id,
        )
        rows = []
        for offset in range(count):
            seed = int(first_seed) + offset
            scenario = self.generator.generate(seed, split=split)
            gt = scenario.spec.gt_camera_xy
            detected = self.detector.detect(
                pre_frames=scenario.pre_frames,
                post_frames=scenario.post_frames,
                known_holes=scenario.spec.known_holes,
                ground_truth=gt,
                candidate_limit=self.candidate_limit,
            )
            ranked = [dict(candidate) for candidate in detected.candidates]
            raw = list(ranked)
            overlay_count = 0
            if self.include_overlay:
                bundle = build_evidence(scenario.pre_frames, scenario.post_frames, config=self.evidence_config)
                overlay = extract_overlay_candidates(bundle.fused, config=self.evidence_config)
                overlay_count = len(overlay)
                raw = merge_candidate_sources(
                    [("current_detector", ranked), ("physical_fusion_v212", overlay)],
                    merge_radius_px=5.0,
                    limit=max(self.candidate_limit * 2, 700),
                )
            extra = {
                "v220_generated": True,
                "v220_scenario": scenario.spec.to_dict(),
                "known_holes_before_shot": [{"camera_x": float(x), "camera_y": float(y)} for x, y in scenario.spec.known_holes],
                "old_hole_count": len(scenario.spec.old_holes),
                "media_split": split,
                "media_category": scenario.spec.media_category,
                "challenge_tags": scenario.spec.challenge_tags,
                "overlay_candidates": overlay_count,
                "physical_acceptance_data": False,
                "rgb_observed_output": True,
            }
            saved = recorder.capture_shot(
                round_id=offset + 1,
                raw_candidates=raw,
                ranked_candidates=ranked,
                pre_gray=self._median_pre(scenario.pre_frames),
                recent_pre_gray=scenario.recent_pre_frame,
                recent_pre_timestamp=0.90,
                post_gray=scenario.post_frames[-1],
                post_frames=[(frame, 1.0 + (index + 1) / 30.0) for index, frame in enumerate(scenario.post_frames)],
                gt_camera_xy=gt,
                gt_screen_xy=None,
                match_radius_px=42.0,
                extra_metadata=extra,
            )
            rows.append({
                "seed": seed,
                "saved": bool(saved.get("saved")),
                "candidate_count": int(saved.get("candidate_count") or 0),
                "media_category": scenario.spec.media_category,
                "challenge_tags": scenario.spec.challenge_tags,
            })
            if (offset + 1) % 10 == 0 or offset + 1 == count:
                print(f"V2.20 candidate compile: {offset + 1}/{count} worlds")
        summary = recorder.finalize()
        result = {
            "schema_version": "2.20",
            "session_id": session_id,
            "split": split,
            "first_seed": int(first_seed),
            "count": count,
            "capture": summary,
            "rows": rows,
            "saved": sum(1 for row in rows if row["saved"]),
            "warning": "Generated candidate packs are training/offline data, never physical holdout.",
        }
        report_path = Path(summary.get("root") or self.output_root / "sessions" / session_id) / "v220_compile_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        result["report_path"] = str(report_path)
        return result
