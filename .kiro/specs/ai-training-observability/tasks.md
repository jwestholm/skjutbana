# Implementation Plan: AI Training Observability

## Overview

This plan implements structured observability, unified reporting, and experiment controls for the AI auto-training system. The core change introduces a `RoundRecord` dataclass as the single source of truth for all per-round metrics, replacing the scattered `auto_stats` dict. All reporting surfaces (Auto_Report, FunnelTracker summary, CSV export) derive from this list. Secondary changes add AI guess pre-facit stats, round-ID state logging, block-per-100 trend tables, candidate count statistics, a configurable `candidate_limit`, and pluggable sampling mode strategies.

Tasks are ordered so foundational changes come first (RoundRecord, settings) before dependent changes (report, CSV, sampling).

## Tasks

- [x] 1. Add RoundRecord dataclass and CSV enhancement to diagnostics.py
  - [x] 1.1 Define `RoundRecord` dataclass in `src/engine/ai/diagnostics.py`
    - Add `from dataclasses import dataclass` import
    - Define `@dataclass` class `RoundRecord` with all fields from the design: `round_id`, `timestamp`, `gt_screen_x`, `gt_screen_y`, `gt_camera_x`, `gt_camera_y`, `candidate_count_raw`, `candidate_count_ranked`, `found`, `top1_correct`, `top3_correct`, `nearest_dist`, `ai_guess_camera_x`, `ai_guess_camera_y`, `ai_guess_dist_to_gt`, `ai_guess_correct`, `sampling_mode`, `match_radius_px`, `background_mode`
    - _Requirements: 1.6, 2.1, 5.5_

  - [x] 1.2 Enhance `FunnelTracker.save_csv()` to accept optional `round_records` parameter
    - When `round_records` is provided, write one row per `RoundRecord` with all fields including AI guess pre-facit fields (`ai_guess_correct`, `ai_guess_dist_to_gt`, `ai_guess_camera_x`, `ai_guess_camera_y`, `candidate_count_raw`, `candidate_count_ranked`)
    - Merge existing `ShotDiagnostics` data with `RoundRecord` data by index
    - Preserve existing behavior when `round_records` is not provided
    - _Requirements: 2.4, 2.5_

  - [ ]* 1.3 Write property test: CSV export round-trip preserves all RoundRecord fields
    - **Property 4: CSV export round-trip preserves all RoundRecord fields**
    - Generate random lists of `RoundRecord` using Hypothesis, export to CSV, parse back, verify same row count and all fields present
    - **Validates: Requirements 2.4, 2.5**

- [x] 2. Add `candidate_limit` and `sampling_mode` to AIRuntime settings
  - [x] 2.1 Modify `AIRuntime.__init__()` in `src/engine/ai/runtime.py`
    - Add `"candidate_limit": 200` and `"sampling_mode": "center_bias"` to `DEFAULT_SETTINGS`
    - Add a `candidate_limit` property that reads from `self.settings["candidate_limit"]`, casts to `int`, and clamps to range `[1, 2000]` with fallback to 200 on error
    - Add a `sampling_mode` property that reads from `self.settings["sampling_mode"]` with fallback to `"center_bias"`
    - _Requirements: 6.1, 6.2, 6.5, 6.6, 7.1, 7.2_

  - [ ]* 2.2 Write property test: candidate_limit clamping
    - **Property 8: Candidate limit clamping**
    - Generate arbitrary integers with Hypothesis, verify clamping to `[1, 2000]`
    - **Validates: Requirements 6.5, 6.6**

- [x] 3. Wire `candidate_limit` through bootstrap.py → hit_scanner.py
  - [x] 3.1 Add `self.candidate_limit: int = 200` instance variable to `HitScanner.__init__()` in `src/engine/camera/hit_scanner.py`
    - Replace the hardcoded `candidates[:150]` slice in `_detect_frame_candidates` with `candidates[:self.candidate_limit]`
    - _Requirements: 6.3, 6.4_

  - [x] 3.2 Modify `wrapped_update` in `src/engine/ai/bootstrap.py`
    - After `runtime.observe_scanner(self)`, add `self.candidate_limit = runtime.candidate_limit`
    - _Requirements: 6.3_

- [x] 4. Checkpoint — Ensure all tests pass
  - All files pass diagnostics with zero errors.

- [x] 5. Refactor AITrainingScene to use RoundRecord list as single source of truth
  - [x] 5.1 Replace `auto_stats` dict with `round_records` list in `src/engine/scenes/ai_training.py`
    - Add `self.round_records: list[RoundRecord] = []` and `self.current_round_id: int = 0`
    - Import `RoundRecord` from `src.engine.ai.diagnostics`
    - Update `_reset_auto_stats()` to clear `round_records` and reset `current_round_id`
    - _Requirements: 2.1_

  - [x] 5.2 Add `_build_round_record()` method to replace `_record_detection_stats()`
    - Increment `self.current_round_id`
    - Compute all `RoundRecord` fields from current state: GT coordinates from `auto_target_screen_xy` + `project_screen_point()`, candidate counts from `hit_scanner.last_candidates` and `self.ranked_candidates`, detection results (found/top1/top3/nearest_dist), AI guess pre-facit from `ranked_candidates[0]` before training click
    - Append the new `RoundRecord` to `self.round_records`
    - _Requirements: 1.6, 2.1, 2.2, 5.5_

  - [x] 5.3 Add round-ID state logging via `_log_round_state(round_id, state_name)`
    - Print `[ROUND {round_id}] {state_name}` to stdout
    - Insert calls at key state transitions: `round_started` (in `_start_auto_iteration`), `hole_created` (after `add_hole`), `shot_triggered` (in `_fire_pending_synthetic_shot`), `candidates_ranked` (in `_on_shot_detected`), `selection_made` (in `_on_training_click`), `round_completed` (after `_do_clean_capture`)
    - _Requirements: 3.1, 3.2_

  - [x] 5.4 Add session summary logging at session end
    - When auto-training completes, log total `current_round_id`, `len(round_records)`, and `len(runtime.funnel.shots)`
    - If any of the three counts differ, log a `[MISMATCH]` warning
    - _Requirements: 3.3, 3.4, 2.6_

  - [ ]* 5.5 Write property test: sequential round IDs
    - **Property 5: Sequential round IDs**
    - Generate random session lengths with Hypothesis, simulate round_id assignment, verify sequence `[1, 2, ..., N]` with no gaps or duplicates
    - **Validates: Requirements 3.1**

- [x] 6. Rewrite `_build_auto_report()` to compute all stats from `round_records`
  - [x] 6.1 Implement core stats computation from `round_records`
    - Compute `found`, `top1`, `top3`, `missed` counts and percentages from `round_records`
    - Compute average nearest distance from records with candidates
    - Remove all references to `self.auto_stats` dict
    - _Requirements: 2.2_

  - [x] 6.2 Add AI guess pre-facit statistics section
    - Compute `ai_guess_correct` count and percentage
    - Compute `ai_guess_avg_dist_to_gt` over records with `candidate_count_ranked > 0`
    - _Requirements: 1.1, 1.2, 1.3_

  - [ ]* 6.3 Write property test: count-based report statistics match RoundRecord aggregation
    - **Property 1: Count-based report statistics match RoundRecord aggregation**
    - Generate random `RoundRecord` lists, compute stats independently, verify they match the report function output
    - **Validates: Requirements 1.1, 1.2, 2.2**

  - [ ]* 6.4 Write property test: average AI guess distance computation
    - **Property 2: Average AI guess distance computation**
    - Generate random `RoundRecord` lists, verify average distance matches independent computation over records with candidates
    - **Validates: Requirements 1.3**

- [x] 7. Checkpoint — Ensure all tests pass
  - All files pass diagnostics with zero errors.

- [x] 8. Add block statistics per 100 shots to report
  - [x] 8.1 Implement block statistics computation in `_build_auto_report()`
    - Only include block table when `len(round_records) > 100`
    - Compute `ceil(len(records) / 100)` blocks, each with `found`, `top1`, `top3`, `ai_guess_correct` counts and `avg_dist`
    - Handle final partial block correctly
    - Format as table rows in the report lines
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ]* 8.2 Write property test: block statistics correctness
    - **Property 6: Block statistics correctness**
    - Generate random `RoundRecord` lists, verify block count equals `ceil(len/100)`, each block's stats match independent computation over the corresponding slice
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**

- [x] 9. Add candidate count statistics to report
  - [x] 9.1 Implement candidate count statistics in `_build_auto_report()`
    - Compute average, min, max of `candidate_count_raw` across all records
    - Compute zero-candidate round count
    - Compute counts of rounds with >50, >100, >200 candidates
    - Format and append to report lines
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ]* 9.2 Write property test: candidate count statistics correctness
    - **Property 7: Candidate count statistics correctness**
    - Generate random `RoundRecord` lists, verify avg/min/max/zero/threshold counts match independent computation
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4**

- [x] 10. Add first-100 vs last-100 comparison to report
  - [x] 10.1 Implement first-100 vs last-100 comparison in `_build_auto_report()`
    - Only include when `len(round_records) >= 200`
    - Compute `found`, `top1`, `ai_guess_correct`, `avg_dist` for `records[0:100]` and `records[-100:]`
    - Format as side-by-side comparison in report lines
    - _Requirements: 1.4, 1.5_

  - [ ]* 10.2 Write property test: first-100 vs last-100 comparison presence and correctness
    - **Property 3: First-100 vs last-100 comparison presence and correctness**
    - Generate random `RoundRecord` lists of varying lengths, verify section present iff `len >= 200`, verify stats match independent computation over respective slices
    - **Validates: Requirements 1.4, 1.5**

- [x] 11. Implement sampling mode functions
  - [x] 11.1 Add four sampling mode functions to `src/engine/scenes/ai_training.py`
    - Implement `_sample_center_bias(vp, margin=12)` — current uniform random behavior
    - Implement `_sample_uniform(vp, margin=12)` — uniform random across full viewport
    - Implement `_sample_edge_bias(vp, margin=12)` — ≥60% of targets within 15% of viewport edge
    - Implement `_sample_corners(vp, margin=12)` — distribute across four quadrants, each within 25% of corner
    - Create `SAMPLING_MODES` dict mapping mode names to functions
    - Handle small viewport edge cases (< 2*margin → clamp to center)
    - _Requirements: 7.3, 7.4, 7.5, 7.6_

  - [x] 11.2 Replace `_choose_auto_screen_point()` with sampling mode dispatch
    - Read `self.runtime.sampling_mode` to select the function from `SAMPLING_MODES`
    - Fall back to `center_bias` with a warning for unrecognized modes
    - _Requirements: 7.1, 7.2, 7.7_

  - [ ]* 11.3 Write property test: sampling modes produce valid viewport points
    - **Property 9: Sampling modes produce valid viewport points**
    - Generate random `pygame.Rect` viewports, verify all generated points fall within viewport bounds and respect margin for `center_bias` and `uniform`
    - **Validates: Requirements 7.3, 7.4**

  - [ ]* 11.4 Write property test: edge bias sampling distribution
    - **Property 10: Edge bias sampling distribution**
    - Generate random viewports, sample 200+ points with `edge_bias`, verify ≥60% fall within 15% of viewport edge
    - **Validates: Requirements 7.5**

  - [ ]* 11.5 Write property test: corners sampling distribution
    - **Property 11: Corners sampling distribution**
    - Generate random viewports, sample 200+ points with `corners`, verify points distributed across all four quadrants and each within 25% of its corner
    - **Validates: Requirements 7.6**

  - [ ]* 11.6 Write property test: unrecognized sampling mode fallback
    - **Property 12: Unrecognized sampling mode fallback**
    - Generate random strings that are not recognized mode names, verify fallback to `center_bias` behavior producing valid viewport points with margin
    - **Validates: Requirements 7.7**

- [x] 12. Wire CSV export with round_records at session end
  - [x] 12.1 Update session-end code to pass `round_records` to `FunnelTracker.save_csv()`
    - In `_build_auto_report()`, call `self.runtime.funnel.save_csv("autotrain", round_records=self.round_records)`
    - _Requirements: 2.4_

- [x] 13. Final checkpoint — Ensure all tests pass
  - All 5 modified files pass diagnostics with zero errors. Property-based tests (marked *) are optional and can be added later.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using Hypothesis
- Unit tests validate specific examples and edge cases
- All 12 correctness properties from the design are covered by property test tasks (Properties 1–12)
- The `auto_stats` dict is fully replaced by computing from `round_records` — no parallel counters
