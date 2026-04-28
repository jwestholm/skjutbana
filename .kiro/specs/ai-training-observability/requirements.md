# Requirements Document

## Introduction

This feature improves observability, reporting, and experiment controls for the AI training system in the digital shooting range application. The focus is strictly on debug/observability tooling — no AI tuning or hole-visual changes. The goal is to give the operator full visibility into what the AI is doing during auto-training sessions, unify all stat sources into a single source of truth, and make key parameters configurable without code changes.

## Glossary

- **Auto_Report**: The graphical on-screen report displayed by `_build_auto_report()` in `AITrainingScene` after an auto-training session completes.
- **Round**: A single iteration in an auto-training session: one synthetic hole is placed, one shot is triggered, candidates are generated and ranked, a training click is performed, and the result is recorded.
- **Round_Record**: A per-round data structure that captures all metrics for a single round, serving as the single source of truth for all reporting.
- **Funnel_Tracker**: The `FunnelTracker` class in `diagnostics.py` that accumulates `ShotDiagnostics` across a session.
- **AI_Guess**: The AI's top-1 ranked candidate before the training click/facit is applied.
- **Ground_Truth**: The known correct position of the synthetic hole during auto-training, derived from the screen coordinate projected to camera space.
- **Match_Radius**: The configurable pixel distance threshold (`click_match_radius_px`) used to determine whether a candidate is "correct" relative to Ground_Truth.
- **Block**: A contiguous group of 100 rounds within a training session, used for trend analysis.
- **Candidate_Limit**: The maximum number of raw hotspot candidates retained by HitScanner after contour detection, currently hardcoded to 150 in `hit_scanner.py`.
- **Settings_File**: The JSON configuration file at `content/ai/settings.json` used to store AI runtime parameters.
- **Sampling_Mode**: A strategy for choosing where synthetic holes are placed during auto-training (e.g., center_bias, uniform, edge_bias, corners).
- **HitScanner**: The `HitScanner` class in `hit_scanner.py` responsible for hotspot detection from camera frames.
- **AIRuntime**: The `AIRuntime` class in `runtime.py` that manages AI scoring, training, and funnel diagnostics.
- **AITrainingScene**: The `AITrainingScene` class in `ai_training.py` that orchestrates auto-training sessions.
- **CSV_Report**: The per-shot CSV file exported by `FunnelTracker.save_csv()` to `content/ai/reports/`.

## Requirements

### Requirement 1: AI Guess Pre-Facit Statistics in Auto Report

**User Story:** As an operator, I want to see how well the AI's top-1 guess matched the ground truth BEFORE the training click, so that I can evaluate whether the AI is learning to pick the right candidate on its own.

#### Acceptance Criteria

1. WHEN an auto-training session completes, THE Auto_Report SHALL display the count and percentage of rounds where AI_Guess was within Match_Radius of Ground_Truth (labeled `ai_guess_correct`).
2. WHEN an auto-training session completes, THE Auto_Report SHALL display the count of rounds where any ranked candidate was within Match_Radius of Ground_Truth (labeled `ai_guess_within_match_radius`).
3. WHEN an auto-training session completes, THE Auto_Report SHALL display the average distance in pixels from AI_Guess to Ground_Truth across all rounds that had candidates (labeled `ai_guess_avg_dist_to_gt`).
4. WHEN an auto-training session with 200 or more rounds completes, THE Auto_Report SHALL display a comparison of the first 100 rounds versus the last 100 rounds for `found`, `top1`, `ai_guess_correct`, and `avg_dist` metrics.
5. WHEN an auto-training session with fewer than 200 rounds completes, THE Auto_Report SHALL omit the first-100 vs last-100 comparison section.
6. THE Round_Record SHALL store the AI_Guess pre-facit fields (`ai_guess_correct`, `ai_guess_dist_to_gt`, `ai_guess_camera_x`, `ai_guess_camera_y`) directly, not only in sidecar metadata.

### Requirement 2: Single Source of Truth for All Statistics (Top Design Priority)

**User Story:** As an operator, I want all statistics (on-screen report, funnel summary, CSV export) to derive from the same per-round data, so that I never see conflicting numbers across different outputs.

#### Acceptance Criteria

1. THE AITrainingScene SHALL maintain a list of Round_Record objects as the single authoritative data source for all per-round metrics during an auto-training session.
2. WHEN the Auto_Report is built, THE AITrainingScene SHALL compute all displayed statistics exclusively from the Round_Record list.
3. WHEN the Funnel_Tracker summary is generated, THE Funnel_Tracker SHALL derive its counts from the same ShotDiagnostics objects that correspond one-to-one with Round_Record entries.
4. WHEN the CSV_Report is exported, THE Funnel_Tracker SHALL write one row per Round_Record, including all fields present in the Round_Record.
5. THE CSV_Report SHALL include the AI_Guess pre-facit fields (`ai_guess_correct`, `ai_guess_dist_to_gt`, `candidate_count`) for each round.
6. IF the Round_Record list and the Funnel_Tracker shot count differ, THEN THE AITrainingScene SHALL log a warning identifying the mismatch.

### Requirement 3: Round-ID and State Logging

**User Story:** As a developer, I want each auto-training round to log its state transitions with a unique round ID, so that I can debug iteration count mismatches (e.g., 1000/999/996 discrepancies).

#### Acceptance Criteria

1. THE AITrainingScene SHALL assign a sequential integer round_id to each round, starting at 1 for the first round in a session.
2. WHEN a round transitions between states, THE AITrainingScene SHALL log the transition to stdout in the format `[ROUND {round_id}] {state_name}` where state_name is one of: `round_started`, `hole_created`, `shot_triggered`, `candidates_generated`, `candidates_ranked`, `selection_made`, `review_finished`, `round_completed`.
3. WHEN an auto-training session completes, THE AITrainingScene SHALL log a session summary line containing the total round_id count, the Round_Record list length, and the Funnel_Tracker shot count.
4. IF any of the three counts in the session summary differ, THEN THE AITrainingScene SHALL log a `[MISMATCH]` warning with the differing values.

### Requirement 4: Block Statistics per 100 Shots

**User Story:** As an operator, I want to see statistics broken down per block of 100 shots in the auto report, so that I can observe whether the AI is improving during a training run.

#### Acceptance Criteria

1. WHEN an auto-training session with more than 100 rounds completes, THE Auto_Report SHALL display a block statistics table with one row per Block of 100 rounds.
2. THE block statistics table SHALL include the following columns for each Block: block number, `found` count, `top1` count, `top3` count, `ai_guess_correct` count, and `avg_dist` in pixels.
3. WHEN the total number of rounds is not evenly divisible by 100, THE Auto_Report SHALL include a final partial block containing the remaining rounds.
4. WHEN an auto-training session has 100 or fewer rounds, THE Auto_Report SHALL omit the block statistics table.

### Requirement 5: Candidate Count Statistics

**User Story:** As an operator, I want to see statistics about how many candidates the detector generates per shot, so that I can identify rounds with too few or too many candidates.

#### Acceptance Criteria

1. WHEN an auto-training session completes, THE Auto_Report SHALL display the average number of candidates per round.
2. WHEN an auto-training session completes, THE Auto_Report SHALL display the minimum and maximum candidate count observed across all rounds.
3. WHEN an auto-training session completes, THE Auto_Report SHALL display the count of zero-candidate rounds.
4. WHEN an auto-training session completes, THE Auto_Report SHALL display the count of rounds with more than 50, more than 100, and more than 200 candidates.
5. THE Round_Record SHALL store the raw candidate count (before noise rejection) and the ranked candidate count (after noise rejection and AI ranking) for each round.

### Requirement 6: Configurable Candidate Limit via Settings

**User Story:** As an operator, I want to change the maximum number of raw hotspot candidates without editing source code, so that I can experiment with different limits (50, 100, 200, 500) between training sessions.

#### Acceptance Criteria

1. THE AIRuntime SHALL read a `candidate_limit` integer value from Settings_File on startup.
2. WHEN `candidate_limit` is not present in Settings_File, THE AIRuntime SHALL use a default value of 200.
3. WHEN HitScanner truncates the raw candidate list, THE HitScanner SHALL use the `candidate_limit` value provided by AIRuntime instead of a hardcoded constant.
4. THE `candidate_limit` value SHALL be applied at the point where `last_candidates` is sliced in `_detect_frame_candidates`, replacing the hardcoded `[:150]` slice.
5. WHEN `candidate_limit` is set to a value less than 1, THE AIRuntime SHALL clamp the value to 1.
6. WHEN `candidate_limit` is set to a value greater than 2000, THE AIRuntime SHALL clamp the value to 2000.

### Requirement 7: Edge/Corner Sampling Toggle (Prepared but Off)

**User Story:** As an operator, I want auto-training sampling modes prepared in the code so that I can test different spatial distributions in future sessions without writing new code.

#### Acceptance Criteria

1. THE AIRuntime SHALL read a `sampling_mode` string value from Settings_File on startup.
2. WHEN `sampling_mode` is not present in Settings_File, THE AIRuntime SHALL use `"center_bias"` as the default value.
3. THE AITrainingScene SHALL implement four Sampling_Mode strategies for `_choose_auto_screen_point`: `center_bias` (current behavior with uniform random), `uniform` (uniform random across viewport), `edge_bias` (higher probability near viewport edges), and `corners` (targets placed near the four viewport corners).
4. WHEN `sampling_mode` is set to `"center_bias"`, THE AITrainingScene SHALL use the current uniform random placement within the viewport with a 12-pixel margin.
5. WHEN `sampling_mode` is set to `"edge_bias"`, THE AITrainingScene SHALL place at least 60% of targets within 15% of the viewport edge.
6. WHEN `sampling_mode` is set to `"corners"`, THE AITrainingScene SHALL distribute targets evenly across the four quadrants of the viewport, each within 25% of the respective corner.
7. WHEN `sampling_mode` is set to an unrecognized value, THE AITrainingScene SHALL fall back to `"center_bias"` and log a warning.
