# V2.20.2 generated validation dataset summary

**Source report:** `v220_compile_report.json` supplied after the validation candidate compile.

## Compile facts

- schema version: 2.20,
- split: `validation`,
- first seed: `9000001`,
- count: 100,
- saved: 100,
- capture errors: 0,
- candidate cap: 384,
- max post frames: 3,
- candidate patch size: 64,
- session: `v220_validation_9000001_100_1787833691`,
- capture remains shadow-only.

Every row in this compile report saved exactly 384 candidates, giving 38,400 candidate rows in total.

## Media distribution

- `game`: 51 scenarios,
- `photo_or_image`: 49 scenarios.

## Challenge-tag counts

- `camera_captured_hole_appearance`: 100,
- `rgb_observed_output`: 100,
- `incomplete_known_holes`: 93,
- `dynamic_background`: 51,
- `hard_edges`: 51,
- `shared_camera_state`: 49,
- `dense_old_holes`: 43,
- `near_edge`: 19,
- `near_old_hole`: 19,
- `hole_in_hole`: 1.

The later V2.18 inspect reported 100/100 groups with a candidate <=20 px and <=42 px, so this set is appropriate for testing ranking independently of candidate-recall failure.

## Interpretation

The set contains useful challenge variation, but the media/category distribution is still small compared with the intended eventual world bank, and the near-perfect V2.18 result confirms that it should be treated as a controlled generated-domain validation set, not a physical-realism benchmark.
