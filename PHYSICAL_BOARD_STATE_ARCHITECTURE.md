# Physical board research architecture and inventory

The 2026-09-09 design map supplied by the user is a research direction, not
authorization to replace the engine. This inventory accompanies the D01 pass.
The objective is **100% correct emitted physical results; 95% is the minimum**.
No central class or live policy was added. S03 remains entirely untouched.

## Existing responsibilities and reuse points

| Responsibility | Current implementation / history | Consequence |
|---|---|---|
| Static scene and physical surface references | `HitScanner.scene_reference_gray`, `surface_reference_gray`, reference capture and `_rebuild_artifact_mask()`; `ai_training.py` white/black reference capture and projector-response debug map. History includes `79e7df6` and `485092d`. | Consolidate ownership later; distinguish calibration appearance from immediate causal PRE. Static reference alone cannot prove a new impact. |
| Holes and recent physical state | `HitScanner.known_holes`, tracking, V2 banks/vaults, V2.22.2 novelty cleanup, V2.25.3 shared novelty state. | Multiple partial histories already exist. Inventory update/reset rules before consolidation. A hole is context; nearby/reopened hits must remain legal. |
| Coordinates and bounds | `input/hit_input.py`, camera ArUco calibrator, calibration scenes, saved viewport/scanport/content rectangles, V2.22.1 analysis geometry, V2.24 object-to-camera transforms. | Reuse these transforms; do not create a parallel coordinate engine. A future normalized physical board plane needs a saved physical boundary and round-trip tests. |
| Board/panel motion | Existing `common_verifier.py`, `accuracy_verifier.py`, motion-residual and temporal offline audits; new `clean_physical_change.py` experiments. | Whole-frame registration exists. Split-crop/flow tests are research evidence; no persistent motion model exists in the inspected paths. |
| Audio visualization | `audio/audio_peak_detector.py:get_waveform_snapshot`, `scenes/audio_peak_settings.py`; history `f9d5cae` (audio settings), `4989c96` (audio hit switch). Peak/RMS/noise/crest diagnostics already exist. | Extend the existing calibration flow if physical evidence later justifies weapon identification. No physical WeaponProfile was found by current-code and all-history string/path searches. |
| Projectile identity | `game_objects/projectiles.py:ProjectileProfile` explicitly models gameplay damage/penetration. | This is not a measured physical weapon/audio profile. Do not silently reuse game parameters as physical truth. |
| Game objects | Existing model/manager/geometry/events and frozen HitRegions. | Preserve downstream ownership of game rules and fictional z order; no game implementation in this pass. |

The targeted all-history deletion search for weapon/audio/calibration/surface/
board paths found no removed replacement to revive. This is a bounded inventory,
not proof that every historical idea was implemented or absent.

## Direction after measurement

If consolidation is justified, one owner can expose physical bounds, surface
revision, static/projected appearance, holes/repairs, uncertainty and temporal
stability. Existing buffers and canonical transforms should supply it. Paper,
cardboard, camera/projector geometry, lighting or major repair changes require
revision-aware history. The user's workflow repairs before/between games and
accumulates holes during play; never assume repair after every shot.

Visual change, board motion, board memory and weapon/event context are separate
evidence families. Correlated scores within a family are not independent votes.
Known edges, holes, tape and motion are context, never hard vetoes. A calibrated
confidence gate and independent truth are prerequisites for persistent learning;
an emitted `state=confirmed` or an uncalibrated 0.95 score is insufficient.

Future motion learning should ask whether regional movement predicts impact
location across sessions better than empirical spatial priors. Keep recent and
long-term models separate and test drift/revision changes chronologically before
enabling adaptation. Lighting and support conditions must be observed metadata,
not inferred weapon or pressure labels.

D01 has saved camera frames/crop evidence but `calibration=UNAVAILABLE` and no
saved projector transform or physical seam. The new experiments therefore use
camera PRE structure and two overlapping crop halves. They cannot claim exact
projected-edge attribution, normalized physical board coordinates or calibrated
panel motion. Do not substitute today's settings for missing capture provenance.

See [D01_PHYSICAL_FINDINGS.md](D01_PHYSICAL_FINDINGS.md) for measured clean-change,
motion and ranking results, their limits and the proposed next capture.
