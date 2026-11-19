# Evidence chain — inventory and plan (2026-11-18)

Purpose: one place that lists every computation that judges "is this candidate a real NEW hole?",
where it runs, whether it changes decisions, and how the chain should grow (many evidence families,
a precision/latency-driven scheduler, content-aware skipping, a calculated fusion and a learned
selector). Inventory from a code study on 2026-11-18 (file:function names verified by search;
line numbers drift). Rules that apply throughout (AGENTS.md): evidence is context, never an
automatic veto; missing evidence is UNKNOWN, not negative; one hypothesis at a time; measure first.

## 1. Inventory

Status: **LIVE-D** = live, affects the decision; **LIVE-diag** = live, diagnostics only;
**SHADOW** = computed live, not used; **OFF** = flag default off; **OFFLINE** = research/replay only.

| # | family / name | where | measures | status |
|---|---|---|---|---|
| 1 | V2 change maps (absdiff, darkening, z-score vs temporal noise, DoG, saliency, persistence, artifact weight, edge penalty) | camera/candidate_generator_v2.py (generate, _candidate_features) | registered + photometric PRE→POST change | LIVE-D (proposals + score) |
| 2 | V2.24 patch descriptor + patch prior (core/ring/outer, compactness, isotropy, bipolar, local_snr, ringness, 3×3 grid) | camera/detector_v24_extension.py (_patch_descriptor, _patch_prior) | morphology of the change patch | LIVE-D (tile probe, cluster quality) |
| 3 | Shot accumulator | detector_v24_extension.py (_update_shot_accumulator) | repeated observation ≤7 px over ≤0.7 s | LIVE-D |
| 4 | V2.26 shot vault | camera/detector_v26_extension.py | keeps distinct hypotheses (recall) | LIVE-D |
| 5 | pre_shot_change | ai/runtime.py, hit_scanner.py, v24 | patch PRE→POST change /64 | LIVE-D (feeds 6, H9a) |
| 6 | V2.22.2 novelty/stale filter, ridge suppression | camera/analysis_filters_v2222.py | low PRE novelty demotes; stale at known hole removed (unless fresh ≥5); horizontal ridges | LIVE-D |
| 7 | Known-hole context factor | candidate_generator_v2._apply_known_hole_penalty, hit_scanner, winner_selection_flags.known_hole_factor | distance to registered holes | LIVE-D (score factor) |
| 8 | V2.22.5 local confirmation (ConfirmMetricsV2225) | shot_fast_v2225.py (local_confirm_candidates_v2225) | centre/ring/compact/peak/darkening vs shot PRE on a later frame | LIVE-D (gate + ≤5 bonus) |
| 9 | V2.22.6 frame-unique hits | shot_track_v2226.py | temporal persistence over distinct frames | LIVE-D (readiness) |
| 10 | V2.25.2 registered freshness | shot_region_freshness_v252.py | registered disc/ring, z, dark fraction, PRE noise | LIVE-D, object-context shots only |
| 11 | V2.25.3 cross-shot novelty | shot_cross_thread_novelty_v253.py | recurrence of a location across shots | LIVE-D, object-context only |
| 12 | Object physical query | camera/object_physical_query.py | disc/ring, persistence, compactness, freshness beyond PRE envelope, polarity | LIVE-D for OBJECT_HIT |
| 13 | Precision flash | precision_flash.py (find_new_holes) | flash-lit new dark blobs vs flash reference | LIVE-D when EXACT / ZONE+series |
| 14 | **Hole morphology, V2-map source** (H11) | winner_selection_flags.attach_h11 / h11_snr_map | matched filter with the real airgun+AR15 stamp profile on V2 change maps, annulus-noise SNR | OFF (`SKJUTBANA_HOLE_LIKENESS_RANK=1` ranks by it) |
| 15 | Hole morphology, patch source | hole_morphology_evidence.py (measure) | same filter on raw PRE/CUR 61×61 patches, local gain/offset | SHADOW (`SKJUTBANA_HOLE_EVIDENCE=1`) |
| 16 | Ranking factors H6/H7a/H7b/H9a/H9b, H12 MF proposals | winner_selection_flags.py | uncapped score, shape, polarity, snap, soft known-hole, extra proposals | OFF (all measured: no effect / mixed / rejected) |
| 17 | Ranker v6 (23 features) | ai/ranker_v6.py, ranker_v6_extension.py | learned fusion over hypothesis features | SHADOW until its gate (≥120 validation shots + margins) |
| 18 | Confirmation-selection shadow, canonical challenger | ai/confirmation_selection_shadow.py, ai/canonical_challenger.py | frozen selectors | LIVE-diag (physical trace) |
| 19 | Nightly bench scorer | automation/bench/train.py | learned scorer over ALL numeric candidate fields of exported candidates | OFFLINE (OFFLINE_REPLAY_ONLY_UNVALIDATED) |
| 20 | EvidenceBundle / EvidenceOverlay (V2.12) | offline/evidence.py (build_evidence) | full-map temporal consensus, persistent z, local contrast, darkening, absdiff | OFFLINE |
| 21 | CommonTemporalHistory / CommonVerifier | offline/common_verifier.py | causal onset/step, witness lifecycle with revocation, coherent change | OFFLINE (append ~1.3 ms/frame) |
| 22 | Clean physical change, accuracy verifier | offline/clean_physical_change.py, offline/accuracy_verifier.py | onset, persistence, PRE variability, motion compensation | OFFLINE |
| 23 | Research families (temporal collision, locality audit/ablation, patch verifier, structure, retention, temporal impact) | automation/physical_ai_*.py, evidence_*_research.py | various | OFFLINE (locality ablation rejected) |

Not live anywhere: **IMPACT / NO_IMPACT** (offline only: offline/physical_event_truth.py,
heatmap_model_v2236), **revocation** (offline witness lifecycle only).

## 2. Context that could drive content-aware skipping

No live per-region "dark / textured / moving content" map exists. Building blocks:
- `hit_scanner.scene_reference_gray` / `surface_reference_gray` and `artifact_suppression_mask`
  (|scene − surface| > 12 = projector-responsive surface; static, captured at calibration/AI-test start).
- V2 per-pixel `temporal_noise` / `pre_instability` and `edge_weights` (live, full ROI).
- Per-candidate local noise: hole-morphology annulus noise, v252 `pre_noise`, v24 `local_snr`.
- Precision flash lit mask (reference > 25).
- Offline only: render history (what was on screen), optical-lab projector response.
Note (ROADMAP, transfer falsification 2026-10-18): per-pixel render→camera prediction failed, so use
coarse context (brightness/texture/motion classes), not raw render subtraction.

## 3. Plan (agreed with Codex 2026-11-18; extend existing code, no parallel MasterVerifier)

1. **Evidence contract on every family**: value(s), valid/missing reason, family, version, causal time
   window, measured cost. Done for hole morphology (14, 15); next: wrap 8, 10, 12 outputs the same way.
   Both morphology producers use `CandidateEvidence` in `src/engine/candidate_evidence.py`.
   Their legacy callers supply images without causal timestamps and measure cost for a batch.
   The per-observation window and cost therefore remain unknown. Replay exports keep the
   existing source-specific keys.
2. **Ledger** = the existing `bench replay/synth --export-candidates` rows (every numeric candidate
   field) + per-shot stage timings (`DetectorJobResultV2224.stages_ms`). Extend, don't duplicate.
3. **Scheduler/budget per request**: a game declares `hit_precision` (exists) and a latency class
   (new, e.g. FAST ≈0.4–0.7 s for Skyttegrav-type games, NORMAL ≈1 s, ACCURATE ≤2 s for EXACT/ZONE).
   Cheap families decide early; expensive ones run only when the budget allows, otherwise in shadow.
   Late evidence never moves an already-reported gameplay hit.
4. **Content-aware skipping**: a coarse per-region context (brightness from the PRE frame, texture
   from edge_weights, motion from temporal_noise) decides which families are applicable; e.g. skip
   hole morphology where PRE brightness is too low to show a hole and record `missing=dark_content`.
5. **Calculated fusion (non-AI)**: the current selector (onset, then score) plus measured families
   (H11 today). Grow it family by family with paired replay.
6. **Learned selector (AI)**: the nightly bench scorer already trains on every numeric candidate
   field — turning a family on in the nightly synth/replay makes it a feature automatically. Ranker v6
   remains the live SHADOW learner. Promotion only through the existing gates.
7. **New families, priority**: IMPACT/no-hole (abstention → UNKNOWN), onset/birth and persistence/
   revocation as LIVE ports of the offline CommonVerifier, morphology (done), board-motion nuisance.

## 4. Measured so far (details: AUDIT_FINDINGS_2026-11-18.md §14)
- H11 (hole morphology, V2-map source) paired replay: projected 48→77/253, physical AR15 15→15/22;
  live AI-test 30/79 vs 14/78; +~80 ms hit latency.
- Patch source (15): 22–23/78 on the AI-test session (vs 29) — single PRE frame too noisy; kept as shadow.
- Control arm: the gain comes from the matched filter, not from demoting unmeasured (v24 tile) tracks.
