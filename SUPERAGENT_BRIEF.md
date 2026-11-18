# Brief for a deep-work agent session (prepared 2026-11-18)

Read first: AGENTS.md (rules), EVIDENCE_CHAIN.md (evidence inventory + plan),
AUDIT_FINDINGS_2026-11-18.md (§9–§14: physical + optical results), CURRENT_STATE.md (top sections),
automation/bench/README.md (tools, flag table). Docs may be stale — verify claims against code.

## Goal
Reliable hit detection (≥95 %, target 100 % exact physical XY) for a projector range with CO2 AR15 /
pistol (holes ~22 camera px core, depth 0.05–0.19) and airgun (40 px, 0.28), within a per-game
latency budget (fast shooters ≪1 s; exact/zone games up to ~2 s).

## Current state in one paragraph
Live chain: audio peak → PRE/POST frames → FAST proposals (V2 maps) → local confirmation → tracks →
winner (onset first, then score). On clean bright paper it works (physical AR15 replay 12–15/15 visible
holes); on textured/moving projected content recall and ranking fail. H11 (hole-morphology evidence,
matched filter with real stamps; `SKJUTBANA_HOLE_LIKENESS_RANK=1`, default OFF) lifts projected
replay 48→77/253 with physical unchanged and live AI-test 14/78→30/79. Live hit latency median ~1.0 s
(p90 1.8 s) even without H11; worker ~560 ms per shot; each CO2 shot gives two audio peaks 66–94 ms
apart → two shot jobs.

## Work packages (pick in order; one hypothesis at a time; measure before/after)
1. **Latency profile and reduction** (highest value). Use `DetectorJobResultV2224.stages_ms` /
   `DetectorStageProfilerV2224` to split queue/extract/confirm/evidence per shot on the recorded
   sessions below. Measure how often the double CO2 peak spawns duplicate jobs; test coalescing that
   preserves both raw peaks and never merges genuine rapid shots (doubles ≥0.15 s apart). Target: hit
   median < 0.6 s for FAST requests without accuracy loss in paired replay.
2. **Evidence scheduler + latency classes** (EVIDENCE_CHAIN.md §3.3–3.4): game-declared latency class
   next to `hit_precision`; content-aware applicability (dark PRE region → skip hole morphology, record
   `missing=dark_content`); every family emits the evidence contract.
3. **Uniform evidence for every candidate source**: v24 tile-probe candidates currently lack
   `ev_hole_morph_snr` (handled by a measured-first split in `hit_scanner._best_track_for_event_flagged`).
   Measure them from the V2 maps or route them through the same map.
4. **No-hole / IMPACT abstention family**: low hole evidence → UNKNOWN or slower analysis (never a veto).
   Offline emission study: SNR < 6 drops 13/43 false emissions, 0/18 correct (AI-test session).
5. **Port offline CommonVerifier onset/persistence/revocation** (src/engine/offline/common_verifier.py)
   to a live evidence family with measured cost.
6. **Nightly learned selector**: the bench trainer uses all numeric candidate fields — enable the
   evidence families in nightly synth/replay exports (decide baseline handling first).

## Data and tools
- Paired replay A/B: `python3 -m automation.bench replay --use-current-settings SESSION --output DIR`
  with env arms; MIN_POST_AGE 0.15 + CAUSAL_OWNERSHIP=1 are the current live settings.
- Sessions with GT (`/data/skjutbana/source_sessions/`): physical AR15 `20261118T155321Z_04f1be81`
  (fresh paper, 15 holes + 7 empty-mag), `20261118T135339Z_da06f87a_ar15cut` (taped board);
  AI-test `20261118T162349Z_66d659ce`, `20261118T181112Z_6cb18ae7` (live H11), `20261118T123405Z_76d489ed`;
  validation (real AR15 stamps, never used for design) `20261118T1701..1704*` (pattern, moving, 3 photos).
- Measurement tools: `bench hole-likeness*`, `bench optical-holes --hole-style stamps_ar15`,
  `bench board-snapshot`, `bench calibration-check`. Results so far: `/data/skjutbana/evaluation_runs/h11_ab_20261118/`.
- Selftests: `automation/*selftest*.py` (140; run all before handing back).

## Hard rules (AGENTS.md)
No push/merge; never stage `content/ai/settings.json`, traces or evaluation artifacts; one major
hypothesis at a time, default OFF until physically validated; evidence is never an automatic veto;
missing evidence is UNKNOWN; new evaluation dirs, never overwrite baselines; projected/optical evidence
is not physical validation.
