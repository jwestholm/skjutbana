# Installed AI and detector stack, 2026-09-07

This is a source/install audit plus inspection of saved physical provenance and
local model registries. Import/install failures are fail-open: source presence is
not proof a component ran on a particular historical event. The traces do not save
an exhaustive installed-method inventory. The current physical settings snapshot
says `mode=advisory`, async detector and async AI enabled.

## Live shot ownership

`main.py` installs V2.22.3 shot priority/context, V2.22.4 async CV/AI,
V2.22.5 fast proposals and local confirmation, V2.22.6 frame-unique tracking/audio
telemetry, V2.24 object ROI mapping, and V2.25 region/freshness/novelty wrappers.
`ai/bootstrap.py` installs the resolver and V2.22.1/.2 crop and cleanup wrappers.
Camera/scenes import hooks install the hybrid candidate generator and V6–V9
extensions. This import-time monkey-patch chain has real composition risk: the
V2.22.6 tracking replacement dropped the V2.22.4 timestamp wrapper, reproduced
and corrected during this session.

The live sequence is:

1. Audio PCM chunk passes absolute, noise-relative, crest and cooldown gates.
2. Main-thread dispatch creates an `AudioShotEvent` and captures PRE/context.
3. CV worker proposes V1/V2/retained/bank candidates inside calibrated camera ROI.
4. V2.22.2 cleanup/candidate limits apply; XY is restored from crop to full camera.
5. V2.22.5 checks local PRE→POST evidence on a later frame; one FULL rescue is legal.
6. Tracks accumulate distinct camera observations. Association and readiness gates
   select a track. Context-enabled V2.25 selection additionally uses registered
   freshness and cross-shot novelty. Missing context falls through to prior policy.
7. AI emission wrapper calls its chooser. V2.22.4 advisory mode returns passthrough
   while heavy AI work runs asynchronously. The detector emits through HitInput.
8. Camera→screen/game mapping and frozen shot-id object collision follow emission.

The simple biathlon image does not supply evidence of active object HitRegions:
trace context is `UNAVAILABLE` and candidate/debug fields lack V2.25 region
selection evidence. Do not assume V2.25 novelty corrected these global-path shots.

## Components and authority

| System | Installed role / possible effect | Evidence and limitations |
|---|---|---|
| Deterministic V1 + CandidateGeneratorV2, banks/vaults | Generate physical candidates; directly affect live pool | Physical traces prove good candidates can survive; also distant persistent artifacts. Registered shifts have very low/negative response in these shots. |
| V2.22.1 ROI, V2.22.2 cleanup | Crop, map XY, remove/demote candidates; direct live effect | Physical coordinates and saved cleanup telemetry; complete raw boundary unavailable in historical traces. |
| V2.22.5 local confirmation / V2.22.6 tracking | Physical eligibility and temporal evidence; direct live effect | Event 3 correct candidate has local-confirm flag and two hits. `state=confirmed` is assigned by emission, not by local proof. |
| V2.25 region/freshness/novelty | Conditional live authority when frozen object regions exist; FULL rescue bypass | Earlier documented bridge acceptance; current global image sessions do not establish acceptance of this authority path. |
| ShotResolverV222 | Cluster/evidence/external votes, advisory or potential AI chooser in authority modes | Async advisory is passthrough on current live path. `off/train_only/advisory` do not authorize AI XY overrides. |
| AIRuntime memory + ranking_v22 + pairwise ranking_v3 | Feature/memory and learned rank blending; training/advisory and possible authority-mode input | Existing online/F2 learning, no current physical success claim. These rankings cannot undo detector emission in advisory mode. |
| V4/V5 | Older alternative runtime extensions and models | Superseded by V6 hook on normal startup; retained for historical reproduction/fallback, not deleted. |
| V6 / V2.8 hypotheses | Wider recall pool and gated learned ranker; can change AIRuntime candidate order if its own validation gates pass | `auto_override_enabled` exists in config, but this is rank authority inside AI, not permission to override advisory camera hits. Not independently physically validated. |
| V7 | Offline pairwise linear cross-validation and shadow analysis | Saved recommendation rejects future authority; reported Top1@20 gain −19.355 percentage points on its historical evaluation. |
| V8 | Monotonic percentile ensemble, shadow only | Saved recommendation fails; development Top1@20 gain −22.222 pp, confirmation gain 0. These are artifact claims, not rerun comparable benchmark metrics. |
| V9 | Physical monotonic listwise shadow ranker | Saved artifact reports development/confirmation gains +18.519/+25 pp, but `enough_data=false`, authority recommendation false. Different historical protocol; not equivalent to tonight's held-out experiment. |
| V2.13–.18 patch/new-hole/pair systems | Historical image/patch experts and offline experiments | Available code/model artifacts are not evidence they execute per live shot. Their feature, patch, and proposal domains differ. |
| V2.23 base linear/MLP trainer | Grouped candidate training, F2 shadow scoring, research registry | Existing champion file says no live authority. Earlier documented fresh-domain failure; ten-iteration experiment ended with no primary holdout improvement. |
| V2.23.2 dense proposal / V2.23.3 reducer | Offline full-frame dense candidates, learned reduction then final ranker | Documentation reports high dense oracle on projected F2; reducer registry has no research cascade champion. Dense recall is not live recall. |
| V2.23.4 patch model | Offline candidate patch classification and final ranking | Registry: bootstrap/research gates false, no champion, live authority false. |
| V2.23.5 registered evidence | Registration and temporal evidence patch features/classification | Registry: bootstrap/research gates false, no champion. Distinct from live V2.25 registered authority. |
| V2.23.6 direct heatmaps | Offline direct localization, F2-triggered research cycle | Registry: bootstrap signal/direct path/research gates false, no champion, no live authority. |
| Canonical challenger | Frozen V2.23 linear listwise model, 17 shared physical features, SHADOW only | Six reproducible historical trials; no Top1@20 gain. Post-decision retained-pool shadow comparison is mixed on seven real physical shots. |

## Consolidation decision

One canonical adapter, `src/engine/ai/canonical_challenger.py`, reuses the existing
V2.23 model and feature extractor. It has no `apply`/emission API. It validates the
manifest's SHADOW mode and model hashes, returns all scores, complete order, Top-N,
features, camera XY, manifest hash and score margin. Confidence is explicitly
uncalibrated; no hit probability is fabricated. The feature schema uses
`extract_physical_features` aliases and zero for absent values. Missing-feature
coverage/domain shift must be inspected; zero does not prove absent physical
phenomena. Feature transforms are shared between training and inference.

The chosen artifact is
`evaluation_runs/overnight_20260907/ranking/challenger.json`, pinning the
`regularized/` model. The model is **OFFLINE_CHALLENGER**. It is not
PHYSICAL_SHADOW_VALIDATED, ELIGIBLE_FOR_ADVISORY or ELIGIBLE_FOR_AUTHORITY.
The next physical helper explicitly enables diagnostic capture of this manifest.
Nothing registers it in an existing champion registry or changes live AI mode.

Future traces freeze detector selection and retained pool at emission boundary.
The challenger evaluates that frozen pool only at trace finalization and records
its hash and timing semantics. It does not restrict itself to locally confirmed
candidates; thus “would rank closer” is not “would have emitted this hit.” Misses
use the last observed retained pool and say so. Historical smoke traces do not
have a separately observed selected track; export no longer infers it from emitted
XY. Raw/filtered/confirmed availability is never reconstructed from downstream
presence alone.

## Duplication and next cleanup

V3–V9 and V2.23 repeatedly solve candidate ranking with incompatible schemas and
validation protocols. Dense/reducer/patch/registered/heatmap systems additionally
solve proposal generation or image localization, so they cannot be folded into a
candidate ranker without explicit adapters and coordinate/provenance contracts.
Keep those historical artifacts. New comparisons should go through the canonical
shadow interface, conditional metrics and physical trace workflow, not another
version-number ranker. A later cleanup can replace monkey-patch installation with
explicit composition, backed by installed-stack tests; doing that simultaneously
with a detector hypothesis would make physical attribution harder.
