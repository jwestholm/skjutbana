# Detector V2.19 — Offline Scenario + Media World Engine

## Why V2.19 exists

V2.18 proved that the same-shot listwise objective is much closer to the real problem than V2.17's pointwise NEW/NOT-NEW classification:

- on the single seed-65432 development subset, V2.18 moved union Top-1 from **6.67% current / 0% V2.17 to 13.33%**, Top-3 to **18.33%**, and median GT rank to **5.5**;
- the same model did **not** beat the existing V2.16/V9 confirmation baseline and did not generalise cleanly to confirmation/holdout;
- learned offset refinement improved the development oracle but degraded confirmation/holdout;
- all 100 physical/projected shots still belonged to one session.

The interpretation is not "listwise failed".  It is:

> The objective can learn the desired ordering, but one 100-shot physical session is too narrow to justify long autonomous optimization.

The next bottleneck is therefore **problem diversity**, not another ranking head.

V2.19 builds a deterministic offline world generator so that a seed can create a genuinely new labelled perception problem:

```text
seed N
  |
  +-- media/background source + exact frame/time
  +-- 0..n old physical holes
  +-- incomplete known_holes registry
  +-- one NEW hole + exact GT
  +-- near-old / hole-in-hole / edge challenges
  +-- camera photometric/noise variation
  +-- moving image/video/game-like background
  |
  v
PRE frames + recent PRE + POST frames
  |
  v
same live V1/V2 detector OFFLINE
  + V2.12 physical overlay
  |
  v
V2.16-compatible candidate packs
  |
  v
V2.17 / V2.18 / future trainer
```

V2.19 is **not** the 12-hour champion/challenger loop yet.  It is the world/data engine that makes such a loop meaningful.

## 1. Media bank

V2.19 indexes local images and video as *source assets*, not individual random frames.  The whole source/family is assigned to one deterministic split.

This is crucial.  Frames 100 and 101 of the same video may not become train and holdout merely because their filenames differ.

Supported first-pass media:

- images: PNG/JPEG/BMP/WebP/TIFF,
- video/animation containers readable by OpenCV: MP4/AVI/MOV/MKV/WebM/M4V/GIF where the local backend supports them,
- the existing `assets/` tree,
- a dedicated `content/ai/media_bank/` tree,
- arbitrary extra local directories supplied to the indexer.

Each manifest row records:

- source path,
- image/video kind,
- category,
- source/family ID,
- deterministic train/validation/holdout split,
- dimensions / frame count / FPS,
- source URL and license metadata when known,
- SHA-256 and a representative perceptual hash.

`offline_v219_media_audit` reports exact or near-duplicate media that crosses split boundaries.  External media with unknown provenance is allowed for local research, but the manifest makes that explicit instead of silently forgetting where the files came from.

### Recommended media challenge classes

The external bank should deliberately contain broad visual classes rather than "random internet images":

1. paintings / posters / targets,
2. faces / people / clothing,
3. nature / forest / grass / stone,
4. indoor / urban / buildings,
5. text / UI / terminals / scoreboards,
6. game screenshots and game-like art,
7. high-frequency grids / checker / stripes / textures,
8. mostly dark scenes,
9. mostly bright scenes,
10. low-contrast photographs,
11. animation / sprite motion,
12. real video clips with camera and object motion.

Whole source identities should stay in one split.  For example, one video clip belongs entirely to training or entirely to holdout.

## 2. Camera-domain hole appearance bank

A key design correction is that V2.19 does **not** have to invent every hole from a perfect procedural circle.

The shooting computer already has ~15k `content/ai/holes/synt_*` patches.  Those holes were synthetic on the projector but were then observed through the real:

```text
projector -> physical surface -> camera
```

V2.19 therefore extracts a compact residual around the known centre of a `synt_*` patch and transplants that residual onto new media backgrounds.  This preserves useful camera-domain scale, blur, edge response and optical character while changing the surrounding scene.

Important semantic rules:

- the raw centred 128x128 source is **never** used as a centred class label;
- only a compact local residual is reused;
- old-hole appearance is applied identically to PRE and POST;
- the new GT hole is added only to POST;
- the 37 real `hole_*` patches remain outside this training generator by default.

If no `synt_*` bank is available, V2.19 falls back to the existing `SyntheticHoleOverlay`, with provenance tagged so generated reports can distinguish the source.

## 3. Old holes and known holes

A generated scenario contains two separate concepts:

- `old_holes`: the true physical old holes in the generated world,
- `known_holes`: a deliberately incomplete subset made visible to the current detector/context.

This mirrors the real system: `HitScanner.known_holes` is useful but is not a perfect inventory of every physical hole already on the target.

The generator can create:

- no old holes,
- dense old-hole fields,
- a new hole near an old hole,
- rare hole-in-hole / re-hit cases,
- an incomplete known-hole registry.

Known-hole state remains a soft diagnostic/context source.  A generated hole-in-hole remains a valid new shot.

## 4. Moving media and future projector-aware residual

For still images, PRE and POST use the same projector image.

For video/game-like media, V2.19 samples a sequence of underlying media frames across the shot time.  Physical old/new holes remain fixed in camera coordinates while the displayed media may change beneath them.

The generator keeps both layers:

- `projector_pre_frames` / `projector_post_frames`: what was displayed,
- observed PRE/POST frames: displayed content plus persistent physical holes and camera effects.

This is intentional preparation for a later **projector-aware residual** source.  We will know which visual motion was expected from the rendered scene and which residual is unexplained physical change.

## 5. Resolution

V2.19 defaults to `width=0,height=0`, meaning:

1. inspect the newest V2.16 candidate-pack metadata and reuse the actual camera frame shape,
2. fall back to 3840x2160 when no captured shape exists.

This avoids accidentally benchmarking the 4K-tuned detector on a toy 640x360 world without saying so.

## 6. Candidate-pack bridge

`offline_v219_compile_candidates` runs:

```text
GeneratedScenarioV219
  -> current live V1/V2 detector via LiveHybridReplayDetector
  -> V2.12 physical overlay candidate source
  -> recall-oriented raw union
  -> existing CandidateShadowRecorderV216
```

The output therefore uses the **same V2.16 candidate-pack schema** already consumed by V2.17 and V2.18.

No new training-data format is invented.

Generated packs include explicit provenance:

- `v219_generated=true`,
- exact seed and scenario parameters,
- media source/split/category,
- old and known holes,
- challenge tags,
- `physical_acceptance_data=false`.

They may be passed to V2.17/V2.18 through `--root`, but they can never substitute for a physical holdout.

## 7. Seed discipline

The default plan reserves different seed spaces:

- generated/adaptive training: `1..8,999,999`,
- frozen generated validation: `9,000,001..9,099,999`,
- frozen generated offline holdout: `99,000,001..99,099,999`.

Exact ranges are policy, not magic numbers.  The important rule is that the overnight optimizer must not start training on a seed after it has been designated as frozen validation/holdout.

Media split and seed split are both required.  A frozen holdout seed rendered on a training-only image is not a true media holdout.

## 8. Commands

Index local media:

```bash
python3 -m automation.offline_v219_media_index
```

Add extra directories:

```bash
python3 -m automation.offline_v219_media_index \
  /path/to/photos /path/to/game_screens /path/to/videos
```

Audit leakage/provenance:

```bash
python3 -m automation.offline_v219_media_audit
```

Selftest:

```bash
python3 -m automation.offline_v219_selftest
```

Generate visible examples:

```bash
python3 -m automation.offline_v219_generate \
  --seed 1 --count 20 --split train --save-images
```

Benchmark current V1/V2 + V2.12 union on frozen generated scenarios:

```bash
python3 -m automation.offline_v219_benchmark \
  --first-seed 9000001 --seeds 100 --split validation
```

Compile generated worlds into V2.16-compatible candidate packs:

```bash
python3 -m automation.offline_v219_compile_candidates \
  --first-seed 1 --count 100 --split train
```

Those packs can then be inspected/trained with existing V2.18 tooling by supplying their root.

## 9. What V2.19 success means

V2.19 is successful when:

1. the same seed reproduces the same world exactly,
2. changing seeds produces meaningfully different problems,
3. old holes persist across PRE/POST and only the GT hole is newly introduced,
4. still, photo, game-like and moving-media challenges can all be represented,
5. media sources cannot silently leak across train/holdout,
6. generated worlds run through the **same live detector code** offline,
7. generated candidate packs are directly consumable by V2.17/V2.18,
8. failure statistics can be broken down by media category and challenge tag,
9. real physical sessions remain separately labelled as the authority.

It is **not** a success criterion that synthetic/media accuracy becomes high.  The generator exists to create useful diversity and hard cases; final acceptance remains physical.

## 10. What comes next

Once V2.19 has been smoke-tested against the shooting-PC media/hole bank, V2.20 should be the first autonomous champion/challenger trainer:

```text
fresh training seeds
 -> compile / reuse generated candidate groups
 -> train challenger
 -> fixed generated validation
 -> protected physical confirmation
 -> KEEP / REJECT
 -> hard-case scheduler changes next seed distribution
 -> repeat for N hours
```

After that, the major remaining ceiling is still candidate recall.  V2.21 should therefore introduce **direct AI proposals/heatmaps** so NEW-hole AI can propose coordinates that V1/V2 never emitted.
