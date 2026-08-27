# V2.19 Media Bank — acquisition and split discipline

The purpose of the media bank is **visual challenge diversity**, not collection size by itself.

## What to add

Aim for a balanced local bank across:

- paintings / posters / paper targets,
- photographs of people / faces / clothing,
- forest / grass / stone / water,
- indoor rooms / walls / furniture,
- city / buildings / roads,
- text-heavy screens / UI / terminals,
- screenshots from games or game-like assets you are allowed to use locally,
- checker/grid/stripe/high-frequency textures,
- dark scenes,
- bright/washed-out scenes,
- low-contrast scenes,
- animations and video with moving objects,
- video with whole-camera motion/panning.

A useful first bank is **hundreds to a few thousand distinct source images** plus **dozens of genuinely different video/animation clips**.  Ten thousand near-duplicate frames are much less useful than 1,000 distinct sources.

## Where

Recommended local location:

```text
content/ai/media_bank/
```

It is also fine to keep a larger bank elsewhere on disk and pass the directories to:

```bash
python3 -m automation.offline_v219_media_index /media/photos /media/games /media/video
```

Do not commit a large external media bank to Git just because the indexer can read it.

## Sidecar metadata

Optional sidecar next to an asset:

```text
scene_001.jpg
scene_001.jpg.json
```

Example:

```json
{
  "family_id": "wikimedia-example-12345",
  "category": "painting",
  "license": "CC0",
  "source_url": "https://source.example/item/12345"
}
```

For extracted frames from the same original video, give every frame the **same `family_id`**.  Better yet, index the original video directly so the whole clip is automatically one split unit.

## Suitable sources

Prefer media whose local use/provenance is clear, for example:

- your own photographs and screenshots,
- this project's own game/art assets,
- public-domain / CC0 material,
- appropriately licensed Creative Commons material with required attribution retained in metadata,
- open-game/open-film assets where the individual asset license permits the intended local use.

Keep `source_url` and `license` in the sidecar for downloaded material.  The audit intentionally reports unknown provenance.

## Do not leak holdout

Never manually copy the same image into separate train/holdout directories and assume the test is independent.

V2.19 assigns a whole source/family to one deterministic split and records SHA-256 + perceptual hash.  Run:

```bash
python3 -m automation.offline_v219_media_audit
```

before treating a media holdout as frozen.

For video, **the clip is the split unit**, not the frame.

## Synthetic versus physical truth

Media-bank scenarios are excellent for:

- generating millions of cheap training cases,
- finding background-specific failure classes,
- hard-negative mining,
- learning moving-background robustness,
- preparing projector-aware residual models.

They are not the final acceptance test.  A model only earns game authority after passing completely unseen physical projector/camera sessions.
