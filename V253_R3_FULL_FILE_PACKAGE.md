# V2.25.3-r3 – Full-file package repair

This is a packaging-only correction to V2.25.3. The cross-thread readiness and cross-shot novelty runtime remain V2.25.3.

## Why r3 exists

Earlier cumulative packages used helper scripts to patch menu/settings/documentation. A test fixture menu escaped into a release package and produced `Unsupported menu.json version`. The release workflow is therefore changed.

## Delivery contract

- No `automation/` directory.
- No `prepare` or `apply` step.
- No menu entry fragments.
- `content/menu.json` is a complete schema version 1 file.
- `src/engine/settings.py` is intentionally not included; V2.25.3 does not require replacing it.
- All Python and Markdown files in the archive are complete replacement files.

## Install

From the repository root, unzip with overwrite and start normally. No Python install helper is run.

```bash
unzip -o skjutbana_v2.25.3-r3_full_files_only.zip -d .
python3 main.py
```

## Physical test

Run Game Objects Test (V2.25.3) and repeat the same five-shot sequence used for V2.25.2: breakable, living, no-shoot, moving living, rear target/penetration.
