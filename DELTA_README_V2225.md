# Skjutbana V2.22.5 DELTA

**Base:** working V2.22.4-r2 installation.

## Purpose

V2.22.5 removes the repeated full global detector pass from normal track confirmation and introduces a sparse live V2 proposal extractor.

### Normal path

```text
PANG -> async FAST global proposal -> candidates/hits=1
     -> next real camera frame -> LOCAL confirmation -> hits=2/3 -> HIT
```

### Rescue path

If FAST returns no usable candidate or local confirmation finds no fresh compact PRE->POST evidence, exactly one full historical high-recall extractor pass is permitted.

## Files changed/added

- `main.py` — installs V2.22.5 after V2.22.4.
- `src/engine/shot_fast_v2225.py` — runtime implementation.
- `automation/v2225_selftest.py`
- `automation/v2225_verify_install.py`
- `automation/v2225_apply_docs.py`
- `V2225_PLAN.md`
- `V2225_FAST_PROPOSAL_AND_CONFIRMATION.md`
- `V2225_TEST_PLAN.md`
- `V2225_DOC_PATCH.md`
- previous V2.22.3/V2.22.4 planning MD files are carried forward in this ZIP as requested.

## Install

Unpack over the repository root, then:

```bash
git status --short
git diff --check
python3 -m automation.v2225_selftest
python3 -m automation.v2225_verify_install
python3 -m automation.v2225_apply_docs
```

Then start:

```bash
python3 main.py
```

Expected new banner:

```text
[V2.22.5] fast proposal + local confirmation installed
```

## First physical test

AI Training, `advisory`, no F2. Shoot **one** physical shot first.

Expected new lines:

```text
[V2.22.5 FAST-EXTRACT] ...
[V2.22.5 LOCAL-CONFIRM] ...
```

For a normal confirmed shot there should not be a second expensive global `[V2.22.4 CV]` before HIT. See `V2225_TEST_PLAN.md`.
