# Skjutbana V2.23.0 delta

**Base:** install on top of the currently working V2.22.6 checkout/delta chain.

**Scope:** unified training/model pipeline. V2.22 live authority remains unchanged.

## Install

From repo root, unpack this ZIP over the project. Then run:

```bash
git status --short
git diff --check
python3 -m automation.v2230_selftest
python3 -m automation.v2230_verify_install
python3 -m automation.v2230_audit
```

Optional documentation merge:

```bash
python3 -m automation.v2230_apply_docs
```

The updater is append-only/idempotent for `CURRENT_STATE.md`, `HIT_DETECTION_PLAN.md`, and `AI_CONTEXT.md`.

## First offline verification

```bash
python3 -m automation.v2230_status
python3 -m automation.v2230_train --quick
python3 -m automation.v2230_status
```

The quick trainer uses development + validation only. Any protected holdout remains untouched for automatic selection.

## First F2 verification

Start normally:

```bash
python3 main.py
```

Expected startup line:

```text
[V2.23.0] unified training/model pipeline installed (capture + shadow champion/challenger; live authority unchanged)
```

Run a small F2 session first. V2.23 saves actual candidate groups and schedules a quick background challenger run after >=10 captured rounds. Existing F2/SimpleAIMemory behavior is left intact for comparison.

## Offline unattended training

```bash
python3 -m automation.v2230_autotrain --hours 1 --quick
```

Later, once verified:

```bash
python3 -m automation.v2230_autotrain --hours 8
```

## Safety/measurement rules

- V2.23.0 never grants live authority.
- Automatic model selection never evaluates protected holdout.
- Storage-forced GT rows are excluded.
- No GT/rank/policy/model-output shortcuts enter the physical feature vector.
- Proposal/oracle recall and ranker accuracy are reported separately.
- The known loading/mechanical audio false-trigger is parked as a TODO.
