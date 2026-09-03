# V2.23.5 Test Plan

## Installation

```bash
python3 -m automation.v2235_selftest
python3 -m automation.v2235_verify_install
```

Expected: all PASS. Verify checks that V2.21 direct-map generation and V2.21.5 dense proposal remain importable and that F2 autotrain points to V2.23.5.

## Existing-session preparation

```bash
python3 -m automation.v2235_prepare --session latest
```

Expected for the current data: 100 registered evidence banks, no errors. This is an offline recomputation of physical maps; it can take materially longer than V2.23.4 patch extraction.

## Bootstrap learnability

```bash
python3 -m automation.v2235_train --quick --no-prepare
python3 -m automation.v2235_status
```

Compare directly with the V2.23.4 bootstrap baseline:

- R128 ~0.13
- R512 ~0.30
- median rank ~949

The V2.23.5 decision gate is R512 >=0.70, R128 >=0.45, median rank <=200. Do not run another F2 session if this gate fails.

## Fresh-domain test

Only after bootstrap passes, capture a new independent F2 session, compile proposals and V2.23.5 evidence, then train with the newest session reserved as fresh domain. The domain session must not participate in fitting, hard-negative mining, or trial selection.

## Safety

- candidate proposal generation is GT-free
- GT anchors are training-only
- 6..42 px candidates are neutral for patch training
- protected holdouts remain outside automatic model selection
- live authority remains unchanged / NO
