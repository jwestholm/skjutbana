# Skjutbana V2.23.2 DELTA

Install this ZIP over an existing V2.23.1 checkout. V2.22 runtime behaviour is intentionally not changed.

V2.23.2 adds:

- full-frame PRE/POST labelled framepacks for F1/F2/manual GT;
- offline V2.21/V2.21.2/V2.21.5 high-recall proposal expansion;
- cached proposal sidecars merged into the unified dataset;
- proposal vs ranking metrics;
- meaningful baseline-score fallback;
- fresh-F2 projector/camera domain validation;
- research champion quarantine/promotion gates;
- real soft centre-biased F2 sampling with uniform exploration;
- offline proposal/train cycle commands;
- idempotent MD documentation updater.

No live/game authority is enabled.

## Install

```bash
cd ~/skjutbana/skjutbana
unzip -o skjutbana_v2.23.2_proposal_data_domain_pipeline_delta.zip
git diff --check
python3 -m automation.v2232_selftest
python3 -m automation.v2232_verify_install
python3 -m automation.v2232_apply_docs
```

See `V2232_TEST_PLAN.md` before running another full F2 session.
