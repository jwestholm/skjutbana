# V2.25.3 cumulative delta

Base: V2.25.2. Package: cumulative V2.24.1 → V2.25.3 overlay.

## Install

```bash
unzip -o skjutbana_v2.25.3_cross_thread_novelty_authority_delta.zip -d .
python3 -m automation.v253_prepare
python3 -m automation.v253_selftest
python3 -m automation.v253_verify_install
python3 -m automation.v253_status
python3 main.py
```

V2.25.3 fixes the worker/main registered-ready state boundary and adds a soft
cross-shot physical recurrence prior in full-camera coordinates. It does not modify the
GameObject model or grant game semantics to the detector.
