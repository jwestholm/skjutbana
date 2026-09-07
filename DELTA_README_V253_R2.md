# V2.25.3-r2 delta

This replaces the first V2.25.3 archive, whose cumulative package accidentally shipped
a unit-test `src/engine/settings.py` stub.

## Install

Extract over the current repository, then run **prepare before main.py**:

```bash
unzip -o skjutbana_v2.25.3-r2_cross_thread_novelty_authority_delta.zip -d .
python3 -m automation.v253_prepare
python3 -m automation.v253_selftest
python3 -m automation.v253_verify_install
python3 -m automation.v253_status
python3 main.py
```

The prepare step automatically repairs a settings.py already damaged by the first
V2.25.3 archive by recovering the newest complete copy from Git history and reapplying
the viewport-local content_rect fix.

V2.25.3 runtime logic is otherwise unchanged.
