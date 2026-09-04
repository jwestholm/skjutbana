# V2.25.2 delta — registered freshness authority

Physical V2.25.1 testing showed region balancing was active but local hit authority could
still leak through early/legacy/bank candidates. V2.25.2 requires exact candidate XY to
be independently supported by CandidateGeneratorV2's registered immediate PRE→POST maps
before it may become the normal object-context winner.

Install over V2.25.1/dev:

```bash
unzip -o skjutbana_v2.25.2_registered_freshness_authority_delta.zip -d .
python3 -m automation.v252_prepare
python3 -m automation.v252_selftest
python3 -m automation.v252_verify_install
python3 -m automation.v252_status
python3 main.py
```

The ZIP is cumulative from V2.24.1 through V2.25.2.

Important: V2.22.5 FULL rescue remains global and candidate XY is never snapped to a
GameObject.
