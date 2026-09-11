# Adversarial workflow QA

The real JSON/CLI boundary selftest exercises duplicate mappings, AMBIGUOUS
labels, wrong plan identity, validation operation matrix, and process-level
preflight. Additional cases were checked against the quality/finalization
implementation using temporary fixtures.

| Case | Expected | Result |
|---|---|---|
| 1–3 false audio events | explicit event mapping | WARN/fixture-supported; labels must carry event IDs |
| 4 deliberate no-impact | separate classification | PASS by explicit label status |
| 5–6 missing PRE/POST | quality failure/capability loss | PASS via quality capability checks |
| 7–8 zero/truncated artifact | actionable quality failure | PASS; invalid JSON/artifact is reported |
| 9–10 missing PRE/POST | patch/full capability separation | PASS in quality report |
| 11 missing complete pool | patch yes, full replay no | PASS by inventory distinction |
| 12 commit mismatch | metadata warning/failure | WARN; binding records source commit |
| 13 dirty settings | preserve bytes | PASS; workflow never stages settings |
| 14 restore failure | recovery required | WARN; hardware restore remains external |
| 15 process restart | persisted metadata | PASS; binding is JSON-backed |
| 16 wrong plan | refuse | PASS; plan ID mismatch refuses finalize |
| 17 validation leakage | refuse all tuning actions | PASS; train/tune/fit/config/model/threshold/calibration guarded |
| 18 S01/S02/S03 mock flow | development/validation split | PASS at metadata/guard layer |
| 19 repeated finalize | safe or explicit refusal | WARN; caller should retain finalized marker |
| 20 error quality | actionable messages | PASS for mapping/plan/label guards; quality messages retain artifact path |

## Explicit semantics

An `AMBIGUOUS` label blocks finalization and dataset rebuild. It is never
converted to positive or negative and is excluded from metrics until resolved.
An interrupted 4/10 session is incomplete; resume is not assumed safe, so a
new session binding is required. Ten physical shots are always distinct from
any number of runtime audio events or deliberate no-impact events.

The remaining warning is restore failure: actual settings restoration depends
on the existing physical-test lifecycle and hardware process. The operator
must rerun the documented stop/restore command and verify the settings hash
before another session.
