# skjutbana V2.22 DELTA

Overlay this ZIP on the repository root of the latest local `dev` checkout.

Read in this order:

1. `V222_SHOT_RESOLVER.md`
2. `V222_TEST_PLAN.md`
3. `V222_DOC_NOTES.md`

V2.22 is intentionally safe/advisory-first. It adds the fast resolver and expert APIs but does not make the current V2.21.5 full-frame model live-authoritative.

The only existing source file replaced by this delta is:

- `src/engine/ai/bootstrap.py`

All other Python files in the delta are new. Before committing, inspect:

```bash
git diff --check
git diff -- src/engine/ai/bootstrap.py
git status --short
```

Then follow `V222_TEST_PLAN.md`.
