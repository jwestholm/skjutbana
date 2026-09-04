# V2.25.2 documentation patch

`automation/v252_apply_docs.py` appends the V2.25.2 registered-freshness authority
checkpoint to living repository docs without replacing existing history.

Stable references shipped with this delta:

- `V252_REGISTERED_FRESHNESS_AUTHORITY.md`
- `AI_REGISTERED_FRESHNESS.md`
- `V252_TEST_PLAN.md`

V2.25.2 originally moved continuous moving-object updates to V2.25.3. Physical V2.25.2 acceptance then exposed another authority issue, so V2.25.3 became the cross-thread/novelty correction and moving-object continuity is now V2.25.4.
