# Skjutbana V2.23.3 delta

**Learned Candidate Reduction + Rich NEW-hole Ranking**

Install this ZIP on top of the current V2.23.2/V2.22.6 checkout.

V2.23.3 does not change hit authority. It adds:

- GT-free rich PRE/POST feature maps for dense V2.21.5 candidates;
- compact NPZ training caches;
- pairwise learned linear/MLP reducer;
- top-K retention metrics;
- reducer -> final listwise ranker cascade;
- explicit one-session bootstrap mode;
- fresh-session domain validation when a second dense F2 session exists;
- F2 completion scheduling of the V2.23.3 shadow cycle;
- progress logging so long training phases do not look frozen.

Start with:

```bash
python3 -m automation.v2233_selftest
python3 -m automation.v2233_verify_install
python3 -m automation.v2233_prepare --session latest
python3 -m automation.v2233_train --quick --no-prepare
python3 -m automation.v2233_status
```

See `V2233_TEST_PLAN.md` for the second independent F2 session/domain test.
