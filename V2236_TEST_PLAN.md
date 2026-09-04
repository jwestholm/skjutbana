# V2.23.6 Test Plan

1. Run `python3 -m automation.v2236_selftest`.
2. Run `python3 -m automation.v2236_verify_install` on the shooting PC.
3. Run `python3 -m automation.v2236_prepare --session latest`.
4. Confirm 100 heatmap caches and zero preparation errors.
5. Run `python3 -m automation.v2236_train --quick --no-prepare`.
6. Compare deterministic map baselines with the learned heatmap model.
7. Inspect Top1@20, Top3@20, median/P95 XY error and selected direct policy.
8. Do not run new F2 until the bootstrap result has been reviewed.
9. If direct-path gate passes, next step is resolver-shadow integration and fresh-session validation, not another global candidate reducer.
10. Live authority remains NO throughout V2.23.6.
