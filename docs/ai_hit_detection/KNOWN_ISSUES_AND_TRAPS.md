# Known issues and traps

## 1. Synthetic accuracy can be misleading

The current strongest example is V2.18:

- ~98–100% Top-1 on unseen generated validation,
- 0% Top-1 on the existing physical session.

Never promote based on synthetic score alone.

## 2. Proposal oracle is a hard ceiling

Physical union oracle <=20 px is only 35% confirmation / 50% holdout.

A better ranker cannot recover absent candidates.

## 3. Old holes are not non-holes

Do not reuse far-from-GT candidate rows as static negative hole examples.

They are valid `NOT_NEW` examples only.

## 4. Known-hole registry is incomplete

Do not use it as hard truth or create another competing map.

## 5. Hole-in-hole must remain valid

Any logic that hard-excludes candidates near an existing hole can make legitimate re-hits impossible.

## 6. `shot_diag` is not raw training truth

Diagnostic images may contain crosshairs/text/other visual annotations.

Do not train full-frame models directly on them unless raw imagery is independently verified.

## 7. Centered source-patch shortcut

Historical `synt_*` patches are centered around projected hole locations.

Any training task using them must jitter/reposition or extract only calibrated appearance statistics. Do not let “centre pixel” become the answer.

## 8. V2.19 signed-residual hole rendering was invalid

It produced black/white streaks and context leakage. Keep the compact V2.20+ hole approach or a better physically calibrated replacement.

## 9. Independent PRE/POST augmentation was unrealistic

Static PRE and POST should share camera state. Only small physically plausible temporal nuisance should vary.

## 10. Transparent sprite -> black background trap

RGBA game/sprite images must not implicitly become black in transparent areas. V2.20.2 uses deterministic procedural compositing.

## 11. Generated worlds currently appear easier than physical data

Evidence:

- generated current Top-1 roughly 65–80% on frozen validation subsets,
- physical current Top-1 only ~5–10%.

This is a strong sign that simulator difficulty/distribution is not yet physically matched.

## 12. Benchmark report-path output

A V2.18 benchmark run printed:

`content/ai/reports/v218/new_hole_v218_rebenchmark.json`

even when a model under another output directory was used. Verify actual file modification time/path before archiving results. Treat this as a small tooling issue to clean up in the next code pass.

## 13. One physical session is not a final holdout

The current 60/20/20 split is shot-level within one session and therefore provisional.

Final authority requires multiple independent physical sessions/backgrounds.

## 14. Do not optimize to holdout feedback

Once a validation/holdout seed/session has been inspected, it must not silently become training data or be used repeatedly for hyperparameter fishing beyond its intended role.
