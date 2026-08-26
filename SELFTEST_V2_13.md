# V2.13 selftest / validation notes

## Mandatory selftest

Command:

```bash
python3 -m automation.hole_v213_selftest
```

The selftest creates a temporary hole archive and verifies:

- synthetic/real asset discovery,
- real data never enters train/validation/test,
- session-level split has no overlap,
- novel-background holdout is separate,
- centred source images are converted into candidate-centred jittered samples,
- the numpy neural network actually reduces loss on held-out toy data,
- off-centre positive recall remains useful,
- X/Y offset localisation remains useful off-centre,
- model save/load works.

Expected final line:

```text
All V2.13 selftests passed.
```

## End-to-end CLI smoke test performed while packaging

The following path was tested end to end on generated archives:

```text
hole_v213_inspect
  -> hole_v213_train
  -> hole_v213_evaluate
  -> hole_v213_visualize
```

A 1,720-source scale smoke test (1,600 ordinary synthetic + 120 novel-background + 20 real-style holdout) trained for two epochs with the default 64-hidden-unit model. On the packaging machine it used about 157 MB peak RSS and completed the entire command in about 5 seconds. This runtime is **not a promise** for the shooting computer; it only confirms the implementation does not require loading the full image bank into memory at once.

The generated scale test showed the intended behaviour:

- training loss dropped,
- held-out session performance exceeded the simple centre-contrast baseline,
- novel-background performance was separately measurable,
- real-style holdout performance was lower than synthetic performance (a healthy indication that the test can expose domain shift),
- off-centre stress remained measurable instead of being hidden by normal accuracy.

These percentages are synthetic implementation tests and must **not** be quoted as shooting-range accuracy. The first meaningful V2.13 result is the report generated from the real `content/ai/holes` archive on the shooting PC.
