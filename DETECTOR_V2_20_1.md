# V2.20.1 – visibility + generation performance correction

This is a small correction on top of V2.20 after manual inspection of real 2048×1152 exports.

## What the inspection showed

Among the first five generated PRE/POST pairs, two contained a real compact change while three were so weak that the new hole was effectively absent. V2.20 could still emit the last retry even when QA had not passed.

Generation was also far slower than necessary because:

- static images were decoded/resized once per PRE/POST frame instead of once per scenario;
- every old-hole render copied the whole full-resolution image;
- failed QA could repeat the complete camera simulation up to eight times.

## Fixes

- Strict `qa_passed` semantics; a normal failed attempt triggers local visibility rescue.
- ROI-only in-place hole rendering.
- Static image/media decode once per scenario.
- Shared static frame reuse before physical-hole layers are applied.
- Camera simulation only once per scenario.
- If the new-hole signal is too weak, only the small GT ROI is boosted/rechecked.
- `--save-debug` exports a GT-marked debug image and an enlarged GT crop. These files are inspection-only and are never candidate/training inputs.

## Expected performance

A local 2048×1152 stress test with 30–55 old holes dropped from more than two minutes for five scenarios in the previous retry-heavy path to roughly 8 seconds for five procedural scenarios. A static 2048×1152 image test was about 3.5 seconds for five scenarios in the development environment.

Actual speed depends on video decoding, storage and machine performance.
