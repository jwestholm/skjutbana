# V2.21.3 checkpoint

## Input evidence

V2.21.2 on the first 30 honest full-frame projector/camera rounds:

- current <=20: 8/30,
- current <=42: 27/30,
- V2.21 global direct <=20: 1/30,
- V2.21.2 current+local <=20: 19/30,
- V2.21.2 current+local <=42: 28/30,
- local rescued 11 current misses at <=20.

No useful global calibration offset was found. Full-frame phase registration was also near zero. Temporal GT salience is strongest in blackhat/tophat/persistent-abs maps, while the old fused map is often dominated by structured projector/camera nuisance.

## Decision

Do not retrain V2.18 yet.

Test a GT-free anchored temporal-consensus proposal layer plus a candidate-derived target mask for direct rescue. Tune only on development, freeze, then inspect protected confirmation/holdout.

If the evidence-backed oracle cannot clear the next recall gate, move to learned physical-domain dense change evidence rather than adding more hand-tuned global peaks.
