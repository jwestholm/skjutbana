# V2.21.2 diagnostic decision tree

Observed V2.21 full-frame physical/projector-camera result:

- global direct proposals do not currently rescue current misses,
- current candidate oracle is 26.7% @20 px but 90% @42 px on the new 30-shot white session.

That gap means V2.21.2 must first distinguish **localisation** from **candidate absence**.

Decision after the V2.21.2 report:

- If current <=42 offsets have a stable median dx/dy with low MAD: inspect calibration/coordinate mapping before ML.
- If temporal map GT percentiles are high but global direct recall is low: direct V2.21 is suffering top-N/threshold crowding; redesign proposal extraction.
- If `current_plus_local` strongly raises @20 on DEVELOPMENT and survives confirmation/holdout: promote local temporal refinement to the next rank/fusion experiment.
- If GT percentiles are low and local refinement does not help: the PRE/POST evidence formulation is wrong for projector/camera data; build a physical-domain learned dense change model from full-frame captures.

Do not train V2.18 on these 30 packs before this diagnosis is complete.
