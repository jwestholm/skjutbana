# Physical data inventory

Inventory is read-only and based on files under `content/ai/physical_traces`.

| session | shots/traces | labels | frames | patch dataset | full rank replay | limitations |
|---|---:|---:|---:|---|---|---|
| session_20260907_biathlon5 | 6 | 5 | 303 | yes | no | legacy lacks decision/selectors |
| session_20260907_smoke2 | 2 | 2 | 110 | yes | no | legacy lacks decision/selectors |
| session_20260907_test1 | 1 | 1 | 381 | partial | no | referenced frame files missing |
| session_20260908_144822_d6713dec | 10 | 10 | 556 | yes | no | no complete eligible-pool selector snapshot |
| session_20260908_163626_9b47fac6 | 22 | 20 | 1197 | yes | yes | two explicit false events |
| session_20260908_194746_b38de674 | 10 | 10 | 451 | yes | yes | latest post-fix development |

There are **61 trace events**, **48 labelled physical shots**, and **47 usable
positive patch examples** after excluding the shot whose referenced frame files
are missing. The unified conservative patch dataset contains **47 positives
and 39 recorded selected-false negatives** across five sessions. Only the two
newer sessions support complete eligible-pool final-rank replay.

All traces use `physical-shot-trace-1` but not identical semantics. The legacy
sessions have real PRE/POST arrays and camera GT, so they are
`PATCH_DATASET_COMPATIBLE`; they are not silently upgraded to full replay.
