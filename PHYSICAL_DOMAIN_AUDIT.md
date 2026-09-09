# Physical domain audit

Coarse PRE-frame descriptors show the largest measured session difference in
edge/gradient structure, not mean brightness:

| session | brightness | contrast | gradient | edge fraction |
|---|---:|---:|---:|---:|
| biathlon5 | 137.0 | 55.1 | 5.21 | .118 |
| smoke2 | 147.0 | 54.8 | 5.57 | .128 |
| test1 | 150.3 | 54.8 | 5.54 | .129 |
| 144822 | 141.7 | 55.4 | 5.28 | .121 |
| dart20 | 145.2 | 55.1 | 6.15 | .152 |
| latest10 | 138.7 | 54.9 | 5.41 | .127 |

The 22-event dart session has materially higher gradient/edge density and is
the clearest measurable domain shift. The remaining sessions are closer in
coarse brightness/contrast. These descriptors are scene statistics, not visual
semantic labels; no unsupported background names are assigned.

The next collection should deliberately include edge/line-heavy and textured
conditions across separate sessions, because those are underrepresented as
independent domains and correspond to the largest measured shift.
