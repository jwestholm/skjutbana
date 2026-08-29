# V2.22.6 — Audio Raw / Near-Miss Telemetry

The existing trigger still requires the same four conditions:

- absolute peak threshold,
- adaptive noise-floor ratio,
- crest factor,
- cooldown.

V2.22.6 reproduces that decision without retuning it and records strong rejected transients.

Example accepted trigger:

```text
[V2.22.6 AUDIO-RAW] TRIGGER peak=... rms=... noise=... abs=... dyn=... crest=.../...
```

Example near miss:

```text
[V2.22.6 AUDIO-RAW] NEAR-MISS peak=... rms=... noise=... abs=... dyn=... crest=.../... required=... reject=abs,crest
```

`reject=` may contain:

- `abs`
- `noise`
- `crest`
- `cooldown`

Near-miss telemetry never queues `AudioPeakEvent`; it is observation only.
