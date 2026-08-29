# V2.22.6 Physical Test Plan

## 1. Static checks

```bash
python3 -m automation.v2226_selftest
python3 -m automation.v2226_verify_install
python3 -m automation.v2226_apply_docs
```

Then start:

```bash
python3 main.py
```

Expected startup line:

```text
[V2.22.6] frame-unique tracking + audio near-miss telemetry installed
```

## 2. First physical shot

Do not F2/autotrain. Use normal AI Training/advisory.

Expected accepted audio sequence:

```text
[V2.22.6 AUDIO-RAW] TRIGGER ...
[V2.22.3 AUDIO-PRIORITY] ...
```

The first global candidate result should log:

```text
[V2.22.6 TRACK] ... new=... temporal=... same_frame=...
```

The real candidate may have high same-frame support, but a newly created track must still start at one temporal hit.

## 3. Local confirmation

A later camera frame should produce:

```text
[V2.22.5 LOCAL-CONFIRM] ... confirmed=...
[V2.22.6 TRACK] ... temporal=...
```

That is the desired second observation.

## 4. Audio miss case

If a physical shot produces no `AUDIO-PRIORITY`, look immediately for:

```text
[V2.22.6 AUDIO-RAW] NEAR-MISS ... reject=...
```

Return that line unchanged. If there is neither TRIGGER nor NEAR-MISS, the signal was below the diagnostic near-gate or audio capture itself did not see a strong transient; that becomes the next audio investigation.

## 5. What to send back

For 3 shots, send the terminal block containing:

- `AUDIO-RAW`
- `AUDIO-PRIORITY`
- `V2.22.4 CV`
- `V2.22.5 FAST-EXTRACT`
- `V2.22.6 TRACK`
- `V2.22.5 LOCAL-CONFIRM`
- `SHOT #`
- `VISIBLE`

Also say whether the selected ring was on the physical hole and approximately how long it felt until visible.
