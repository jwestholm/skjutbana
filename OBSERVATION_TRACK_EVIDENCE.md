# Observation-to-track evidence

This is an offline audit of the latest physical session. It uses only saved
track association histories and recorded frames; it does not change runtime
selection.

## Coordinate chain

The complete physical export does not contain a full per-frame observation
ledger for every track. Where association history is present, most tracks have
one unique coordinate; shots 3 and 10 show about 9–10 px observation drift,
and the selected false winner in shot 8 shows about 8 px. The available data
therefore does not support a claim of broad representative drift. Missing
observation identity is reported as missing rather than inferred.

The temporal extractor now evaluates every exported observation coordinate and
the final track coordinate. In this trace most true tracks have identical
available coordinate sets, so the exploratory persistence AUC remains about
0.71 at the final coordinate. A confirmation-specific coordinate could not be
reconstructed separately for most rows because the saved confirmation record
does not preserve its best-search XY.

## Rank versus localization

The nine @42 causal candidates all reach eligible tracks. Their final
representative errors range from approximately 2.2 to 12.5 px for the
@42-positive examples; shot 5 is the expected no-oracle case at 53.3 px.
Thus the dominant physical failure remains rank, not wholesale track
localization. Best-observation error and confirmation-XY error cannot be
computed for all rows from the current export.

## Status

**PROVEN:** complete replay preserves the final selected track and the causal
candidate-to-eligible funnel.

**STRONG HYPOTHESIS:** temporal impact evidence may be better attached to the
supporting observation than to a drifting representative.

**NOT YET OBSERVABLE:** a full confirmation-search displacement distribution
or all observation-level temporal curves. Future tracing should preserve the
confirmation best XY and every consumed observation explicitly.

Reproduction:

```text
python3 -m automation.temporal_impact_research --root content/ai/physical_traces/session_20260908_194746_b38de674 --features evaluation_runs/registered_impact_20260909/physical_v2/physical_features.json --output evaluation_runs/registered_impact_20260909/temporal_v2
```

The exact final comparator is now documented in
[FINAL_RANKING_AUDIT.md](FINAL_RANKING_AUDIT.md): onset distance first, then
negative historical `best_score`. Confirmation and source are eligibility and
provenance inputs, not final ordering terms.
