"""Paired A/B comparison on identical cases (same seeds / same recorded shots).

Neutral evaluation helper: no detector, model or settings access. A case is
identified by a key; both arms must contain exactly the same keys, otherwise the
comparison is refused (never compare different datasets as one experiment).

Outcome per case and arm: ``correct`` (bool). Verdict uses an exact two-sided
McNemar (binomial sign test on discordant pairs) and a paired bootstrap CI on
the difference in correct rate. ``BETTER``/``WORSE`` require p < alpha AND a CI
excluding zero; otherwise ``NO_SIGNIFICANT_DIFFERENCE``.
"""
from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence


def exact_sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial test with p=0.5 on discordant pairs."""
    n = int(wins) + int(losses)
    if n == 0:
        return 1.0
    k = min(int(wins), int(losses))
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_compare(
    baseline: Mapping[str, bool],
    candidate: Mapping[str, bool],
    *,
    alpha: float = 0.05,
    bootstrap: int = 2000,
    seed: int = 20261117,
) -> dict[str, Any]:
    keys_a, keys_b = set(baseline), set(candidate)
    if keys_a != keys_b:
        raise ValueError(f"Arms differ in cases: only_baseline={len(keys_a - keys_b)} "
                         f"only_candidate={len(keys_b - keys_a)}")
    keys = sorted(keys_a)
    n = len(keys)
    if n == 0:
        raise ValueError("No cases to compare")
    a = [bool(baseline[k]) for k in keys]
    b = [bool(candidate[k]) for k in keys]
    wins = sum((not x) and y for x, y in zip(a, b))      # candidate fixed a baseline failure
    losses = sum(x and (not y) for x, y in zip(a, b))    # candidate broke a baseline success
    rate_a, rate_b = sum(a) / n, sum(b) / n
    rng = random.Random(seed)
    diffs = []
    for _ in range(int(bootstrap)):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    p = exact_sign_test_p(wins, losses)
    if p < alpha and lo > 0:
        verdict = "BETTER"
    elif p < alpha and hi < 0:
        verdict = "WORSE"
    else:
        verdict = "NO_SIGNIFICANT_DIFFERENCE"
    return dict(cases=n, baseline_correct=sum(a), candidate_correct=sum(b),
                baseline_rate=rate_a, candidate_rate=rate_b, difference=rate_b - rate_a,
                ci95=[lo, hi], discordant_wins=wins, discordant_losses=losses,
                p_value=p, alpha=alpha, verdict=verdict,
                fixed=[k for k, x, y in zip(keys, a, b) if not x and y],
                broken=[k for k, x, y in zip(keys, a, b) if x and not y])


def grouped_compare(
    baseline: Mapping[str, bool], candidate: Mapping[str, bool],
    groups: Mapping[str, str], **kwargs: Any,
) -> dict[str, Any]:
    """Overall verdict plus one per group (e.g. scenario); same case keys required."""
    out = dict(overall=paired_compare(baseline, candidate, **kwargs), groups={})
    for group in sorted(set(groups.values())):
        keys = [k for k, g in groups.items() if g == group]
        out["groups"][group] = paired_compare({k: baseline[k] for k in keys},
                                              {k: candidate[k] for k in keys}, **kwargs)
    return out


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a single rate (e.g. hit rate vs a 95% goal)."""
    if total <= 0:
        return (0.0, 1.0)
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


__all__: Sequence[str] = ("exact_sign_test_p", "paired_compare", "grouped_compare", "wilson_interval")
