from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.engine.ai.ranker_v9 import MODEL_PATH, PHYSICAL_FEATURE_KEYS
from src.engine.ai.ranker_v9_optimizer import (
    evaluate_rule,
    feature_evidence_table,
    fit_feature_evidence,
    fit_rule,
    prepare_rows,
    recommendation,
    search_rules,
    stable_confirmation_split,
)
from src.engine.ai.ranking_dataset_v29 import DATA_ROOT, load_session


REPORT_ROOT = DATA_ROOT / "v211_experiments"
CANDIDATE_PATH = Path("content/ai/ranker_v9_candidate.json")


def _metric_line(result: dict[str, Any], radius: int) -> str:
    model = result["model"][str(radius)]
    base = result["baseline"][str(radius)]
    return (
        f"<= {radius:2d}px MODEL top1={model['top1_pct']:6.2f}% "
        f"top3={model['top3_pct']:6.2f}% top5={model['top5_pct']:6.2f}% "
        f"med={str(model['median_rank']):>6s} MRR={model['mrr']} | "
        f"BASE top1={base['top1_pct']:6.2f}% top3={base['top3_pct']:6.2f}% "
        f"top5={base['top5_pct']:6.2f}% med={str(base['median_rank']):>6s}"
    )


def _model_payload(
    *,
    session: str | None,
    shots: int,
    rule: Any,
    development_cv: dict[str, Any],
    confirmation: dict[str, Any],
    rec: dict[str, Any],
    shadow_ready: bool,
) -> dict[str, Any]:
    payload = {
        "schema_version": "2.11",
        "model_version": 9,
        "model_type": "physical_monotonic_listwise_ranker",
        "shadow_only": True,
        "shadow_ready": bool(shadow_ready),
        "trained_session": session,
        "trained_shots": int(shots),
        "created_at": time.time(),
        "feature_keys": list(rule.feature_keys),
        "weights": {
            key: float(weight)
            for key, weight in zip(
                rule.feature_keys,
                rule.signed_weights.tolist(),
            )
        },
        "feature_evidence": rule.feature_evidence,
        "development_cv": development_cv,
        "confirmation": confirmation,
        "recommendation": rec,
        "policy_features_forbidden": [
            "baseline_score",
            "rel_baseline_score",
            "core_member",
            "reason_*",
            "rel_*",
        ],
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V2.11 physical-feature-only broad-negative/listwise optimizer. "
            "No camera or projector is used."
        )
    )
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--top-features", type=int, default=12)
    parser.add_argument("--top-configs", type=int, default=12)
    args = parser.parse_args()

    rows, session = load_session(args.session)
    if not rows:
        print("No V2.9 ranking dataset found.")
        raise SystemExit(1)

    shots = prepare_rows(rows)
    if len(shots) < 20:
        print(f"Only {len(shots)} usable shots. At least 20 are required.")
        raise SystemExit(1)

    development, confirmation = stable_confirmation_split(shots)
    if not development or not confirmation:
        print("Could not create development/confirmation split.")
        raise SystemExit(1)

    print("=" * 92)
    print("V2.11 PHYSICAL-FEATURE / BROAD-NEGATIVE LISTWISE OPTIMIZER")
    print("=" * 92)
    print(f"Session:             {session}")
    print(f"Usable shots:         {len(shots)}")
    print(f"Development:          {len(development)}")
    print(f"Confirmation:         {len(confirmation)}  (untouched during search)")
    print(f"Physical features:    {len(PHYSICAL_FEATURE_KEYS)}")
    print("Policy features:      FORBIDDEN")
    print("Camera/projector:     NOT USED")
    print()

    evidence = fit_feature_evidence(development)
    evidence_rows = feature_evidence_table(evidence)

    print("BROAD-NEGATIVE PHYSICAL FEATURE EVIDENCE:")
    for row in evidence_rows[:20]:
        print(
            f"  {row['feature']:24s} family={row['family']:9s} "
            f"GT {row['direction']:4s} "
            f"win={100.0 * float(row['win_rate']):5.1f}% "
            f"strength={row['strength']:.3f} "
            f"pairs={row['comparisons']}"
        )

    print()
    print("Searching single features + family-diverse 2/3-feature rules...")
    started = time.time()
    search = search_rules(
        development,
        top_features=max(5, int(args.top_features)),
    )
    elapsed = time.time() - started

    print(
        f"Evaluated {search['configs_evaluated']} listwise configurations "
        f"in {elapsed:.2f}s"
    )
    print()

    print("SINGLE-FEATURE HELD-OUT CV:")
    for row in search["single_features"][:20]:
        cv = row["cv"]
        print(
            f"  {row['feature']:24s} family={row['family']:9s} "
            f"gate={'PASS' if row['top1_gate_pass'] else 'FAIL'} | "
            + _metric_line(cv, 20)
        )

    best = search.get("best")
    top_configs = search["all"][: max(1, int(args.top_configs))]

    print()
    print("TOP LISTWISE CONFIGURATIONS:")
    for index, item in enumerate(top_configs, start=1):
        config = item["config"]
        cv = item["cv"]
        print(
            f"#{index:02d} gate={'PASS' if item['top1_gate_pass'] else 'FAIL'} "
            f"objective={item['objective']:.3f} "
            f"features={'+'.join(config.feature_keys)} "
            f"weights={list(config.weights)}"
        )
        print("    " + _metric_line(cv, 20))
        print("    " + _metric_line(cv, 42))

    report: dict[str, Any] = {
        "schema_version": "2.11",
        "created_at": time.time(),
        "session": session,
        "shots": len(shots),
        "development_shots": len(development),
        "confirmation_shots": len(confirmation),
        "physical_feature_keys": list(PHYSICAL_FEATURE_KEYS),
        "policy_features_forbidden": True,
        "search_seconds": round(elapsed, 4),
        "configs_evaluated": int(search["configs_evaluated"]),
        "feature_evidence": evidence_rows,
        "single_features": [
            {
                "feature": row["feature"],
                "family": row["family"],
                "objective": row["objective"],
                "top1_gate_pass": row["top1_gate_pass"],
                "cv": row["cv"],
            }
            for row in search["single_features"]
        ],
        "top_configs": [
            {
                "config": item["config"].as_dict(),
                "objective": item["objective"],
                "top1_gate_pass": item["top1_gate_pass"],
                "cv": item["cv"],
            }
            for item in top_configs
        ],
    }

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    if best is None:
        rec = recommendation(
            total_shots=len(shots),
            development_cv=None,
            confirmation=None,
        )
        report["recommendation"] = rec
        report["best"] = None
        print()
        print("=" * 92)
        print("NO V9 RULE PASSED THE HARD DEVELOPMENT TOP-1 <=20PX GATE")
        print("=" * 92)
        print("Baseline remains the only acceptable ranking reference.")
        print("No V9 shadow model was activated.")
    else:
        config = best["config"]
        development_cv = best["cv"]

        dev_evidence = fit_feature_evidence(development)
        dev_rule = fit_rule(dev_evidence, config)
        confirmation_result = evaluate_rule(confirmation, dev_rule)

        all_evidence = fit_feature_evidence(shots)
        final_rule = fit_rule(all_evidence, config)
        full_in_sample = evaluate_rule(shots, final_rule)

        rec = recommendation(
            total_shots=len(shots),
            development_cv=development_cv,
            confirmation=confirmation_result,
        )

        print()
        print("=" * 92)
        print("BEST HARD-GATED CONFIGURATION")
        print("=" * 92)
        print(f"Features: {', '.join(config.feature_keys)}")
        print(f"Search weights: {list(config.weights)}")
        print()
        print("DEVELOPMENT CROSS-VALIDATION:")
        print("  " + _metric_line(development_cv, 20))
        print("  " + _metric_line(development_cv, 42))
        print()
        print("UNTOUCHED CONFIRMATION SET:")
        print("  " + _metric_line(confirmation_result, 20))
        print("  " + _metric_line(confirmation_result, 42))
        print()
        print("FINAL PHYSICAL FEATURE DIRECTIONS:")
        for key, weight in zip(
            final_rule.feature_keys,
            final_rule.signed_weights.tolist(),
        ):
            evidence_row = final_rule.feature_evidence.get(key, {})
            print(
                f"  {key:24s} "
                f"{'HIGH' if float(weight) > 0 else 'LOW ':4s} "
                f"weight={abs(float(weight)):.3f} "
                f"broad-win={100.0 * float(evidence_row.get('win_rate', 0.5)):5.1f}%"
            )

        print()
        print("RECOMMENDATION:")
        for key, value in rec.items():
            print(f"  {key}: {value}")

        candidate_payload = _model_payload(
            session=session,
            shots=len(shots),
            rule=final_rule,
            development_cv=development_cv,
            confirmation=confirmation_result,
            rec=rec,
            shadow_ready=bool(rec.get("shadow_ready", False)),
        )
        CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CANDIDATE_PATH.write_text(
            json.dumps(candidate_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Only a model that survives both development Top-1 gate and the
        # untouched confirmation sanity gate becomes loadable by V9 shadow.
        if bool(rec.get("shadow_ready", False)):
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            MODEL_PATH.write_text(
                json.dumps(candidate_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            # Prevent a stale previous V9 model from accidentally shadowing a
            # newly rejected experiment.
            try:
                MODEL_PATH.unlink()
            except FileNotFoundError:
                pass

        report["best"] = {
            "config": config.as_dict(),
            "development_cv": development_cv,
            "confirmation": confirmation_result,
            "full_in_sample": full_in_sample,
            "final_rule": final_rule.payload(),
        }
        report["recommendation"] = rec

    report_path = (
        REPORT_ROOT
        / f"{time.strftime('%Y%m%d_%H%M%S')}_{session}.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"Experiment report: {report_path}")
    if CANDIDATE_PATH.exists():
        print(f"V9 candidate:       {CANDIDATE_PATH}")
    if MODEL_PATH.exists():
        print(f"V9 shadow-ready:    {MODEL_PATH}")
    else:
        print("V9 shadow-ready:    NO (model file intentionally absent)")
    print("V2.11 NEVER gives V9 authority. It is shadow-only.")
    print("=" * 92)


if __name__ == "__main__":
    main()
