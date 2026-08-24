from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.engine.ai.ranker_v8 import MODEL_PATH
from src.engine.ai.ranker_v8_optimizer import (
    evaluate_model,
    feature_profile_table,
    fit_feature_profile,
    fit_model_from_profile,
    prepare_rows,
    recommendation,
    search_configs,
    stable_confirmation_split,
)
from src.engine.ai.ranking_dataset_v29 import DATA_ROOT, load_session


REPORT_ROOT = DATA_ROOT / "v210_experiments"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V2.10 offline monotonic feature-search optimizer. No camera or "
            "projector is used."
        )
    )
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--top-configs", type=int, default=8)
    parser.add_argument("--no-save-model", action="store_true")
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

    print("=" * 82)
    print("V2.10 OFFLINE MONOTONIC RANK OPTIMIZER")
    print("=" * 82)
    print(f"Session:       {session}")
    print(f"Usable shots:  {len(shots)}")
    print(f"Development:   {len(development)}")
    print(f"Confirmation:  {len(confirmation)}  (never used to choose config)")
    print("Camera/projector: NOT USED")
    print()

    dev_profile = fit_feature_profile(development)
    print("STRONGEST MONOTONIC FEATURE EVIDENCE ON DEVELOPMENT DATA:")
    for row in feature_profile_table(dev_profile, limit=18):
        print(
            f"  {row['feature']:27s} GT {row['direction']:4s} "
            f"win={100.0 * float(row['win_rate']):5.1f}% "
            f"strength={row['strength']:.3f} pairs={row['comparisons']}"
        )

    print()
    print("Searching deterministic monotonic configurations...")
    started = time.time()
    search = search_configs(development)
    elapsed = time.time() - started
    if not search:
        print("No configurations could be evaluated.")
        raise SystemExit(1)

    top_count = max(1, min(int(args.top_configs), len(search)))
    print(f"Evaluated {len(search)} configurations in {elapsed:.2f}s")
    print()
    print("TOP DEVELOPMENT-CV CONFIGURATIONS:")
    for index, item in enumerate(search[:top_count], start=1):
        cfg = item["config"]
        cv = item["cv"]
        print(
            f"#{index:02d} objective={item['objective']:.3f} "
            f"features={cfg.feature_count} min_strength={cfg.min_strength:.2f} "
            f"corr={cfg.corr_limit:.2f} evidence_p={cfg.evidence_power:.2f} "
            f"weight_p={cfg.weight_power:.2f} baseline_blend={cfg.baseline_blend:.2f}"
        )
        print("    " + _metric_line(cv, 20))
        print("    " + _metric_line(cv, 42))

    best = search[0]
    best_config = best["config"]
    best_cv = best["cv"]

    # The confirmation set has not participated in config selection.
    dev_model = fit_model_from_profile(dev_profile, best_config)
    confirmation_result = evaluate_model(confirmation, dev_model)

    # Final shadow model uses all available rows, but the quality recommendation
    # is based only on development CV + untouched confirmation result.
    full_profile = fit_feature_profile(shots)
    final_model = fit_model_from_profile(full_profile, best_config)
    final_in_sample = evaluate_model(shots, final_model)
    rec = recommendation(
        shots=len(shots),
        development_cv=best_cv,
        confirmation=confirmation_result,
    )

    print()
    print("=" * 82)
    print("BEST CONFIGURATION")
    print("=" * 82)
    for key, value in best_config.as_dict().items():
        print(f"  {key:20s}: {value}")
    print(f"  selected_features   : {', '.join(dev_model.feature_keys)}")
    print()
    print("DEVELOPMENT CROSS-VALIDATION:")
    print("  " + _metric_line(best_cv, 20))
    print("  " + _metric_line(best_cv, 42))
    print()
    print("UNTOUCHED CONFIRMATION SET:")
    print("  " + _metric_line(confirmation_result, 20))
    print("  " + _metric_line(confirmation_result, 42))
    print()
    print("FINAL SHADOW MODEL FEATURES (fit on all captured shots):")
    for key, weight in zip(final_model.feature_keys, final_model.weights.tolist()):
        profile = final_model.feature_profile.get(key, {})
        direction = "HIGH" if weight > 0 else "LOW"
        print(
            f"  {key:27s} {direction:4s} weight={abs(float(weight)):.4f} "
            f"win={100.0 * float(profile.get('win_rate', 0.5)):5.1f}%"
        )
    print()
    print("RECOMMENDATION:")
    for key, value in rec.items():
        print(f"  {key}: {value}")

    report = {
        "schema_version": "2.10",
        "created_at": time.time(),
        "session": session,
        "shots": len(shots),
        "development_shots": len(development),
        "confirmation_shots": len(confirmation),
        "search_seconds": round(elapsed, 4),
        "configs_evaluated": len(search),
        "best_config": best_config.as_dict(),
        "development_cv": best_cv,
        "confirmation": confirmation_result,
        "final_in_sample": final_in_sample,
        "development_feature_profile": feature_profile_table(dev_profile, limit=40),
        "final_model": final_model.as_payload(),
        "recommendation": rec,
        "top_configs": [
            {
                "config": item["config"].as_dict(),
                "objective": item["objective"],
                "cv": item["cv"],
            }
            for item in search[:top_count]
        ],
    }

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_ROOT / f"{time.strftime('%Y%m%d_%H%M%S')}_{session}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not args.no_save_model:
        payload = {
            "schema_version": "2.10",
            "model_version": 8,
            "model_type": "monotonic_percentile_rank_ensemble",
            "shadow_only": True,
            "trained_session": session,
            "trained_shots": len(shots),
            "created_at": time.time(),
            "feature_keys": list(final_model.feature_keys),
            "weights": {
                key: float(weight)
                for key, weight in zip(final_model.feature_keys, final_model.weights.tolist())
            },
            "evidence_power": float(final_model.evidence_power),
            "baseline_blend": float(final_model.baseline_blend),
            "feature_profile": final_model.feature_profile,
            "best_config": best_config.as_dict(),
            "development_cv": best_cv,
            "confirmation": confirmation_result,
            "recommendation": rec,
        }
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print()
    print(f"Experiment report: {report_path}")
    if not args.no_save_model:
        print(f"V8 shadow model:   {MODEL_PATH}")
    print("V2.10 NEVER gives V8 authority. It is shadow-only.")
    print("=" * 82)


if __name__ == "__main__":
    main()
