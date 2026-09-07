"""Isolated native-capture ranker research; never registers or promotes models."""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import shutil
from pathlib import Path

import numpy as np

from src.engine.ai.training_v223.model import RankModelV223, train_rank_model
from src.engine.ai.training_v223.schema import FEATURE_NAMES, ShotTrainingRecord
from src.engine.offline.evaluation import DEFAULT_RADII, digest, encoded, evaluate, points, provenance

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation_runs/offline_10iter_20260907"
BASE_CONFIG = dict(method="baseline", kind="linear", features="physical", transform="identity",
                   epochs=80, learning_rate=.02, l2=.001, seed=2230, hidden=24,
                   max_candidates_per_shot=256)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(encoded(value))


def load_split(out, name):
    records = []
    for entry in read(out / "splits.json")["splits"][name]:
        path = ROOT / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Frozen input changed: {path}")
        raw = read(path)
        points([{"camera_x": raw["gt_camera_x"], "camera_y": raw["gt_camera_y"]}])
        points(raw["candidates"])
        for c in raw["candidates"]:
            if c.get("baseline_score") is None or not math.isfinite(c["baseline_score"]):
                raise ValueError("Incomplete reference score")
            if any(not math.isfinite(v) for v in c["features"].values()):
                raise ValueError("Nonfinite features")
        record = ShotTrainingRecord.from_dict(raw)
        # Recompute distances from immutable coordinates, never trust stored labels.
        record.finalize_labels()
        records.append(record)
    return records


def feature_names(config):
    physical = [f for f in FEATURE_NAMES if not f.startswith("dense_") and not f.startswith("v2233_")
                and f not in ("x_norm", "y_norm")]
    if config["features"] == "physical":
        return physical
    if config["features"] == "evidence":
        return [f for f in physical if f not in ("area", "radius", "circularity", "detector_score")]
    if config["features"] == "geometry":
        return ["area", "radius", "circularity", "detector_score"]
    raise ValueError("Unknown feature set")


def transformed(records, config):
    result = copy.deepcopy(records)
    names = feature_names(config)
    for r in result:
        for name in names:
            values = np.asarray([c.features.get(name, 0.) for c in r.candidates])
            if config["transform"] == "signed_log":
                values = np.sign(values) * np.log1p(np.abs(values))
            elif config["transform"] == "within_shot_percentile":
                # Equal values receive equal ranks; no labels or candidate order feature.
                values = np.asarray([np.mean(values < v) for v in values])
            elif config["transform"] != "identity":
                raise ValueError("Unknown transform")
            for c, v in zip(r.candidates, values):
                c.features[name] = float(v)
    return result


def orders(records, config, model=None):
    if config["method"] == "baseline":
        return [sorted(range(len(r.candidates)), key=lambda i: -r.candidates[i].baseline_score) for r in records]
    return [model.rank_indices(r).tolist() for r in transformed(records, config)]


def measure(records, ranking):
    if len(records) != len(ranking):
        raise ValueError("Each shot requires exactly one complete ranking")
    observations = []
    rr = {str(int(t)): [] for t in DEFAULT_RADII}
    for r, order in zip(records, ranking):
        if sorted(order) != list(range(len(r.candidates))):
            raise ValueError("Ranking changed candidate membership")
        cs = [{"camera_x": c.camera_x, "camera_y": c.camera_y, "candidate_id": c.candidate_id} for c in r.candidates]
        ranked = [cs[i] for i in order]
        observations.append(dict(session_id=r.session_id, shot_id=r.shot_id, coordinate_space="camera",
            source_kind=r.source_kind, ground_truth={"camera_x": r.gt_camera_x, "camera_y": r.gt_camera_y},
            saved_pool=cs, ranked=ranked))
        for tolerance in DEFAULT_RADII:
            rank = next((j for j, i in enumerate(order, 1) if r.candidates[i].gt_distance_px <= tolerance), None)
            rr[str(int(tolerance))].append(rank)
    scorecard = evaluate(observations, mode="offline_candidate", top_k=10)
    metrics = {"samples": len(records), "tolerances": {}}
    for tolerance, block in scorecard["tolerances_camera_px"].items():
        stages = block["stages"]
        ranks = rr[tolerance]
        present = [r for r in ranks if r is not None]
        metrics["tolerances"][tolerance] = {
            **{key: stages[name]["correct"] for key, name in
               (("top1", "top_1"), ("top3", "top_3"), ("top10", "top_k"), ("oracle", "saved_pool"))},
            "mrr_all": sum(1 / r for r in present) / len(ranks) if ranks else None,
            "mean_rank_when_present": float(np.mean(present)) if present else None,
            "ranks": ranks}
    return metrics, scorecard


def objective(metrics):
    m = metrics["tolerances"]["20"]
    return m["top1"], m["mrr_all"], m["top3"]


def guardrails(metrics, baseline):
    return all(metrics["tolerances"][t]["oracle"] == baseline["tolerances"][t]["oracle"] and
               metrics["tolerances"][t]["top10"] >= baseline["tolerances"][t]["top10"] for t in baseline["tolerances"])


def baseline(out):
    target = out / "baseline"
    target.mkdir(exist_ok=False)
    records = load_split(out, "DEVELOPMENT")
    metrics, scorecard = measure(records, orders(records, BASE_CONFIG))
    frozen = read(out / "baseline_before_changes.json")
    assert scorecard["tolerances_camera_px"] == frozen["tolerances_camera_px"]
    result = dict(iteration=0, config=BASE_CONFIG, metrics=metrics, decision="baseline", hypothesis="Frozen saved-score reference")
    write(target / "result.json", result)
    write(target / "scorecard.json", scorecard)
    write(out / "environment.json", provenance(ROOT,
        [ROOT / e["path"] for es in read(out / "splits.json")["splits"].values() for e in es],
        "native-f2-chronological-3sessions", {"effective_detector_settings": None,"models": [],"calibration":None,
        "detector_executed":False,"numpy":np.__version__,"python":platform.python_version(),
        "reference_policy":"complete descending saved baseline_score; stable ties", "training_defaults":BASE_CONFIG}))
    print(encoded(result))


def result_path(out, iteration):
    return out / ("baseline" if iteration == 0 else f"iteration_{iteration:02d}") / "result.json"


def trial(out, iteration, parent, hypothesis, change):
    if (out / "frozen_challenger.json").exists():
        raise ValueError("Holdout phase started; tuning is closed")
    if not 1 <= iteration <= 10 or not 0 <= parent < iteration or len(change) != 1:
        raise ValueError("One config change per sequential iteration required")
    previous = read(result_path(out, iteration - 1))
    all_results = [read(result_path(out, i)) for i in range(iteration)]
    reference = all_results[0]
    incumbent = max([r for r in all_results if r["decision"] in ("baseline", "keep")], key=lambda r: objective(r["metrics"]))
    config = dict(all_results[parent]["config"])
    if set(change) - set(config):
        raise ValueError("Unknown config key")
    config.update(change)
    directory = out / f"iteration_{iteration:02d}"
    directory.mkdir(exist_ok=False)
    write(directory / "hypothesis.json", dict(iteration=iteration,parent=parent,hypothesis=hypothesis,change=change,
        largest_measured_failure={"missing_from_pool":incumbent["metrics"]["samples"]-incumbent["metrics"]["tolerances"]["20"]["oracle"],
        "present_but_not_top1":incumbent["metrics"]["tolerances"]["20"]["oracle"]-incumbent["metrics"]["tolerances"]["20"]["top1"]}))
    train = transformed(load_split(out, "TRAIN"), config)
    settings = {k:config[k] for k in ("kind","epochs","learning_rate","l2","seed","hidden","max_candidates_per_shot")}
    model, info = train_rank_model(train, feature_names=feature_names(config), **settings,
                                   metadata={"status":"OFFLINE_CHALLENGER_ONLY","experiment_config":config})
    model.save(directory)
    model_hashes = {p.name:digest(p) for p in (directory/"model.npz", directory/"model.json")}
    dev = load_split(out, "DEVELOPMENT")
    metrics, scorecard = measure(dev, orders(dev, config, model))
    safe = guardrails(metrics,reference["metrics"])
    decision = "keep" if safe and objective(metrics)>objective(incumbent["metrics"]) else "discard"
    result = dict(iteration=iteration,parent=parent,hypothesis=hypothesis,change=change,config=config,
        model_hashes=model_hashes,training=info,metrics=metrics,guardrails_passed=safe,decision=decision,
        comparisons={name:{"top1_20_delta":metrics["tolerances"]["20"]["top1"]-other["metrics"]["tolerances"]["20"]["top1"],
                         "mrr20_delta":metrics["tolerances"]["20"]["mrr_all"]-other["metrics"]["tolerances"]["20"]["mrr_all"]}
                     for name,other in (("original_baseline",reference),("previous_iteration",previous),("best_before",incumbent))},
        conclusion="Replaces development incumbent; remains offline challenger" if decision=="keep" else
                   "Rejected: coverage guardrail failed" if not safe else "Rejected: no development objective improvement")
    write(directory / "result.json", result)
    write(directory / "scorecard.json", scorecard)
    print(encoded({k:result[k] for k in ("iteration","hypothesis","metrics","decision","comparisons")}))


def finalize(out):
    results = [read(result_path(out,i)) for i in range(11)]
    best = max([r for r in results if r["decision"] in ("baseline","keep")],key=lambda r:objective(r["metrics"]))
    frozen = {"iteration":best["iteration"],"config":best["config"],"model_hashes":best.get("model_hashes"),
              "holdout_tuning_forbidden":True}
    write(out / "frozen_challenger.json", frozen)  # exclusive freeze BEFORE opening labels for scoring
    model = None
    if best["iteration"]:
        model_dir = result_path(out,best["iteration"]).parent
        for name,h in best["model_hashes"].items():
            assert digest(model_dir/name)==h
        shutil.copytree(model_dir,out/"best_challenger")
        model = RankModelV223.load(model_dir)
    holdout = load_split(out,"PROTECTED_FINAL_HOLDOUT")
    comparisons = {}
    for name,config,m in (("baseline",BASE_CONFIG,None),("challenger",best["config"],model)):
        metrics,scorecard = measure(holdout,orders(holdout,config,m))
        comparisons[name]=metrics
        write(out / f"holdout_{name}_scorecard.json",scorecard)
    report = dict(schema_version="offline-10iter-1",protocol=read(out/"protocol.json"),splits=read(out/"splits.json"),
        provenance=read(out/"environment.json"),iterations=results,best_development_iteration=best["iteration"],
        protected_holdout=comparisons,challenger_only=True,live_behavior_changed=False,
        limitations=read(out/"splits.json")["limitations"]+["Only projected captures; no physical/live accuracy inference."])
    write(out/"experiment_report.json",report)
    print(encoded({"best_iteration":best["iteration"],"holdout":comparisons}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("baseline","trial","finalize"))
    p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--iteration",type=int);p.add_argument("--parent",type=int)
    p.add_argument("--hypothesis");p.add_argument("--change",type=json.loads)
    args=p.parse_args()
    if args.command=="baseline":baseline(args.output)
    elif args.command=="trial":trial(args.output,args.iteration,args.parent,args.hypothesis,args.change)
    else:finalize(args.output)


if __name__=="__main__":main()
