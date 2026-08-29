from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .dataset import compile_dataset
from .registry import load_champion_entry, load_registry

AI_ROOT = Path("content/ai")
REPORT_ROOT = AI_ROOT / "training_v223" / "reports"

MODEL_PATTERNS = (
    "memory.json", "model.json", "ranker_v*.json", "ranker_v*.npz",
    "**/*model*.json", "**/*model*.npz", "**/*ranker*.json", "**/*ranker*.npz",
    "**/*champion*.json", "**/*hole*v21*.npz", "**/*hole*v21*.json",
    "reports/v2215/*.npz", "reports/v2215/*.json", "reports/v21*/*.json",
)
DATA_DIRS = (
    "holes", "candidate_shadow_v216", "candidate_synthetic_v220",
    "candidate_synthetic_v220_validation", "media_bank", "offline", "training_v223",
)
MODULE_PROBES = (
    "src.engine.offline.physical_dense_v2215",
    "automation.physical_dense_v2215_train",
    "automation.physical_dense_v2215_benchmark",
    "automation.newhole_v217_train",
    "automation.newhole_v217_benchmark",
    "automation.ranker_v211_optimize",
)


def _path_info(path: Path) -> dict[str, Any]:
    try:
        st = path.stat()
        return {"path": str(path), "size_bytes": st.st_size, "mtime": st.st_mtime}
    except Exception:
        return {"path": str(path)}


def audit_repository_state() -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    if AI_ROOT.exists():
        for pattern in MODEL_PATTERNS:
            for path in AI_ROOT.glob(pattern):
                if path.is_file() and str(path) not in seen:
                    seen.add(str(path)); models.append(_path_info(path))
    dirs: dict[str, Any] = {}
    for name in DATA_DIRS:
        path = AI_ROOT / name
        if not path.exists():
            dirs[name] = {"exists": False}
            continue
        files = sum(1 for p in path.rglob("*") if p.is_file())
        bytes_total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        dirs[name] = {"exists": True, "files": files, "bytes": bytes_total}
    modules: dict[str, Any] = {}
    for name in MODULE_PROBES:
        try:
            mod = importlib.import_module(name)
            modules[name] = {"available": True, "path": getattr(mod, "__file__", None)}
        except Exception as exc:
            modules[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    code_inventory: dict[str, list[str]] = {}
    for label, pattern in (
        ("ai", "src/engine/ai/*.py"),
        ("offline", "src/engine/offline/*.py"),
        ("automation_v2", "automation/*v2*.py"),
    ):
        code_inventory[label] = sorted(str(p) for p in Path(".").glob(pattern) if p.is_file())
    dataset = compile_dataset(include_legacy=True)
    split = dataset.split()
    return {
        "schema_version": "2.23.0",
        "timestamp": time.time(),
        "models_and_reports": models,
        "data_directories": dirs,
        "module_probes": modules,
        "code_inventory": code_inventory,
        "unified_dataset": dataset.summary(),
        "legacy_import": dataset.legacy_report,
        "split_preview": {
            "development": len(split.development), "validation": len(split.validation),
            "holdout_protected": len(split.holdout), "provisional": split.provisional,
            "notes": split.notes,
        },
        "v223_registry": load_registry(),
        "v223_champion": load_champion_entry(),
        "policy": {
            "live_authority_changed": False,
            "protected_holdout_used_for_auto_selection": False,
            "audio_false_trigger_fix": "parked_todo",
        },
    }


def write_audit_report() -> Path:
    report = audit_repository_state()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    path = REPORT_ROOT / f"inventory_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (REPORT_ROOT / "inventory_latest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path
