#!/usr/bin/env python3
"""Audit the frozen WCF-002 Constant-LR protocol before training.

WCF-002 is deliberately not a new WCF architecture.  Its model/data/optimizer
contract must match WCF-001 exactly; the only training-dynamics change is the
Constant scheduler copied from the actual CTRL-002 config.  Experiment names,
IDs, parent metadata, scheduler, and run directory are allowed bookkeeping
differences.  This check never constructs a model and never touches a split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without(value: Any, paths: set[tuple[str, ...]], prefix: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {
            key: _without(item, paths, prefix + (str(key),))
            for key, item in value.items()
            if prefix + (str(key),) not in paths
        }
    if isinstance(value, list):
        return [
            _without(item, paths, prefix + (str(index),))
            for index, item in enumerate(value)
        ]
    return value


def _diff(left: Any, right: Any, path: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    if type(left) is not type(right):
        return [{"path": ".".join(path), "left": left, "right": right}]
    if isinstance(left, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append({"path": ".".join(path + (str(key),)), "left": left.get(key), "right": right.get(key)})
            else:
                differences.extend(_diff(left[key], right[key], path + (str(key),)))
        return differences
    if isinstance(left, list):
        differences = []
        for index in range(max(len(left), len(right))):
            if index >= len(left) or index >= len(right):
                differences.append({"path": ".".join(path + (str(index),)), "left": left[index] if index < len(left) else None, "right": right[index] if index < len(right) else None})
            else:
                differences.extend(_diff(left[index], right[index], path + (str(index),)))
        return differences
    return [] if left == right else [{"path": ".".join(path), "left": left, "right": right}]


def audit(wcf001_path: Path, ctrl002_path: Path, wcf002_path: Path) -> dict[str, Any]:
    wcf001 = _load(wcf001_path)
    ctrl002 = _load(ctrl002_path)
    wcf002 = _load(wcf002_path)

    required_ids = {
        "wcf001": "DINO-LAYER-WCF-001",
        "ctrl002": "DINO-LAYER-CTRL-002",
        "wcf002": "DINO-LAYER-WCF-002",
    }
    if wcf001.get("experiment_id") != required_ids["wcf001"]:
        raise AssertionError("WCF-001 input is not the expected experiment")
    if ctrl002.get("experiment_id", "DINO-LAYER-CTRL-002") != required_ids["ctrl002"]:
        raise AssertionError("CTRL-002 input is not the expected experiment")
    if wcf002.get("experiment_id") != required_ids["wcf002"]:
        raise AssertionError("WCF-002 config has the wrong experiment ID")

    if wcf001["model"] != wcf002["model"]:
        raise AssertionError("WCF-002 model differs from WCF-001")

    allowed_identity_paths = {
        ("name",),
        ("experiment_id",),
        ("parent_experiment_id",),
        ("control_experiment_id",),
        ("run_dir",),
        ("scheduler",),
    }
    wcf_structure_diffs = _diff(
        _without(wcf001, allowed_identity_paths),
        _without(wcf002, allowed_identity_paths),
    )
    if wcf_structure_diffs:
        raise AssertionError(f"WCF-002 has non-scheduler drift: {wcf_structure_diffs}")

    if wcf002["scheduler"] != ctrl002["scheduler"]:
        raise AssertionError(
            "WCF-002 scheduler is not copied from actual CTRL-002: "
            f"{wcf002['scheduler']} != {ctrl002['scheduler']}"
        )
    if wcf002["scheduler"].get("name") != "constant":
        raise AssertionError("WCF-002 must use the exact Constant scheduler")

    compare_paths = {
        "data_root": ("data_root",),
        "input_size": ("input_size",),
        "num_classes": ("num_classes",),
        "ignore_index": ("ignore_index",),
        "optimizer": ("optimizer",),
        "loss": ("loss",),
        "training": ("training",),
        "augmentation": ("augmentation",),
        "preprocessing": ("preprocessing",),
        "seed": ("runtime", "seed"),
    }
    ctrl002_diffs = {}
    for label, path in compare_paths.items():
        def get(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
            current: Any = payload
            for key in keys:
                current = current[key]
            return current
        left = get(ctrl002, path)
        right = get(wcf002, path)
        differences = _diff(left, right, path)
        if differences:
            ctrl002_diffs[label] = differences
    if ctrl002_diffs:
        raise AssertionError(f"WCF-002 differs from CTRL-002 outside model/scheduler identity: {ctrl002_diffs}")

    return {
        "status": "PASS",
        "experiment_id": "DINO-LAYER-WCF-002",
        "wcf001_config": str(wcf001_path),
        "ctrl002_config": str(ctrl002_path),
        "wcf002_config": str(wcf002_path),
        "sha256": {
            "wcf001": _sha256(wcf001_path),
            "ctrl002": _sha256(ctrl002_path),
            "wcf002": _sha256(wcf002_path),
        },
        "model_diff": "none; exact WCF-001 model block retained",
        "scheduler_source": str(ctrl002_path),
        "scheduler": wcf002["scheduler"],
        "training_dynamics_change": "WCF-001 cosine+warmup+min_lr=2e-5 -> exact CTRL-002 Constant protocol",
        "ctrl002_non_model_diff": ctrl002_diffs,
        "test_accessed": False,
        "formal_training_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wcf001", default="configs/dino_layer_wcf_001_frozen512_50e_cf2e5_seed20260901.json")
    parser.add_argument("--ctrl002", default="configs/dino_layer_ctrl_002_l12x4_seed20260901.json")
    parser.add_argument("--wcf002", default="configs/dino_layer_wcf_002_frozen512_50e_constant_seed20260901.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(_resolve(args.wcf001), _resolve(args.ctrl002), _resolve(args.wcf002))
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
