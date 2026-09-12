#!/usr/bin/env python3
"""Post-hoc paired diagnosis of the parent/DPA validation-test gap.

This script evaluates two already-frozen checkpoints on the same deterministic
OSD transform and pairs results by sample id.  It never trains, selects a
checkpoint, or changes either model.  Test is accessed only as a frozen,
post-hoc diagnostic split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import ModelConfig  # noqa: E402
from osd_metrics import CURRENT_CLASSES, REPORT_CLASSES, confusion_from_logits, l4_global_metrics  # noqa: E402
from runtime import build_model  # noqa: E402
from train_osd import _load_config, _make_loader, _normalize_input_size, _resolve_path  # noqa: E402


def _model_config(raw: dict[str, Any]) -> ModelConfig:
    model = raw["model"]
    return ModelConfig(
        dino_size=model["dino_size"],
        dino_repo=str(_resolve_path(model["dino_repo"])),
        dino_ckpt=str(_resolve_path(model["dino_ckpt"])),
        decoder_dim=int(model["decoder_dim"]),
        use_bn=bool(model.get("use_bn", False)),
        num_classes=int(raw["num_classes"]),
        patch_size=int(model.get("patch_size", 16)),
        decoder_variant=str(model.get("decoder_variant", "tpa_sad")),
        spatial_stride=int(model.get("spatial_stride", 4)),
        freeze_backbone=bool(model.get("freeze_backbone", True)),
        layer_mapping=model.get("layer_mapping"),
        adaptive_readout=bool(model.get("adaptive_readout", False)),
        readout_mode=str(model.get("readout_mode", "matrix")),
        readout_init=str(model.get("readout_init", "uniform")),
        readout_temperature=float(model.get("readout_temperature", 1.0)),
        wcf_enabled=bool(model.get("wcf_enabled", False)),
        wcf_reduction=int(model.get("wcf_reduction", 4)),
        wcf_alpha_init=float(model.get("wcf_alpha_init", 1e-2)),
    )


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload at {path}: {type(state)!r}")
    return state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_confusion(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int) -> np.ndarray:
    return confusion_from_logits(logits, targets, num_classes, ignore_index)


def _class_iou(confusion: np.ndarray, class_index: int) -> float:
    tp = float(confusion[class_index, class_index])
    fp = float(confusion[:, class_index].sum() - tp)
    fn = float(confusion[class_index, :].sum() - tp)
    denominator = tp + fp + fn
    return tp / denominator if denominator else float("nan")


def _class_f1(confusion: np.ndarray, class_index: int) -> float:
    tp = float(confusion[class_index, class_index])
    denominator = float(confusion[class_index, :].sum() + confusion[:, class_index].sum())
    return 2.0 * tp / denominator if denominator else float("nan")


def _sample_metrics(confusion: np.ndarray) -> dict[str, float]:
    iou = {_name: _class_iou(confusion, index) for index, _name in enumerate(CURRENT_CLASSES)}
    f1 = {_name: _class_f1(confusion, index) for index, _name in enumerate(CURRENT_CLASSES)}
    return {
        **{f"IoU_{name}": iou[name] for name in CURRENT_CLASSES},
        **{f"F1_{name}": f1[name] for name in CURRENT_CLASSES},
        "mIoU3": float(np.nanmean([iou[name] for name in REPORT_CLASSES])),
        "mF1_3": float(np.nanmean([f1[name] for name in REPORT_CLASSES])),
    }


def _safe_float(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return _safe_float(value)


def _summary(values: list[Any]) -> dict[str, Any]:
    finite_values = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            finite_values.append(value)
    array = np.asarray(finite_values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None,
                "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _group_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    output = {}
    for group, items in sorted(groups.items()):
        delta = [item["delta_mIoU3"] for item in items]
        finite_delta = [
            float(value)
            for value in delta
            if value is not None and math.isfinite(float(value))
        ]
        output[group] = {
            "images": len(items),
            "delta_mIoU3": _summary(delta),
            "improved": sum(value > 0 for value in finite_delta),
            "worsened": sum(value < 0 for value in finite_delta),
            "unchanged": sum(value == 0 for value in finite_delta),
            "delta_IoU": {
                name: _summary([item[f"delta_IoU_{name}"] for item in items])
                for name in REPORT_CLASSES
            },
        }
    return output


def _load_model(raw: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model, _ = build_model(_model_config(raw), str(device))
    if bool(raw["model"].get("freeze_backbone", True)):
        model.lock_backbone()
    state = _checkpoint_state(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {raw.get('name')}: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-config", type=Path, default=ROOT_DIR / "configs/dino_tpa_base_msmlp_osd_512.json")
    parser.add_argument("--parent-checkpoint", type=Path, default=ROOT_DIR / "runs/dino_tpa_base_msmlp_osd_512_seed20260901/best.pth")
    parser.add_argument("--dpa-config", type=Path, default=ROOT_DIR / "configs/dino_dpa_base_msmlp_osd_512.json")
    parser.add_argument("--dpa-checkpoint", type=Path, default=ROOT_DIR / "runs/dino_dpa_base_msmlp_osd_512_seed20260901/best.pth")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT_DIR.parent / "work_dirs/dino_dpa_base_msmlp_osd_512_gap_diagnosis.json")
    args = parser.parse_args()

    parent_config_path = args.parent_config.resolve()
    parent_checkpoint_path = args.parent_checkpoint.resolve()
    dpa_config_path = args.dpa_config.resolve()
    dpa_checkpoint_path = args.dpa_checkpoint.resolve()
    parent_raw = _load_config(parent_config_path)
    dpa_raw = _load_config(dpa_config_path)
    for path in (parent_checkpoint_path, dpa_checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if parent_raw["data_root"] != dpa_raw["data_root"]:
        raise RuntimeError("Parent and DPA data roots differ")
    if parent_raw["input_size"] != dpa_raw["input_size"]:
        raise RuntimeError("Parent and DPA input sizes differ")
    if parent_raw["num_classes"] != dpa_raw["num_classes"] or parent_raw["ignore_index"] != dpa_raw["ignore_index"]:
        raise RuntimeError("Parent and DPA label settings differ")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(20260901)
    parent_model = _load_model(parent_raw, parent_checkpoint_path, device)
    dpa_model = _load_model(dpa_raw, dpa_checkpoint_path, device)

    input_size = _normalize_input_size(parent_raw["input_size"])
    num_classes = int(parent_raw["num_classes"])
    ignore_index = int(parent_raw["ignore_index"])
    training = parent_raw["training"]
    result: dict[str, Any] = {
        "purpose": "post-hoc paired parent-vs-DPA Val/Test gap diagnosis",
        "selection_or_tuning": False,
        "test_used_for_model_selection": False,
        "same_deterministic_eval_transform": True,
        "device": str(device),
        "input_size": list(input_size),
        "class_order_current": list(CURRENT_CLASSES),
        "class_order_report": list(REPORT_CLASSES),
        "parent": {
            "config": str(parent_config_path),
            "checkpoint": str(parent_checkpoint_path),
            "checkpoint_sha256": _sha256(parent_checkpoint_path),
            "name": parent_raw.get("name"),
        },
        "dpa": {
            "config": str(dpa_config_path),
            "checkpoint": str(dpa_checkpoint_path),
            "checkpoint_sha256": _sha256(dpa_checkpoint_path),
            "name": dpa_raw.get("name"),
        },
        "splits": {},
    }

    for split in ("val", "test"):
        loader = _make_loader(
            _resolve_path(parent_raw["data_root"]),
            parent_raw.get(f"{split}_split", split),
            input_size,
            False,
            1,
            int(args.workers),
            parent_raw,
            device,
        )
        parent_global = np.zeros((num_classes, num_classes), dtype=np.int64)
        dpa_global = np.zeros((num_classes, num_classes), dtype=np.int64)
        records: list[dict[str, Any]] = []
        for inputs, targets, sample_ids in loader:
            if len(sample_ids) != 1:
                raise RuntimeError("This paired diagnostic requires batch size 1")
            sample_id = str(sample_ids[0])
            inputs_device = inputs.to(device, non_blocking=True)
            with torch.inference_mode():
                parent_logits = parent_model(inputs_device)
                dpa_logits = dpa_model(inputs_device)
            parent_conf = _sample_confusion(parent_logits, targets, num_classes, ignore_index)
            dpa_conf = _sample_confusion(dpa_logits, targets, num_classes, ignore_index)
            parent_global += parent_conf
            dpa_global += dpa_conf
            parent_metrics = _sample_metrics(parent_conf)
            dpa_metrics = _sample_metrics(dpa_conf)
            valid = targets[0] != ignore_index
            gt_counts = torch.bincount(targets[0][valid].flatten(), minlength=num_classes).numpy()
            report_counts = {name: int(gt_counts[index]) for index, name in enumerate(CURRENT_CLASSES)}
            report_pixels = max(1, sum(report_counts[name] for name in REPORT_CLASSES))
            report_fraction = {
                name: report_counts[name] / report_pixels for name in REPORT_CLASSES
            }
            dominant = max(report_fraction, key=report_fraction.get)
            record = {
                "sample_id": sample_id,
                "gt_pixels": report_counts,
                "gt_report_fraction": report_fraction,
                "gt_dominant_report_class": dominant,
                "gt_present_report_classes": [name for name in REPORT_CLASSES if report_counts[name] > 0],
                "parent": parent_metrics,
                "dpa": dpa_metrics,
                "delta_mIoU3": dpa_metrics["mIoU3"] - parent_metrics["mIoU3"],
            }
            for name in REPORT_CLASSES:
                record[f"delta_IoU_{name}"] = dpa_metrics[f"IoU_{name}"] - parent_metrics[f"IoU_{name}"]
                record[f"delta_F1_{name}"] = dpa_metrics[f"F1_{name}"] - parent_metrics[f"F1_{name}"]
            records.append(record)

        parent_metrics_global = l4_global_metrics(parent_global)
        dpa_metrics_global = l4_global_metrics(dpa_global)
        delta_global = {
            key: dpa_metrics_global[key] - parent_metrics_global[key]
            for key in dpa_metrics_global
            if key in parent_metrics_global
        }
        delta_values = [record["delta_mIoU3"] for record in records]
        finite_delta_values = [
            float(value)
            for value in delta_values
            if value is not None and math.isfinite(float(value))
        ]
        improved = sum(value > 0 for value in finite_delta_values)
        worsened = sum(value < 0 for value in finite_delta_values)
        unchanged = sum(value == 0 for value in finite_delta_values)
        ranked = sorted(
            records,
            key=lambda item: (
                float(item["delta_mIoU3"])
                if item["delta_mIoU3"] is not None and math.isfinite(float(item["delta_mIoU3"]))
                else float("-inf")
            ),
        )
        per_sample_parent = {
            "mIoU3": _summary([item["parent"]["mIoU3"] for item in records]),
            **{f"IoU_{name}": _summary([item["parent"][f"IoU_{name}"] for item in records]) for name in REPORT_CLASSES},
        }
        per_sample_dpa = {
            "mIoU3": _summary([item["dpa"]["mIoU3"] for item in records]),
            **{
                f"IoU_{name}": _summary(
                    [item["dpa"][f"IoU_{name}"] for item in records]
                )
                for name in REPORT_CLASSES
            },
        }
        result["splits"][split] = {
            "evaluated_images": len(records),
            "global": {"parent": parent_metrics_global, "dpa": dpa_metrics_global, "delta_dpa_minus_parent": delta_global},
            "per_sample": {"parent": per_sample_parent, "dpa": per_sample_dpa, "delta_mIoU3": _summary(delta_values)},
            "per_sample_counts": {"dpa_improved": improved, "parent_better": worsened, "unchanged": unchanged},
            "grouped_by_gt_dominant_report_class": _group_summary(records, "gt_dominant_report_class"),
            "grouped_by_gt_present_report_classes": _group_summary(
                records,
                "gt_present_report_classes",
            ),
            "top_10_dpa_improvements": ranked[-10:][::-1],
            "top_10_dpa_regressions": ranked[:10],
            "samples": records,
        }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_clean(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"output={output_path}")
    print(f"selection_or_tuning={result['selection_or_tuning']} test_used_for_model_selection={result['test_used_for_model_selection']}")
    for split, payload in result["splits"].items():
        global_delta = payload["global"]["delta_dpa_minus_parent"]
        print(
            f"{split}: images={payload['evaluated_images']} "
            f"parent_mIoU3={payload['global']['parent']['mIoU3_report_only_global']:.6f}% "
            f"dpa_mIoU3={payload['global']['dpa']['mIoU3_report_only_global']:.6f}% "
            f"delta={global_delta['mIoU3_report_only_global']:+.6f}pp "
            f"sample_improved={payload['per_sample_counts']['dpa_improved']} "
            f"parent_better={payload['per_sample_counts']['parent_better']}"
        )


if __name__ == "__main__":
    main()
