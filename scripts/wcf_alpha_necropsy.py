#!/usr/bin/env python3
"""Val-only alpha intervention for the completed WCF-001 checkpoint.

This is a post-hoc dependency check, not a retraining experiment.  It runs the
frozen WCF-001 Val-best checkpoint on all 203 validation images under the
normal setting and four explicitly fixed alpha interventions.  The test split
is intentionally not exposed by this CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import ModelConfig  # noqa: E402
from osd_metrics import confusion_from_logits, l4_global_metrics  # noqa: E402
from runtime import build_model  # noqa: E402
from train_osd import _load_config, _make_loader, _normalize_input_size, _resolve_path  # noqa: E402


INTERVENTIONS = (
    ("normal", ()),
    ("alpha3_zero", (0,)),
    ("alpha6_zero", (1,)),
    ("alpha9_zero", (2,)),
    ("all_alpha_zero", (0, 1, 2)),
)
METRIC_KEYS = (
    "mIoU3_report_only_global",
    "IoU_oil_global",
    "IoU_water_global",
    "IoU_others_global",
    "IoU_background_global",
    "mIoU4_all_classes_global",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_config(config: dict[str, Any]) -> ModelConfig:
    raw = config["model"]
    return ModelConfig(
        dino_size=raw["dino_size"],
        dino_repo=str(_resolve_path(raw["dino_repo"])),
        dino_ckpt=str(_resolve_path(raw["dino_ckpt"])),
        decoder_dim=int(raw["decoder_dim"]),
        use_bn=bool(raw.get("use_bn", False)),
        num_classes=int(config["num_classes"]),
        patch_size=int(raw.get("patch_size", 16)),
        layer_mapping=raw.get("layer_mapping"),
        adaptive_readout=bool(raw.get("adaptive_readout", False)),
        readout_mode=str(raw.get("readout_mode", "matrix")),
        readout_init=str(raw.get("readout_init", "uniform")),
        readout_temperature=float(raw.get("readout_temperature", 1.0)),
        wcf_enabled=bool(raw.get("wcf_enabled", False)),
        wcf_reduction=int(raw.get("wcf_reduction", 4)),
        wcf_alpha_init=float(raw.get("wcf_alpha_init", 1e-2)),
    )


def _evaluate(model: torch.nn.Module, loader, device: torch.device, config: dict[str, Any]) -> tuple[dict[str, float], np.ndarray, int]:
    confusion = np.zeros((int(config["num_classes"]), int(config["num_classes"])), dtype=np.int64)
    images = 0
    model.eval()
    with torch.inference_mode():
        for inputs, targets, _ in loader:
            logits = model(inputs.to(device, non_blocking=True))
            confusion += confusion_from_logits(
                logits,
                targets,
                int(config["num_classes"]),
                int(config["ignore_index"]),
            )
            images += int(inputs.shape[0])
    return l4_global_metrics(confusion), confusion, images


def _set_alphas(model: torch.nn.Module, values: list[float]) -> None:
    blocks = list(model.get_wcf_blocks())
    if len(blocks) != 3:
        raise AssertionError(f"Expected exactly three WCF blocks, got {len(blocks)}")
    with torch.no_grad():
        for block, value in zip(blocks, values):
            block.alpha.fill_(float(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dino_layer_wcf_001_frozen512_50e_cf2e5_seed20260901.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/dino_layer_wcf_001_frozen512_50e_cf2e5_seed20260901/best.pth")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    checkpoint_path = _resolve_path(args.checkpoint)
    config = _load_config(config_path)
    raw_model = config["model"]
    if not bool(raw_model.get("wcf_enabled", False)):
        raise ValueError("WCF-001 config must have model.wcf_enabled=true")
    if config.get("selection_split", "val") != "val":
        raise ValueError("WCF alpha necropsy is validation-only and requires selection_split=val")
    if int(config.get("num_classes", 0)) != 4:
        raise ValueError("Expected the formal four-class OSD evaluator")

    device = torch.device(args.device or config.get("runtime", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = build_model(_model_config(config), str(device))
    if bool(raw_model.get("freeze_backbone", True)):
        model.lock_backbone()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device).eval()

    input_size = _normalize_input_size(config["input_size"])
    loader = _make_loader(
        _resolve_path(config["data_root"]),
        config.get("val_split", "val"),
        input_size,
        False,
        int(config["training"].get("eval_batch_size", 1)),
        int(args.workers),
        config,
        device,
    )
    if len(loader.dataset) != 203:
        raise RuntimeError(f"Expected val=203 images, found {len(loader.dataset)}")

    normal_alphas = [float(block.alpha.detach().cpu()) for block in model.get_wcf_blocks()]
    rows: list[dict[str, Any]] = []
    normal_metrics: dict[str, float] | None = None
    for name, zero_indices in INTERVENTIONS:
        values = list(normal_alphas)
        for index in zero_indices:
            values[index] = 0.0
        _set_alphas(model, values)
        metrics, confusion, images = _evaluate(model, loader, device, config)
        if images != 203:
            raise RuntimeError(f"{name} evaluated {images} images, expected 203")
        if normal_metrics is None:
            normal_metrics = metrics
        deltas = {key: round(float(metrics[key] - normal_metrics[key]), 6) for key in METRIC_KEYS}
        drops = {key: round(float(normal_metrics[key] - metrics[key]), 6) for key in METRIC_KEYS}
        rows.append(
            {
                "condition": name,
                "zeroed_blocks": [
                    f"alpha{alpha_name}"
                    for block_index, alpha_name in enumerate((3, 6, 9))
                    if block_index in zero_indices
                ],
                "effective_alphas": values,
                "evaluated_images": images,
                "metrics_percent": {key: float(metrics[key]) for key in METRIC_KEYS},
                "delta_to_normal_pp": deltas,
                "relative_drop_pp": drops,
                "confusion_matrix_gt_rows_pred_columns": confusion.tolist(),
            }
        )

    _set_alphas(model, normal_alphas)
    result = {
        "status": "completed",
        "experiment_id": "DINO-LAYER-WCF-001",
        "diagnostic_id": "DINO-LAYER-WCF-001-alpha-necropsy",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "split": "val",
        "evaluated_images": 203,
        "test_accessed": False,
        "retrained": False,
        "interpretation_boundary": "Dependency/sensitivity of the trained WCF checkpoint; not a causal claim about retraining without a block.",
        "all_alpha_zero_boundary": "WCF-trained decoder with all alpha values zero; not CTRL-003.",
        "original_checkpoint_alphas": normal_alphas,
        "conditions": rows,
    }
    output = _resolve_path(args.output) if args.output is not None else _resolve_path(config.get("run_dir", "./runs/osd")) / "wcf_val_alpha_necropsy.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "conditions": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
