#!/usr/bin/env python3
"""DEC-DIAG-001 Phase 1 frozen decoder tomography.

The script uses only the CTRL-003 Val-best checkpoint and the 203-image
validation split.  It does not train, create a test loader, alter checkpoint
weights, or persist any monkey-patched module.  Phase 1 contains:

* learned gamma inventory for every actual ResidualDepthwiseBlock;
* TPA branch leave-one-out zeroing after TPA projection/resampling and before
  SAD;
* one-at-a-time temporary identity bypass of every SAD residual block.

All intervention metrics are compared with the same in-process normal forward
of the frozen checkpoint.  The outputs describe dependency/sensitivity of this
trained decoder, not a causal claim about retraining a changed architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEG_ROOT = Path(__file__).resolve().parents[1]
if str(SEG_ROOT) not in sys.path:
    sys.path.insert(0, str(SEG_ROOT))

from config_loader import ModelConfig  # noqa: E402
from dpt import ResidualDepthwiseBlock  # noqa: E402
from osd_metrics import confusion_from_logits, l4_global_metrics  # noqa: E402
from runtime import build_model  # noqa: E402
from train_osd import _load_config, _make_loader, _normalize_input_size, _resolve_path  # noqa: E402


METRIC_KEYS = (
    "mIoU3_report_only_global",
    "IoU_oil_global",
    "IoU_water_global",
    "IoU_others_global",
    "IoU_background_global",
    "mIoU4_all_classes_global",
)

TPA_BRANCHES = (
    ("x8", "decoder.tpa_branch_1"),
    ("x4", "decoder.tpa_branch_2"),
    ("x2", "decoder.tpa_branch_3"),
    ("x1", "decoder.tpa_branch_4"),
)


def _resolve_output(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


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


def _metrics_delta(metrics: dict[str, float], normal: dict[str, float]) -> dict[str, float]:
    return {key: round(float(metrics[key] - normal[key]), 6) for key in METRIC_KEYS}


def _metric_record(metrics: dict[str, float], normal: dict[str, float], images: int) -> dict[str, Any]:
    return {
        "evaluated_images": images,
        "metrics_percent": {key: float(metrics[key]) for key in METRIC_KEYS},
        "delta_to_normal_pp": _metrics_delta(metrics, normal),
        "relative_drop_pp": {
            key: round(float(normal[key] - metrics[key]), 6) for key in METRIC_KEYS
        },
    }


def _evaluate(model: torch.nn.Module, loader, device: torch.device, config: dict[str, Any]) -> tuple[dict[str, float], int]:
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
    return l4_global_metrics(confusion), images


def _module_map(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    return dict(model.named_modules())


def _stage_info(name: str) -> dict[str, Any]:
    mapping = {
        "decoder.sad_intra_1": {"scale": "x8", "stage": "SAD intra", "kind": "intra"},
        "decoder.sad_intra_2": {"scale": "x4", "stage": "SAD intra", "kind": "intra"},
        "decoder.sad_intra_3": {"scale": "x2", "stage": "SAD intra", "kind": "intra"},
        "decoder.sad_intra_4": {"scale": "x1", "stage": "SAD intra", "kind": "intra"},
        "decoder.sad_inter_4": {"scale": "x1", "stage": "SAD top-down order 1/4", "kind": "inter/top-down"},
        "decoder.sad_inter_3": {"scale": "x2", "stage": "SAD top-down order 2/4", "kind": "inter/top-down"},
        "decoder.sad_inter_2": {"scale": "x4", "stage": "SAD top-down order 3/4", "kind": "inter/top-down"},
        "decoder.sad_inter_1": {"scale": "x8", "stage": "SAD top-down order 4/4", "kind": "inter/top-down"},
    }
    return mapping.get(
        name,
        {"scale": None, "stage": "unmapped ResidualDepthwiseBlock", "kind": "unknown"},
    )


def _gamma_inventory(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, ResidualDepthwiseBlock):
            continue
        gamma = module.gamma.detach().float().cpu().reshape(-1)
        if gamma.numel() != 1:
            raise RuntimeError(f"Expected scalar learned gamma at {name}, got {tuple(gamma.shape)}")
        row = {
            "module_name": name,
            "module_class": type(module).__name__,
            "gamma_learned_value": float(gamma.item()),
        }
        row.update(_stage_info(name))
        rows.append(row)
    if not rows:
        raise RuntimeError("No ResidualDepthwiseBlock found in the loaded decoder")
    return rows


@contextmanager
def _zero_tpa_branch(module: torch.nn.Module) -> Iterator[None]:
    def replace_output(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected tensor TPA branch output, got {type(output)!r}")
        return torch.zeros_like(output)

    handle = module.register_forward_hook(replace_output)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def _bypass_residual(module: ResidualDepthwiseBlock) -> Iterator[None]:
    original_forward = module.forward

    def identity(self: ResidualDepthwiseBlock, x: torch.Tensor) -> torch.Tensor:
        return x

    module.forward = types.MethodType(identity, module)
    try:
        yield
    finally:
        module.forward = original_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SEG_ROOT / "configs/dino_layer_ctrl_003_l12x4_seed20260901.json")
    parser.add_argument("--checkpoint", type=Path, default=SEG_ROOT / "runs/dino_layer_ctrl_003_l12x4_seed20260901/best.pth")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "work_dirs/dino_dec_diag_001_20260911/phase1.json")
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    checkpoint_path = _resolve_path(args.checkpoint)
    config = _load_config(config_path)
    raw_model = config["model"]
    if raw_model.get("layer_mapping") != [3, 3, 3, 3]:
        raise ValueError("DEC-DIAG-001 must use CTRL-003 layer_mapping=[3,3,3,3]")
    if bool(raw_model.get("wcf_enabled", False)):
        raise ValueError("DEC-DIAG-001 Phase 1 core is the CTRL-003 decoder, not WCF")
    if config.get("selection_split", "val") != "val":
        raise ValueError("DEC-DIAG-001 is validation-only")

    device = torch.device(args.device or config.get("runtime", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model, backbone = build_model(_model_config(config), str(device))
    model.lock_backbone()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device).eval()
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise RuntimeError("CTRL-003 backbone is not fully frozen")

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

    modules = _module_map(model)
    gamma_rows = _gamma_inventory(model)
    residuals = {
        row["module_name"]: modules[row["module_name"]]
        for row in gamma_rows
    }

    normal_metrics, normal_images = _evaluate(model, loader, device, config)
    if normal_images != 203:
        raise RuntimeError(f"Normal validation evaluated {normal_images} images, expected 203")

    branch_rows = []
    for scale, module_name in TPA_BRANCHES:
        if module_name not in modules:
            raise RuntimeError(f"Expected TPA branch module not found: {module_name}")
        with _zero_tpa_branch(modules[module_name]):
            metrics, images = _evaluate(model, loader, device, config)
        branch_rows.append(
            {
                "branch": scale,
                "module_name": module_name,
                "intervention_location": "after TPA projection/resampling, before SAD intra",
                **_metric_record(metrics, normal_metrics, images),
            }
        )

    residual_rows = []
    for row in gamma_rows:
        module_name = row["module_name"]
        module = residuals[module_name]
        if not isinstance(module, ResidualDepthwiseBlock):
            raise RuntimeError(f"Mapped module changed type: {module_name}")
        with _bypass_residual(module):
            metrics, images = _evaluate(model, loader, device, config)
        residual_rows.append(
            {
                **row,
                "intervention": "temporary forward identity: out=x",
                **_metric_record(metrics, normal_metrics, images),
            }
        )

    result = {
        "status": "completed",
        "experiment_id": "DINO-DEC-DIAG-001",
        "phase": "Phase 1",
        "base_experiment_id": "DINO-LAYER-CTRL-003",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "source_commits": {
            "nested_repo": _git(SEG_ROOT, "rev-parse", "HEAD"),
            "root_repo": _git(PROJECT_ROOT, "rev-parse", "HEAD"),
        },
        "device": str(device),
        "protocol_id": config.get("protocol_id"),
        "split": "val",
        "evaluated_images": 203,
        "test_accessed": False,
        "training_started": False,
        "normal": {
            "evaluated_images": normal_images,
            "metrics_percent": {key: float(normal_metrics[key]) for key in METRIC_KEYS},
        },
        "phase_1a_gamma_inventory": {
            "actual_residual_depthwise_block_count": len(gamma_rows),
            "stage_map": gamma_rows,
            "stage_order": "TPA x8/x4/x2/x1 -> SAD intra -> SAD top-down (x1 -> x2 -> x4 -> x8)",
        },
        "phase_1b_tpa_leave_one_out": branch_rows,
        "phase_1c_sad_residual_bypass": residual_rows,
        "interpretation_boundary": "These interventions measure dependency/sensitivity of the trained CTRL-003 decoder. They are not evidence that retraining a deleted branch/block would have the same result, and do not identify a causal architecture optimum.",
    }
    output = _resolve_output(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output), "normal": result["normal"], "residual_blocks": len(gamma_rows)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
