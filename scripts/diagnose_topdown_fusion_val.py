#!/usr/bin/env python3
"""Read-only diagnosis of Cross-MSEF + R top-down fusion on OSD Val.

This script does not train, modify checkpoints, or access the test split.  It
measures the tensors immediately before the three real top-down additions in
the current L12 + RGB Cross-MSEF decoder and evaluates three inference-only
branch knockouts: P-only, U-only, and the normal P+U merge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEG_ROOT = Path(__file__).resolve().parents[1]
if str(SEG_ROOT) not in sys.path:
    sys.path.insert(0, str(SEG_ROOT))

from diagnose_cdr_val import (  # noqa: E402
    _boundary_mask,
    _checkpoint_state,
    _load_model,
    _model_config,
)
from osd_metrics import CURRENT_CLASSES, confusion_from_logits, l4_global_metrics  # noqa: E402
from train_osd import _load_config, _make_loader, _normalize_input_size, _resolve_path  # noqa: E402


STAGES = ("P8", "P4", "P2")
REGIONS = ("valid", "boundary", "interior", "error", "correct")
QUANTITIES = ("cosine", "norm_ratio", "cancellation_ratio")


class SampledStats:
    """Streaming moments plus a bounded deterministic sample for quantiles."""

    def __init__(self, sample_cap: int = 120_000) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.sample_cap = int(sample_cap)
        self.samples: list[np.ndarray] = []

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float().reshape(-1)
        if values.numel() == 0:
            return
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return
        self.count += int(values.numel())
        self.total += float(values.sum().item())
        self.total_sq += float((values * values).sum().item())
        take = min(1024, int(values.numel()))
        indices = torch.linspace(0, values.numel() - 1, steps=take, device=values.device).long()
        self.samples.append(values[indices].cpu().numpy())
        current = sum(sample.size for sample in self.samples)
        if current > self.sample_cap:
            merged = np.concatenate(self.samples)
            keep = np.linspace(0, merged.size - 1, num=self.sample_cap).astype(np.int64)
            self.samples = [merged[keep]]

    def summary(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "mean": None, "std": None, "p10": None, "median": None, "p90": None}
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        sample = np.concatenate(self.samples) if self.samples else np.empty(0, dtype=np.float64)
        quantiles = np.percentile(sample, [10, 50, 90]) if sample.size else [np.nan, np.nan, np.nan]
        return {
            "count": self.count,
            "mean": float(mean),
            "std": float(variance ** 0.5),
            "p10": float(quantiles[0]) if np.isfinite(quantiles[0]) else None,
            "median": float(quantiles[1]) if np.isfinite(quantiles[1]) else None,
            "p90": float(quantiles[2]) if np.isfinite(quantiles[2]) else None,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SEG_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cross_r_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    knockout: str = "normal",
) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
    """Run the existing Cross-MSEF + R path with optional inference knockout."""
    if knockout not in {"normal", "p_only", "u_only"}:
        raise ValueError(f"unsupported knockout={knockout}")
    patch_h = inputs.shape[-2] // model.patch_size
    patch_w = inputs.shape[-1] // model.patch_size
    final_layer_idx = model.intermediate_layer_idx[model.encoder_size][-1]
    with torch.no_grad():
        semantic_tokens = model.backbone.get_intermediate_layers(
            inputs, n=[final_layer_idx]
        )[-1]
        spatial_features = model.spm_stem(inputs)
        decoder = model.decoder
        projected = decoder._project_l12(semantic_tokens, patch_h, patch_w)
        p2, p4, p8, p16 = decoder._build_bilinear_pyramid(projected)
        p2 = decoder.cross_msef_2(p2, spatial_features[0])
        p4 = decoder.cross_msef_4(p4, spatial_features[1])
        p8 = decoder.cross_msef_8(p8, spatial_features[2])

        level_1 = decoder.sad_intra_1(p2)
        level_2 = decoder.sad_intra_2(p4)
        level_3 = decoder.sad_intra_3(p8)
        level_4 = decoder.sad_intra_4(p16)

        x4 = decoder.sad_inter_4(level_4)
        u8 = F.interpolate(x4, size=level_3.shape[-2:], mode="bilinear", align_corners=False)
        merge8 = level_3 if knockout == "p_only" else u8 if knockout == "u_only" else u8 + level_3
        x3 = decoder.sad_inter_3(merge8)

        u4 = F.interpolate(x3, size=level_2.shape[-2:], mode="bilinear", align_corners=False)
        merge4 = level_2 if knockout == "p_only" else u4 if knockout == "u_only" else u4 + level_2
        x2 = decoder.sad_inter_2(merge4)

        u2 = F.interpolate(x2, size=level_1.shape[-2:], mode="bilinear", align_corners=False)
        merge2 = level_1 if knockout == "p_only" else u2 if knockout == "u_only" else u2 + level_1
        x1 = decoder.sad_inter_1(merge2)

        lowres_logits = decoder.out_conv(x1)
        logits = F.interpolate(
            lowres_logits, size=inputs.shape[-2:], mode="bilinear", align_corners=False
        )
    trace = {
        "P8": {"P": level_3, "U": u8, "fused": u8 + level_3},
        "P4": {"P": level_2, "U": u4, "fused": u4 + level_2},
        "P2": {"P": level_1, "U": u2, "fused": u2 + level_1},
    }
    return logits, trace


def _mask_at(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(mask.float().unsqueeze(1), size=size, mode="nearest").squeeze(1) > 0.5


def _init_stats() -> dict[str, dict[str, dict[str, SampledStats]]]:
    return {
        stage: {
            region: {quantity: SampledStats() for quantity in QUANTITIES}
            for region in REGIONS
        }
        for stage in STAGES
    }


def _update_stage_stats(
    stage_stats: dict[str, dict[str, SampledStats]],
    p: torch.Tensor,
    u: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    ignore_index: int,
) -> None:
    cosine = F.cosine_similarity(p.float(), u.float(), dim=1, eps=1e-8)
    p_norm = p.float().flatten(1).norm(dim=1)  # only used for shape validation below
    del p_norm
    norm_ratio = u.float().norm(dim=1) / p.float().norm(dim=1).clamp_min(1e-8)
    cancellation = (p.float() + u.float()).norm(dim=1) / (
        p.float().norm(dim=1) + u.float().norm(dim=1) + 1e-8
    )

    valid_full = target != ignore_index
    boundary_full = _boundary_mask(target, ignore_index)
    interior_full = valid_full & ~boundary_full
    error_full = valid_full & (prediction != target)
    correct_full = valid_full & (prediction == target)
    masks_full = {
        "valid": valid_full,
        "boundary": boundary_full,
        "interior": interior_full,
        "error": error_full,
        "correct": correct_full,
    }
    size = tuple(cosine.shape[-2:])
    for region, mask_full in masks_full.items():
        mask = _mask_at(mask_full, size)
        for quantity, value in (
            ("cosine", cosine),
            ("norm_ratio", norm_ratio),
            ("cancellation_ratio", cancellation),
        ):
            stage_stats[region][quantity].update(value[mask])

    for class_index, class_name in enumerate(CURRENT_CLASSES):
        class_mask = _mask_at(valid_full & (target == class_index), size)
        region_key = f"class_{class_name}"
        if region_key not in stage_stats:
            stage_stats[region_key] = {quantity: SampledStats() for quantity in QUANTITIES}
        for quantity, value in (
            ("cosine", cosine),
            ("norm_ratio", norm_ratio),
            ("cancellation_ratio", cancellation),
        ):
            stage_stats[region_key][quantity].update(value[class_mask])


def _finalize_stats(stats: dict[str, dict[str, dict[str, SampledStats]]]) -> dict[str, Any]:
    return {
        stage: {
            region: {quantity: accumulator.summary() for quantity, accumulator in quantities.items()}
            for region, quantities in regions.items()
        }
        for stage, regions in stats.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=SEG_ROOT / "configs/dino_l12_spm_cross_r_osd_512.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SEG_ROOT / "runs/dino_l12_spm_cross_r_osd_512_seed20260901/best.pth",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "work_dirs/dino_dec_diag_001_cross_r_val.json",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    checkpoint_path = _resolve_path(args.checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    config = _load_config(config_path)
    if config.get("model", {}).get("decoder_variant") != "l12_a_cross_r":
        raise RuntimeError("This read-only diagnostic is fixed to the Cross-MSEF + original R parent")
    if config.get("selection_split", "val") != "val":
        raise RuntimeError("The diagnostic must use Val as the selection split")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    model = _load_model(config, checkpoint_path, device)
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("Backbone is not frozen")
    model.eval()

    input_size = _normalize_input_size(config["input_size"])
    loader = _make_loader(
        _resolve_path(config["data_root"]),
        config["val_split"],
        input_size,
        False,
        int(config["training"].get("eval_batch_size", 1)),
        int(args.workers),
        config,
        device,
    )
    if len(loader.dataset) != 203:
        raise RuntimeError(f"Expected Val=203 images, found {len(loader.dataset)}")

    num_classes = int(config["num_classes"])
    ignore_index = int(config["ignore_index"])
    normal_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    knockout_confusions = {
        mode: np.zeros((num_classes, num_classes), dtype=np.int64)
        for mode in ("p_only", "u_only")
    }
    stats = {stage: _init_stats()[stage] for stage in STAGES}

    with torch.inference_mode():
        for inputs, targets, _sample_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            normal_logits, trace = _cross_r_forward(model, inputs, knockout="normal")
            p_only_logits, _ = _cross_r_forward(model, inputs, knockout="p_only")
            u_only_logits, _ = _cross_r_forward(model, inputs, knockout="u_only")
            prediction = normal_logits.argmax(dim=1)
            normal_confusion += confusion_from_logits(normal_logits, targets, num_classes, ignore_index)
            knockout_confusions["p_only"] += confusion_from_logits(p_only_logits, targets, num_classes, ignore_index)
            knockout_confusions["u_only"] += confusion_from_logits(u_only_logits, targets, num_classes, ignore_index)
            for stage in STAGES:
                _update_stage_stats(
                    stats[stage],
                    trace[stage]["P"],
                    trace[stage]["U"],
                    targets,
                    prediction,
                    ignore_index,
                )

    metrics = {"normal_p_plus_u": l4_global_metrics(normal_confusion)}
    metrics.update({mode: l4_global_metrics(confusion) for mode, confusion in knockout_confusions.items()})
    result = {
        "status": "completed",
        "purpose": "read-only top-down fusion diagnosis on OSD Val",
        "training_started": False,
        "test_accessed": False,
        "weights_modified": False,
        "git_head": _git_head(),
        "device": str(device),
        "split": "val",
        "evaluated_images": len(loader.dataset),
        "input_size": list(input_size),
        "parent": {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "decoder_variant": config["model"]["decoder_variant"],
        },
        "definitions": {
            "P": "current-scale SAD intra output immediately before top-down add",
            "U": "bilinear upsampled output of the immediately coarser SAD stage",
            "cancellation_ratio": "||P+U||/(||P||+||U||)",
            "knockouts": "inference-only P-only or U-only at all three merges; not model variants",
        },
        "topdown_statistics": _finalize_stats(stats),
        "branch_knockout_metrics_percent": metrics,
    }
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"output={output_path}")
    print("test_accessed=False training_started=False")
    for name, values in metrics.items():
        print(
            f"{name}: mIoU3={values['mIoU3_report_only_global']:.6f}% "
            f"mF1_3={values['mF1_3_report_only_global']:.6f}% "
            f"PixelAcc={values['PixelAccuracy_global']:.6f}%"
        )
    for stage in STAGES:
        overall = result["topdown_statistics"][stage]["valid"]
        print(
            f"{stage}: cosine={overall['cosine']['mean']:.6f} "
            f"norm_ratio={overall['norm_ratio']['mean']:.6f} "
            f"cancellation={overall['cancellation_ratio']['mean']:.6f}"
        )


if __name__ == "__main__":
    main()
