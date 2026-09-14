"""Inference-only local shift probe for the Cross-MSEF+R top-down path.

For each of P8/P4/P2, shift only the upsampled coarse feature by one feature
grid cell in four directions and keep the other two merges unchanged.  This
is a cheap test of whether a fixed spatial alignment error is visible in the
current trained checkpoint; it is not a trained model and never reads Test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEG_ROOT = Path(__file__).resolve().parents[1]
if str(SEG_ROOT) not in sys.path:
    sys.path.insert(0, str(SEG_ROOT))

from diagnose_cdr_val import _load_model  # noqa: E402
from osd_metrics import confusion_from_logits, l4_global_metrics  # noqa: E402
from train_osd import _load_config, _make_loader, _normalize_input_size, _resolve_path  # noqa: E402


STAGES = ("P8", "P4", "P2")
SHIFTS = ((-1, 0), (1, 0), (0, -1), (0, 1))


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


def _shift_with_replicate(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    if abs(int(dy)) > 1 or abs(int(dx)) > 1:
        raise ValueError("This probe only supports one-grid-cell shifts")
    height, width = x.shape[-2:]
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    y0 = 1 - int(dy)
    x0 = 1 - int(dx)
    return padded[..., y0 : y0 + height, x0 : x0 + width]


def _prepare_parent_path(model, inputs):
    patch_h = inputs.shape[-2] // model.patch_size
    patch_w = inputs.shape[-1] // model.patch_size
    final_layer_idx = model.intermediate_layer_idx[model.encoder_size][-1]
    decoder = model.decoder
    semantic_tokens = model.backbone.get_intermediate_layers(
        inputs, n=[final_layer_idx]
    )[-1]
    spatial_features = model.spm_stem(inputs)
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
    x3 = decoder.sad_inter_3(u8 + level_3)
    u4 = F.interpolate(x3, size=level_2.shape[-2:], mode="bilinear", align_corners=False)
    x2 = decoder.sad_inter_2(u4 + level_2)
    u2 = F.interpolate(x2, size=level_1.shape[-2:], mode="bilinear", align_corners=False)
    return {
        "decoder": decoder,
        "level_1": level_1,
        "level_2": level_2,
        "level_3": level_3,
        "x4": x4,
        "u8": u8,
        "x3_normal": x3,
        "u4_normal": u4,
        "x2_normal": x2,
        "u2_normal": u2,
    }


def _logits_for_shift(path, stage, dy, dx, output_size):
    decoder = path["decoder"]
    u8 = path["u8"]
    if stage == "P8":
        u8 = _shift_with_replicate(u8, dy, dx)
    x3 = decoder.sad_inter_3(path["level_3"] + u8)

    u4 = F.interpolate(x3, size=path["level_2"].shape[-2:], mode="bilinear", align_corners=False)
    if stage == "P4":
        u4 = _shift_with_replicate(u4, dy, dx)
    x2 = decoder.sad_inter_2(path["level_2"] + u4)

    u2 = F.interpolate(x2, size=path["level_1"].shape[-2:], mode="bilinear", align_corners=False)
    if stage == "P2":
        u2 = _shift_with_replicate(u2, dy, dx)
    x1 = decoder.sad_inter_1(path["level_1"] + u2)
    return F.interpolate(
        decoder.out_conv(x1), size=output_size, mode="bilinear", align_corners=False
    )


def main():
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "work_dirs/dino_dec_shift_sweep_val.json",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    checkpoint_path = _resolve_path(args.checkpoint)
    config = _load_config(config_path)
    if config.get("model", {}).get("decoder_variant") != "l12_a_cross_r":
        raise RuntimeError("The shift probe is fixed to the Cross-MSEF + original R parent")
    if config.get("selection_split", "val") != "val":
        raise RuntimeError("The shift probe must use Val as the selection split")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    model = _load_model(config, checkpoint_path, device)
    model.eval()
    input_size = _normalize_input_size(config["input_size"])
    loader = _make_loader(
        _resolve_path(config["data_root"]),
        config["val_split"],
        input_size,
        False,
        int(args.batch_size),
        int(args.workers),
        config,
        device,
    )
    if len(loader.dataset) != 203:
        raise RuntimeError(f"Expected Val=203 images, found {len(loader.dataset)}")

    num_classes = int(config["num_classes"])
    ignore_index = int(config["ignore_index"])
    candidates = ["normal"] + [
        f"{stage}/dy={dy:+d},dx={dx:+d}"
        for stage in STAGES
        for dy, dx in SHIFTS
    ]
    confusions = {name: np.zeros((num_classes, num_classes), dtype=np.int64) for name in candidates}

    with torch.inference_mode():
        for inputs, targets, _sample_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            path = _prepare_parent_path(model, inputs)
            decoder = path["decoder"]
            normal = F.interpolate(
                decoder.out_conv(
                    decoder.sad_inter_1(
                        path["u2_normal"] + path["level_1"]
                    )
                ),
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            confusions["normal"] += confusion_from_logits(
                normal, targets, num_classes, ignore_index
            )
            for stage in STAGES:
                for dy, dx in SHIFTS:
                    name = f"{stage}/dy={dy:+d},dx={dx:+d}"
                    logits = _logits_for_shift(path, stage, dy, dx, inputs.shape[-2:])
                    confusions[name] += confusion_from_logits(
                        logits, targets, num_classes, ignore_index
                    )

    metrics = {name: l4_global_metrics(confusion) for name, confusion in confusions.items()}
    baseline = metrics["normal"]["mIoU3_report_only_global"]
    best_by_stage = {}
    for stage in STAGES:
        choices = {
            name: value["mIoU3_report_only_global"]
            for name, value in metrics.items()
            if name.startswith(stage + "/")
        }
        best_name = max(choices, key=choices.get)
        best_by_stage[stage] = {
            "name": best_name,
            "mIoU3": choices[best_name],
            "delta_pp": choices[best_name] - baseline,
        }
    result = {
        "status": "completed",
        "purpose": "inference-only one-grid-cell top-down shift sweep on OSD Val",
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
        "definition": "Shift only the upsampled U at one merge by one local feature-grid cell; use replicate boundary; other merges stay normal.",
        "shifts": [list(value) for value in SHIFTS],
        "metrics_percent": metrics,
        "baseline_mIoU3": baseline,
        "best_by_stage": best_by_stage,
    }
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"output={output_path}")
    print("test_accessed=False training_started=False")
    for stage, value in best_by_stage.items():
        print(
            f"{stage}: best={value['name']} mIoU3={value['mIoU3']:.6f}% "
            f"delta={value['delta_pp']:+.6f}pp"
        )
    print(f"normal: mIoU3={baseline:.6f}%")


if __name__ == "__main__":
    main()
