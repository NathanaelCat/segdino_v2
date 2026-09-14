"""Inference-only sweep of P + alpha*U on the Cross-MSEF+R Val checkpoint.

The sweep changes no weights and never reads the test split.  The backbone and
the Cross-MSEF/R decoder front end are executed once per batch; only the three
top-down merge coefficients are varied in the already computed feature path.
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
    x3_normal = decoder.sad_inter_3(u8 + level_3)
    u4 = F.interpolate(x3_normal, size=level_2.shape[-2:], mode="bilinear", align_corners=False)
    x2_normal = decoder.sad_inter_2(u4 + level_2)
    u2 = F.interpolate(x2_normal, size=level_1.shape[-2:], mode="bilinear", align_corners=False)
    return {
        "decoder": decoder,
        "level_1": level_1,
        "level_2": level_2,
        "level_3": level_3,
        "x4": x4,
        "u8": u8,
        "u4_normal": u4,
        "u2_normal": u2,
        "x3_normal": x3_normal,
        "x2_normal": x2_normal,
    }


def _logits_for_alphas(path, alphas, output_size):
    decoder = path["decoder"]
    a8, a4, a2 = [float(value) for value in alphas]
    x3 = decoder.sad_inter_3(path["level_3"] + a8 * path["u8"])
    # The cached normal u4 is valid only when P8 also stayed at alpha=1.
    # If P8 changes, its new x3 must be propagated through the P4 merge.
    if a4 == 1.0 and a8 == 1.0:
        u4 = path["u4_normal"]
    else:
        u4 = F.interpolate(x3, size=path["level_2"].shape[-2:], mode="bilinear", align_corners=False)
    x2 = decoder.sad_inter_2(path["level_2"] + a4 * u4)
    if a2 == 1.0 and a4 == 1.0 and a8 == 1.0:
        u2 = path["u2_normal"]
    else:
        u2 = F.interpolate(x2, size=path["level_1"].shape[-2:], mode="bilinear", align_corners=False)
    x1 = decoder.sad_inter_1(path["level_1"] + a2 * u2)
    logits = decoder.out_conv(x1)
    return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


def _key(stage, value):
    return f"{stage}/alpha={float(value):g}"


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
        "--alphas",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 1.0, 1.25],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "work_dirs/dino_dec_alpha_sweep_val.json",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    checkpoint_path = _resolve_path(args.checkpoint)
    config = _load_config(config_path)
    if config.get("model", {}).get("decoder_variant") != "l12_a_cross_r":
        raise RuntimeError("The sweep is fixed to the Cross-MSEF + original R parent")
    if config.get("selection_split", "val") != "val":
        raise RuntimeError("The sweep must use Val as the selection split")
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
    values = [float(value) for value in args.alphas]
    names = [_key(stage, value) for stage in STAGES for value in values]
    confusions = {name: np.zeros((num_classes, num_classes), dtype=np.int64) for name in names}
    confusions["normal"] = np.zeros((num_classes, num_classes), dtype=np.int64)
    normal_max_diff = 0.0

    with torch.inference_mode():
        for inputs, targets, _sample_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            path = _prepare_parent_path(model, inputs)
            normal = _logits_for_alphas(path, (1.0, 1.0, 1.0), inputs.shape[-2:])
            direct = model(inputs)
            normal_max_diff = max(normal_max_diff, float((normal - direct).abs().max().item()))
            confusions["normal"] += confusion_from_logits(
                normal, targets, num_classes, ignore_index
            )
            for stage_index, stage in enumerate(STAGES):
                for value in values:
                    alphas = [1.0, 1.0, 1.0]
                    alphas[stage_index] = value
                    logits = _logits_for_alphas(path, alphas, inputs.shape[-2:])
                    confusions[_key(stage, value)] += confusion_from_logits(
                        logits, targets, num_classes, ignore_index
                    )

    metrics = {name: l4_global_metrics(confusion) for name, confusion in confusions.items()}
    result = {
        "status": "completed",
        "purpose": "inference-only top-down alpha sweep on OSD Val",
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
        "definition": "At one merge only, replace P+U by P+alpha*U; all other merges remain alpha=1.",
        "alpha_values": values,
        "normal_path_max_abs_diff_vs_model": normal_max_diff,
        "metrics_percent": metrics,
    }
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"output={output_path}")
    print("test_accessed=False training_started=False")
    for name, value in metrics.items():
        print(
            f"{name}: mIoU3={value['mIoU3_report_only_global']:.6f}% "
            f"mF1_3={value['mF1_3_report_only_global']:.6f}% "
            f"PixelAcc={value['PixelAccuracy_global']:.6f}%"
        )


if __name__ == "__main__":
    main()
