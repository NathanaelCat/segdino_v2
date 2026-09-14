"""Probe whether Lite-SPM contains spatial information missing from L12.

The experiment is deliberately a frozen-feature information diagnostic, not a
segmentation model experiment.  It trains only small linear classifiers on
train features and evaluates them on Val.  Test is never read.

For every 16x16 DINO patch, a patch is called *mixed* when its valid ground
truth contains at least two classes.  DINO features are broadcast at patch
resolution, while the 1/2-resolution Lite-SPM map is sampled at pixel
resolution.  A within-patch SPM permutation preserves local feature content
but removes its position relationship to the pixels.
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
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEG_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_path in (SEG_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from diagnose_cdr_val import _boundary_mask, _load_model  # noqa: E402
from osd_metrics import CURRENT_CLASSES, l4_global_metrics  # noqa: E402
from train_osd import _load_config, _make_loader, _normalize_input_size, _resolve_path  # noqa: E402


MODES = ("dino", "spm", "dino_spm", "dino_spm_shuffled")


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


def _mixed_pixel_mask(target: torch.Tensor, ignore_index: int, patch_size: int = 16) -> torch.Tensor:
    if target.ndim != 3:
        raise ValueError(f"Expected target [B,H,W], got {tuple(target.shape)}")
    batch, height, width = target.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Input dimensions must be divisible by DINO patch size")
    grid_h, grid_w = height // patch_size, width // patch_size
    target_blocks = target.contiguous().view(batch, grid_h, patch_size, grid_w, patch_size)
    valid_blocks = (target_blocks != ignore_index)
    present = []
    for class_index in range(len(CURRENT_CLASSES)):
        present.append(
            ((target_blocks == class_index) & valid_blocks).any(dim=(2, 4))
        )
    mixed_patches = torch.stack(present, dim=-1).sum(dim=-1) >= 2
    return (
        mixed_patches[:, :, None, :, None]
        .expand(batch, grid_h, patch_size, grid_w, patch_size)
        .reshape(batch, height, width)
    )


def _within_patch_shuffle(feature: torch.Tensor, patch_size: int = 16, seed: int = 0) -> torch.Tensor:
    """Permute 1/2-resolution SPM cells independently inside each DINO patch."""
    batch, channels, height, width = feature.shape
    if height % (patch_size // 2) or width % (patch_size // 2):
        raise ValueError("SPM map is not aligned to the DINO patch grid")
    cells = patch_size // 2
    grid_h, grid_w = height // cells, width // cells
    blocks = (
        feature.contiguous()
        .view(batch, channels, grid_h, cells, grid_w, cells)
        .permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .view(batch * grid_h * grid_w, channels, cells * cells)
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutations = torch.argsort(
        torch.rand(blocks.shape[0], cells * cells, generator=generator), dim=-1
    ).to(feature.device)
    permutations = permutations.unsqueeze(1).expand(-1, channels, -1)
    shuffled = torch.gather(blocks, dim=2, index=permutations)
    return (
        shuffled.view(batch, grid_h, grid_w, channels, cells, cells)
        .permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .view(batch, channels, height, width)
    )


def _sample_train_indices(mask: torch.Tensor, target: torch.Tensor, cap: int, seed: int) -> torch.Tensor:
    """Class-balanced sample from one image's mixed pixels."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    flat_mask = mask.reshape(-1)
    flat_target = target.reshape(-1)
    available = torch.nonzero(flat_mask, as_tuple=False).flatten()
    if available.numel() <= cap:
        return available
    selected = []
    per_class = max(1, int(cap) // len(CURRENT_CLASSES))
    for class_index in range(len(CURRENT_CLASSES)):
        candidates = available[flat_target[available] == class_index]
        if candidates.numel() > per_class:
            order = torch.randperm(candidates.numel(), generator=generator)[:per_class]
            candidates = candidates[order]
        selected.append(candidates)
    selected = torch.cat(selected) if selected else available[:0]
    if selected.numel() < cap:
        selected_set = torch.zeros(flat_mask.numel(), dtype=torch.bool)
        selected_set[selected] = True
        remaining = available[~selected_set[available]]
        if remaining.numel() > cap - selected.numel():
            order = torch.randperm(remaining.numel(), generator=generator)[: cap - selected.numel()]
            remaining = remaining[order]
        selected = torch.cat((selected, remaining))
    return selected


def _feature_maps(model, inputs, shuffle_seed):
    patch_h = inputs.shape[-2] // model.patch_size
    patch_w = inputs.shape[-1] // model.patch_size
    final_layer_idx = model.intermediate_layer_idx[model.encoder_size][-1]
    semantic_tokens = model.backbone.get_intermediate_layers(
        inputs, n=[final_layer_idx]
    )[-1]
    dino_map = model.decoder._tokens_to_feature_map(semantic_tokens, patch_h, patch_w)
    spm_native = model.spm_stem(inputs)[0]
    spm_shuffled_native = _within_patch_shuffle(
        spm_native, patch_size=model.patch_size, seed=shuffle_seed
    )
    spm = F.interpolate(spm_native, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
    spm_shuffled = F.interpolate(
        spm_shuffled_native, size=inputs.shape[-2:], mode="bilinear", align_corners=False
    )
    return dino_map, spm, spm_shuffled


def _gather_features(dino_map, spm, spm_shuffled, indices, width):
    y = indices // width
    x = indices % width
    dino = dino_map[:, y // 16, x // 16].transpose(0, 1).contiguous()
    spatial = spm[:, y, x].transpose(0, 1).contiguous()
    spatial_shuffled = spm_shuffled[:, y, x].transpose(0, 1).contiguous()
    return {
        "dino": dino,
        "spm": spatial,
        "dino_spm": torch.cat((dino, spatial), dim=1),
        "dino_spm_shuffled": torch.cat((dino, spatial_shuffled), dim=1),
    }


def _append_train_features(store, features, labels):
    for mode in MODES:
        store[mode].append(features[mode].detach().float().cpu())
    store["labels"].append(labels.detach().long().cpu())


def _standardize(features):
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (features - mean) / std, mean, std


def _train_linear_probe(features, labels, device, seed, epochs, batch_size, lr):
    torch.manual_seed(int(seed))
    normalized, mean, std = _standardize(features)
    normalized = normalized.to(device)
    labels = labels.to(device)
    classifier = nn.Linear(normalized.shape[1], len(CURRENT_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=float(lr), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    n_samples = normalized.shape[0]
    last_loss = None
    for _epoch in range(int(epochs)):
        order = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, int(batch_size)):
            indices = order[start : start + int(batch_size)]
            logits = classifier(normalized[indices])
            loss = criterion(logits, labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
    return classifier, mean.to(device), std.to(device), last_loss


def _update_confusion(confusion, logits, labels, region_mask):
    predictions = logits.argmax(dim=1)
    selected_labels = labels[region_mask].detach().long().cpu().numpy()
    selected_predictions = predictions[region_mask].detach().long().cpu().numpy()
    if selected_labels.size:
        packed = selected_labels * len(CURRENT_CLASSES) + selected_predictions
        confusion += np.bincount(
            packed, minlength=len(CURRENT_CLASSES) * len(CURRENT_CLASSES)
        ).reshape(len(CURRENT_CLASSES), len(CURRENT_CLASSES))


def _evaluate_probes(
    model,
    loader,
    probes,
    device,
    ignore_index,
    max_eval_per_image,
    seed,
):
    confusions = {
        mode: {region: np.zeros((len(CURRENT_CLASSES), len(CURRENT_CLASSES)), dtype=np.int64)
               for region in ("mixed", "mixed_boundary", "mixed_interior")}
        for mode in MODES
    }
    counts = {region: 0 for region in ("mixed", "mixed_boundary", "mixed_interior")}
    input_width = None
    with torch.inference_mode():
        for batch_index, (inputs, targets, _sample_ids) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            input_width = inputs.shape[-1]
            dino_map, spm, spm_shuffled = _feature_maps(
                model, inputs, shuffle_seed=seed + batch_index
            )
            mixed = _mixed_pixel_mask(targets, ignore_index, patch_size=model.patch_size)
            boundary = _boundary_mask(targets, ignore_index)
            for image_index in range(inputs.shape[0]):
                selected = _sample_train_indices(
                    mixed[image_index], targets[image_index], max_eval_per_image,
                    seed=seed + 10000 + batch_index * 101 + image_index,
                )
                if selected.numel() == 0:
                    continue
                features = _gather_features(
                    dino_map[image_index],
                    spm[image_index],
                    spm_shuffled[image_index],
                    selected,
                    input_width,
                )
                labels = targets[image_index].reshape(-1)[selected]
                boundary_selected = boundary[image_index].reshape(-1)[selected]
                regions = {
                    "mixed": torch.ones_like(labels, dtype=torch.bool),
                    "mixed_boundary": boundary_selected,
                    "mixed_interior": ~boundary_selected,
                }
                for region, region_mask in regions.items():
                    counts[region] += int(region_mask.sum().item())
                for mode in MODES:
                    normalized = (features[mode].to(device) - probes[mode]["mean"]) / probes[mode]["std"]
                    logits = probes[mode]["classifier"](normalized)
                    for region, region_mask in regions.items():
                        _update_confusion(
                            confusions[mode][region], logits, labels.to(device), region_mask.to(device)
                        )
    metrics = {
        mode: {
            region: l4_global_metrics(confusion)
            for region, confusion in region_confusions.items()
        }
        for mode, region_confusions in confusions.items()
    }
    return metrics, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=SEG_ROOT / "configs/dino_l12_spm_cross_r_osd_512.json",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=SEG_ROOT / "runs/dino_l12_spm_cross_r_osd_512_seed20260901/best.pth",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-train-per-image", type=int, default=256)
    parser.add_argument("--max-val-per-image", type=int, default=2048)
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--probe-batch-size", type=int, default=4096)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "work_dirs/dino_mixed_patch_information_probe.json",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    checkpoint_path = _resolve_path(args.checkpoint)
    config = _load_config(config_path)
    if config.get("model", {}).get("decoder_variant") != "l12_a_cross_r":
        raise RuntimeError("This probe is fixed to the Cross-MSEF + original R checkpoint")
    if config.get("selection_split", "val") != "val":
        raise RuntimeError("This probe must keep Val as the selection split")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    model = _load_model(config, checkpoint_path, device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("DINO backbone is not frozen")

    input_size = _normalize_input_size(config["input_size"])
    # The probe uses the canonical, unaugmented representation of each image.
    train_loader = _make_loader(
        _resolve_path(config["data_root"]), config["train_split"], input_size, False,
        int(args.batch_size), int(args.workers), config, device,
    )
    val_loader = _make_loader(
        _resolve_path(config["data_root"]), config["val_split"], input_size, False,
        int(args.batch_size), int(args.workers), config, device,
    )
    if len(train_loader.dataset) != 811 or len(val_loader.dataset) != 203:
        raise RuntimeError(
            f"Expected train/val=811/203, found {len(train_loader.dataset)}/{len(val_loader.dataset)}"
        )

    store = {mode: [] for mode in MODES}
    store["labels"] = []
    train_images = 0
    train_mixed_pixels = 0
    map_shapes = {}
    with torch.inference_mode():
        for batch_index, (inputs, targets, _sample_ids) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            dino_map, spm, spm_shuffled = _feature_maps(
                model, inputs, shuffle_seed=args.seed + batch_index
            )
            map_shapes = {
                "dino_map": list(dino_map.shape),
                "spm_upsampled": list(spm.shape),
                "spm_shuffled_upsampled": list(spm_shuffled.shape),
            }
            mixed = _mixed_pixel_mask(targets, int(config["ignore_index"]), patch_size=model.patch_size)
            for image_index in range(inputs.shape[0]):
                selected = _sample_train_indices(
                    mixed[image_index], targets[image_index], args.max_train_per_image,
                    seed=args.seed + 1000 + batch_index * 101 + image_index,
                )
                train_mixed_pixels += int(mixed[image_index].sum().item())
                if selected.numel() == 0:
                    continue
                features = _gather_features(
                    dino_map[image_index], spm[image_index], spm_shuffled[image_index],
                    selected, inputs.shape[-1],
                )
                labels = targets[image_index].reshape(-1)[selected]
                _append_train_features(store, features, labels)
                train_images += 1

    train_features = {
        mode: torch.cat(store[mode], dim=0) for mode in MODES
    }
    train_labels = torch.cat(store["labels"], dim=0)
    probes = {}
    probe_losses = {}
    for mode_index, mode in enumerate(MODES):
        classifier, mean, std, last_loss = _train_linear_probe(
            train_features[mode], train_labels, device,
            seed=args.seed + 2000 + mode_index,
            epochs=args.probe_epochs,
            batch_size=args.probe_batch_size,
            lr=args.probe_lr,
        )
        probes[mode] = {"classifier": classifier, "mean": mean, "std": std}
        probe_losses[mode] = last_loss
        del train_features[mode]
        torch.cuda.empty_cache() if device.type == "cuda" else None

    metrics, val_counts = _evaluate_probes(
        model, val_loader, probes, device, int(config["ignore_index"]),
        args.max_val_per_image, args.seed + 5000,
    )
    result = {
        "status": "completed",
        "purpose": "frozen-feature mixed 16x16 patch information probe",
        "training_started": False,
        "main_model_weights_modified": False,
        "test_accessed": False,
        "git_head": _git_head(),
        "device": str(device),
        "input_size": list(input_size),
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "probe_train_images_with_mixed_pixels": train_images,
        "train_mixed_pixel_count": train_mixed_pixels,
        "probe_train_sample_count": int(train_labels.numel()),
        "val_probe_counts": val_counts,
        "probe_definition": {
            "mixed_patch": "valid GT within one 16x16 DINO patch contains >=2 of 4 classes",
            "dino": "raw final-layer L12 token, broadcast to pixels by nearest patch replication",
            "spm": "trained Lite-SPM D2 (1/2 resolution), bilinear sampled to 512",
            "dino_spm": "concatenated DINO and SPM features",
            "dino_spm_shuffled": "SPM D2 cells randomly permuted independently within each 16x16 DINO patch before upsampling",
            "probe": "independent Linear(feature_dim,4), train features from train split only",
            "probe_preprocessing": "unaugmented canonical 512 representation; per-feature train standardization",
        },
        "feature_shapes_first_batch": map_shapes,
        "probe_training": {
            "epochs": args.probe_epochs,
            "batch_size": args.probe_batch_size,
            "lr": args.probe_lr,
            "weight_decay": 1e-4,
            "last_train_loss": probe_losses,
            "feature_dims": {mode: int(probes[mode]["classifier"].in_features) for mode in MODES},
        },
        "parent": {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "decoder_variant": config["model"]["decoder_variant"],
        },
        "metrics_percent": metrics,
    }
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"output={output_path}")
    print("training_started=False main_model_weights_modified=False test_accessed=False")
    for mode in MODES:
        for region, values in metrics[mode].items():
            print(
                f"{mode}/{region}: mIoU3={values['mIoU3_report_only_global']:.6f}% "
                f"mF1_3={values['mF1_3_report_only_global']:.6f}% "
                f"PixelAcc={values['PixelAccuracy_global']:.6f}%"
            )


if __name__ == "__main__":
    main()
