#!/usr/bin/env python3
"""Preflight the official DINOv3 segmentation adapter on OSD.

This is an interface/gradient audit only.  It uses one validation image,
does not access the OSD test split, and never calls optimizer.step().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import ModelConfig  # noqa: E402
from osd_dataset import OSDDataset, build_osd_transform  # noqa: E402
from runtime import build_model, summarize_parameters  # noqa: E402
from train_osd import _normalize_input_size, _resolve_path  # noqa: E402


def _shape(value: torch.Tensor) -> list[int]:
    return list(value.shape)


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_config(raw: dict) -> ModelConfig:
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


def _mib(value: int) -> float:
    return round(value / 1024**2, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "configs/dino_official_adapter_osd_512_seed20260901.json",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw["model"].get("decoder_variant") != "dinov3_adapter":
        raise ValueError("This preflight requires decoder_variant=dinov3_adapter")
    if int(raw["num_classes"]) != 4:
        raise ValueError("This preflight requires four OSD classes")
    if not bool(raw["model"].get("freeze_backbone", True)):
        raise ValueError("Official Adapter preflight requires a frozen backbone")

    input_size = _normalize_input_size(raw["input_size"])
    model_cfg = _model_config(raw)
    device = torch.device(
        args.device
        or raw.get("runtime", {}).get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    data_root = _resolve_path(raw["data_root"])
    val_transform = build_osd_transform(
        input_size,
        train=False,
        augment=None,
        resize_mode=raw.get("preprocessing", {}).get("resize_mode", "stretch"),
        ignore_index=int(raw["ignore_index"]),
    )
    val_dataset = OSDDataset(
        data_root,
        split=raw.get("val_split", "val"),
        transform=val_transform,
        num_classes=int(raw["num_classes"]),
        ignore_index=int(raw["ignore_index"]),
    )
    image, target, sample_id = val_dataset[0]
    x = image.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)

    model, backbone = build_model(model_cfg, str(device))
    model.lock_backbone()
    decoder = model.decoder
    adapter = decoder.adapter
    if backbone is not adapter.backbone:
        raise RuntimeError("DPT backbone and official Adapter backbone are not the same object")

    backbone_trainable = [
        name for name, parameter in backbone.named_parameters() if parameter.requires_grad
    ]
    if backbone_trainable:
        raise RuntimeError(f"Backbone is not fully frozen: {backbone_trainable[:10]}")

    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        spm_outputs = adapter.spm(x)
        adapter_outputs = adapter(x)
        ordered = [adapter_outputs[str(index)] for index in range(1, 5)]
        head_logits = decoder.head(ordered)
        logits = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    eval_seconds = time.perf_counter() - start

    # The official SPM keeps c1 at 1/4 as BCHW, then flattens c2/c3/c4 to
    # [B, N, C] before the deformable interactions.  The Adapter restores
    # all four outputs to BCHW at its public boundary.
    expected_spm = [
        (1, 384, 128, 128),
        (1, 4096, 384),
        (1, 1024, 384),
        (1, 256, 384),
    ]
    actual_spm = [tuple(value.shape) for value in spm_outputs]
    if actual_spm != expected_spm:
        raise RuntimeError(f"Official SPM shape mismatch: got {actual_spm}, expected {expected_spm}")
    expected_adapter = [
        (1, 384, 128, 128),
        (1, 384, 64, 64),
        (1, 384, 32, 32),
        (1, 384, 16, 16),
    ]
    actual_adapter = [tuple(value.shape) for value in ordered]
    if actual_adapter != expected_adapter:
        raise RuntimeError(
            f"Official Adapter output shape mismatch: got {actual_adapter}, expected {expected_adapter}"
        )
    if tuple(head_logits.shape) != (1, 4, 128, 128):
        raise RuntimeError(f"Official head shape mismatch: {tuple(head_logits.shape)}")
    if tuple(logits.shape) != (1, 4, input_size[0], input_size[1]):
        raise RuntimeError(f"Final OSD logits shape mismatch: {tuple(logits.shape)}")
    spm_shapes = [list(value.shape) for value in spm_outputs]
    adapter_shapes = [list(value.shape) for value in ordered]

    del spm_outputs, adapter_outputs, ordered, head_logits, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    model.zero_grad(set_to_none=True)
    start = time.perf_counter()
    train_logits = model(x)
    valid = target != int(raw["ignore_index"])
    if not bool(valid.any()):
        raise RuntimeError(f"Validation sample {sample_id} has no valid target pixels")
    loss = F.cross_entropy(train_logits, target, ignore_index=int(raw["ignore_index"]))
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - start

    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    missing_grad_names = [name for name, parameter in trainable_parameters if parameter.grad is None]
    nonfinite_grad_names = [
        name
        for name, parameter in trainable_parameters
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if missing_grad_names:
        raise RuntimeError(f"Trainable parameters without gradients: {missing_grad_names[:10]}")
    if nonfinite_grad_names or not bool(torch.isfinite(loss).item()):
        raise RuntimeError(f"Non-finite preflight state: loss={loss.item()}, grads={nonfinite_grad_names[:10]}")

    optimizer_cfg = raw["optimizer"]
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable_parameters],
        lr=float(optimizer_cfg["lr"]),
        betas=tuple(float(value) for value in optimizer_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    del optimizer

    total_params, backbone_params, decoder_params = summarize_parameters(model, backbone)
    trainable_params = sum(parameter.numel() for _, parameter in trainable_parameters)
    memory = {
        "allocated_mib": None,
        "reserved_mib": None,
        "peak_allocated_mib": None,
    }
    if device.type == "cuda":
        memory = {
            "allocated_mib": _mib(torch.cuda.memory_allocated(device)),
            "reserved_mib": _mib(torch.cuda.memory_reserved(device)),
            "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated(device)),
        }

    official_repo = Path(model_cfg.dino_repo).resolve()
    official_ckpt = Path(model_cfg.dino_ckpt).resolve()
    result = {
        "current_branch": _git(ROOT_DIR, "branch", "--show-current"),
        "current_commit": _git(ROOT_DIR, "rev-parse", "HEAD"),
        "model": raw["name"],
        "decoder_variant": model_cfg.decoder_variant,
        "device": str(device),
        "input_shape": _shape(x),
        "validation_sample": sample_id,
        "official_dinov3_repo": str(official_repo),
        "official_dinov3_repo_commit": _git(official_repo, "rev-parse", "HEAD"),
        "dino_checkpoint": str(official_ckpt),
        "dino_checkpoint_sha256": _sha256(official_ckpt),
        "official_interaction_indexes_zero_based": [2, 5, 8, 11],
        "official_interaction_names": ["L3", "L6", "L9", "L12"],
        "backbone_frozen": all(not parameter.requires_grad for parameter in backbone.parameters()),
        "backbone_trainable_names": backbone_trainable,
        "backbone_shared_with_decoder_adapter": backbone is adapter.backbone,
        "official_spatial_prior_shapes": spm_shapes,
        "official_adapter_output_shapes": adapter_shapes,
        "feature_strides": [4, 8, 16, 32],
        "official_linear_head_logits_shape": [1, 4, 128, 128],
        "final_logits_shape": list(train_logits.shape),
        "backbone_parameters": backbone_params,
        "decoder_unique_parameters": decoder_params,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_parameter_tensor_count": len(trainable_parameters),
        "optimizer": {
            "name": optimizer_cfg.get("name", "AdamW"),
            "lr": float(optimizer_cfg["lr"]),
            "betas": optimizer_cfg.get("betas", [0.9, 0.999]),
            "weight_decay": float(optimizer_cfg["weight_decay"]),
            "parameter_tensor_count": len(trainable_parameters),
            "parameter_names_first10": [name for name, _ in trainable_parameters[:10]],
            "parameter_names_last10": [name for name, _ in trainable_parameters[-10:]],
        },
        "loss": float(loss.detach().item()),
        "finite_forward_backward": True,
        "forward_seconds_eval": round(eval_seconds, 4),
        "forward_backward_seconds_probe": round(backward_seconds, 4),
        "memory": memory,
        "test_accessed": False,
        "optimizer_step_called": False,
        "formal_training_started": False,
        "sanity_check": "PASS",
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        output_path = args.output if args.output.is_absolute() else ROOT_DIR / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
