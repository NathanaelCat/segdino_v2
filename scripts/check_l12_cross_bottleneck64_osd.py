#!/usr/bin/env python3
"""Preflight the rank-64 nonlinear bottleneck SAD replacement.

This check never builds a dataset loader and never accesses the test split. It
compares a fresh bottleneck model with a fresh Cross-MSEF+R reference, checks
identity initialization, gradients, shapes, parameter counts, and a short
forward/backward benchmark.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import ModelConfig  # noqa: E402
from dpt import (  # noqa: E402
    BottleneckResidualBlock,
    CrossMSEF,
    OfficialMSEFBlock,
    ResidualDepthwiseBlock,
)
from runtime import build_model, summarize_parameters  # noqa: E402


CONFIG_PATH = ROOT_DIR / "configs/dino_l12_spm_cross_msef_bottleneck64_001.json"
OUTPUT_PATH = ROOT_DIR / "work_dirs/dino_l12_cross_bottleneck64_preflight.json"
SEED = 20260901


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def _model_config(raw: dict) -> ModelConfig:
    model = raw["model"]
    return ModelConfig(
        dino_size=model["dino_size"],
        dino_repo=str(_resolve(model["dino_repo"])),
        dino_ckpt=str(_resolve(model["dino_ckpt"])),
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


def _build(raw: dict, variant: str, device: torch.device):
    model_raw = copy.deepcopy(raw)
    model_raw["model"]["decoder_variant"] = variant
    torch.manual_seed(SEED)
    model, backbone = build_model(_model_config(model_raw), str(device))
    model.lock_backbone()
    return model, backbone


def _count(parameters) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _grad_norm(parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm().item())


def _finite_grad(parameter) -> bool:
    return parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())


def _capture_sad_shapes(model):
    names = (
        "sad_intra_1", "sad_intra_2", "sad_intra_3", "sad_intra_4",
        "sad_inter_4", "sad_inter_3", "sad_inter_2", "sad_inter_1",
    )
    captured = {}
    hooks = []

    def make_hook(name):
        def hook(_module, inputs):
            captured[name] = list(inputs[0].shape)

        return hook

    for name in names:
        hooks.append(getattr(model.decoder, name).register_forward_pre_hook(make_hook(name)))
    return captured, hooks


def _benchmark(model, device: torch.device, batch_size: int, steps: int):
    model.train()
    x = torch.randn(batch_size, 3, 512, 512, device=device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(x).float().mean().backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        model(x).float().mean().backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak = (
        float(torch.cuda.max_memory_allocated(device) / 2**20)
        if device.type == "cuda" else None
    )
    return {
        "steps": int(steps),
        "batch_size": int(batch_size),
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(steps, 1),
        "peak_memory_mib": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--benchmark-steps", type=int, default=100)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT_DIR / args.config
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw["model"].get("decoder_variant") != "l12_a_cross_bottleneck64":
        raise RuntimeError("Unexpected bottleneck preflight decoder_variant")
    if raw.get("selection_split", "val") != "val":
        raise RuntimeError("Preflight requires validation-only selection")
    if raw.get("init_checkpoint") is not None or raw["model"].get("init_checkpoint") is not None:
        raise RuntimeError("Preflight must not load an experiment checkpoint")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    candidate, candidate_backbone = _build(raw, "l12_a_cross_bottleneck64", device)
    reference, reference_backbone = _build(raw, "l12_a_cross_r", device)
    candidate.eval()
    reference.eval()

    bottleneck_blocks = [
        module for module in candidate.decoder.modules()
        if isinstance(module, BottleneckResidualBlock)
    ]
    r_blocks = [
        module for module in reference.decoder.modules()
        if isinstance(module, ResidualDepthwiseBlock)
    ]
    official_msef = [
        module for module in candidate.decoder.modules()
        if isinstance(module, OfficialMSEFBlock)
    ]
    cross_blocks = [
        module for module in candidate.decoder.modules()
        if isinstance(module, CrossMSEF)
    ]
    if len(bottleneck_blocks) != 8:
        raise RuntimeError(f"Expected 8 bottleneck blocks, got {len(bottleneck_blocks)}")
    if len(r_blocks) != 8:
        raise RuntimeError(f"Reference must contain 8 R blocks, got {len(r_blocks)}")
    if official_msef:
        raise RuntimeError(f"Candidate unexpectedly retains {len(official_msef)} MSEF blocks")
    if len(cross_blocks) != 3:
        raise RuntimeError(f"Expected 3 Cross-MSEF blocks, got {len(cross_blocks)}")

    backbone_frozen = all(not parameter.requires_grad for parameter in candidate_backbone.parameters())
    patch_embed_frozen = all(
        not parameter.requires_grad for parameter in candidate_backbone.patch_embed.parameters()
    )
    if not backbone_frozen or not patch_embed_frozen:
        raise RuntimeError("Backbone or PatchEmbed is not frozen")

    x = torch.linspace(-1.0, 1.0, steps=3 * 512 * 512, device=device).reshape(1, 3, 512, 512)
    with torch.no_grad():
        candidate_logits = candidate(x)
        reference_logits = reference(x)
    logits_diff = float((candidate_logits - reference_logits).abs().max().item())
    logits_shape = list(candidate_logits.shape)
    if logits_shape != [1, int(raw["num_classes"]), 512, 512]:
        raise RuntimeError(f"Unexpected logits shape: {logits_shape}")

    shapes, hooks = _capture_sad_shapes(candidate)
    with torch.no_grad():
        _ = candidate(x)
    for hook in hooks:
        hook.remove()
    expected_shapes = {
        "sad_intra_1": [1, 256, 256, 256],
        "sad_intra_2": [1, 256, 128, 128],
        "sad_intra_3": [1, 256, 64, 64],
        "sad_intra_4": [1, 256, 32, 32],
        "sad_inter_4": [1, 256, 32, 32],
        "sad_inter_3": [1, 256, 64, 64],
        "sad_inter_2": [1, 256, 128, 128],
        "sad_inter_1": [1, 256, 256, 256],
    }
    if shapes != expected_shapes:
        raise RuntimeError(f"Unexpected SAD shapes: {shapes}")

    candidate.train()
    for parameter in candidate.parameters():
        parameter.grad = None
    candidate(x).float().mean().backward()
    gamma_grads = [_grad_norm(block.gamma) for block in bottleneck_blocks]
    first_branch_grads = {
        "depthwise": [_grad_norm(block.depthwise.weight) for block in bottleneck_blocks],
        "down": [_grad_norm(block.down.weight) for block in bottleneck_blocks],
        "up": [_grad_norm(block.up.weight) for block in bottleneck_blocks],
    }
    if not all(_finite_grad(parameter) for parameter in candidate.parameters() if parameter.requires_grad):
        raise RuntimeError("Non-finite gradient in first backward")
    if not all(value > 0.0 for value in gamma_grads):
        raise RuntimeError(f"At least one gamma gradient is zero: {gamma_grads}")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in candidate.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    candidate(x).float().mean().backward()
    second_branch_grads = {
        "depthwise": [_grad_norm(block.depthwise.weight) for block in bottleneck_blocks],
        "down": [_grad_norm(block.down.weight) for block in bottleneck_blocks],
        "up": [_grad_norm(block.up.weight) for block in bottleneck_blocks],
    }
    if not all(_finite_grad(parameter) for parameter in candidate.parameters() if parameter.requires_grad):
        raise RuntimeError("Non-finite gradient in second backward")
    if not all(value > 0.0 for values in second_branch_grads.values() for value in values):
        raise RuntimeError(f"Bottleneck branch did not open after gamma step: {second_branch_grads}")

    candidate_total, candidate_backbone_params, candidate_decoder_params = summarize_parameters(
        candidate, candidate_backbone
    )
    ref_total, ref_backbone_params, ref_decoder_params = summarize_parameters(
        reference, reference_backbone
    )
    candidate_trainable = _count(parameter for parameter in candidate.parameters() if parameter.requires_grad)
    ref_trainable = _count(parameter for parameter in reference.parameters() if parameter.requires_grad)
    block_params = _count(bottleneck_blocks[0].parameters())
    r_block_params = _count(r_blocks[0].parameters())

    del reference, reference_backbone, candidate_logits, reference_logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    benchmark = _benchmark(candidate, device, args.batch_size, args.benchmark_steps)

    result = {
        "preflight": "PASS",
        "test_accessed": False,
        "config": str(config_path),
        "device": str(device),
        "current_variant": "l12_a_cross_bottleneck64",
        "reference_variant": "l12_a_cross_r",
        "input_shape": list(x.shape),
        "logits_shape": logits_shape,
        "initial_logits_max_abs_diff_vs_R": logits_diff,
        "sad_shapes": shapes,
        "bottleneck_block_count": len(bottleneck_blocks),
        "reference_r_block_count": len(r_blocks),
        "cross_msef_count": len(cross_blocks),
        "backbone_frozen": backbone_frozen,
        "patch_embed_frozen": patch_embed_frozen,
        "backbone_source": str(_resolve(raw["model"]["dino_ckpt"])),
        "params": {
            "candidate_total": candidate_total,
            "candidate_backbone": candidate_backbone_params,
            "candidate_decoder_and_stem": candidate_decoder_params,
            "candidate_trainable": candidate_trainable,
            "reference_r_total": ref_total,
            "reference_r_backbone": ref_backbone_params,
            "reference_r_decoder_and_stem": ref_decoder_params,
            "reference_r_trainable": ref_trainable,
            "delta_trainable_candidate_minus_r": candidate_trainable - ref_trainable,
            "candidate_block_each": block_params,
            "r_block_each": r_block_params,
            "candidate_blocks_total": block_params * 8,
            "r_blocks_total": r_block_params * 8,
        },
        "gamma_grad_norms_first_backward": gamma_grads,
        "candidate_branch_grad_norms_first_backward": first_branch_grads,
        "candidate_branch_grad_norms_after_gamma_step": second_branch_grads,
        "benchmark": benchmark,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
