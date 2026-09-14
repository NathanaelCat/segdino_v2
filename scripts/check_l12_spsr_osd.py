#!/usr/bin/env python3
"""Preflight the L12 + Lite-SPM + SPSR prototype.

This script intentionally does not construct an OSD data loader, read the
test split, save a checkpoint, or start formal training.  It checks the
prototype's tensor contract, exact gamma=0 degradation, and the two-stage
gradient path required by the zero-initialized residual design.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import ModelConfig  # noqa: E402
from runtime import build_model, summarize_parameters  # noqa: E402


DEFAULT_CONFIG = ROOT_DIR / "configs/dino_l12_spm_spsr_msmlp_osd_512.json"
DEFAULT_OUTPUT = ROOT_DIR / "work_dirs/dino_l12_spsr_preflight.json"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def _load_config(path: Path) -> tuple[dict, ModelConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    model = raw["model"]
    config = ModelConfig(
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
    return raw, config


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT_DIR, text=True).strip()


def _shape(value: torch.Tensor) -> list[int]:
    return [int(dim) for dim in value.shape]


def _count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _trainable_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _finite(value: torch.Tensor | None) -> bool:
    return value is not None and bool(torch.isfinite(value).all().item())


def _norm(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().float().norm().item())


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _memory(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"allocated_mib": None, "reserved_mib": None, "peak_allocated_mib": None}
    return {
        "allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 2),
        "reserved_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2),
    }


def _flop_hooks(root: nn.Module):
    total = {"conv_linear_flops": 0, "conv2d_calls": 0, "linear_calls": 0}
    handles = []

    def hook(module: nn.Module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            return
        if isinstance(module, nn.Conv2d):
            output_elements = output.numel()
            kernel_elements = module.kernel_size[0] * module.kernel_size[1]
            per_output = module.in_channels // module.groups * kernel_elements
            total["conv_linear_flops"] += int(2 * output_elements * per_output)
            total["conv2d_calls"] += 1
        elif isinstance(module, nn.Linear):
            output_elements = output.numel()
            total["conv_linear_flops"] += int(2 * output_elements * module.in_features)
            total["linear_calls"] += 1

    for module in root.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    return total, handles


def _remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


def _run_decoder(model, backbone, x, patch_h, patch_w, diagnostics=False):
    with torch.no_grad():
        semantic_tokens = backbone.get_intermediate_layers(x, n=[11])[-1]
        spatial_features = model.spm_stem(x)
    if diagnostics:
        native_logits, trace = model.decoder.forward_with_diagnostics(
            semantic_tokens, spatial_features, patch_h, patch_w
        )
        return semantic_tokens, spatial_features, native_logits, trace
    native_logits = model.decoder(semantic_tokens, spatial_features, patch_h, patch_w)
    return semantic_tokens, spatial_features, native_logits, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = _resolve(args.config)
    raw, model_config = _load_config(config_path)
    if model_config.decoder_variant != "l12_spsr_ms_mlp":
        raise ValueError(
            "Expected decoder_variant='l12_spsr_ms_mlp', "
            f"got {model_config.decoder_variant!r}"
        )
    if model_config.layer_mapping is not None:
        raise ValueError("SPSR uses the single final L12 source and requires layer_mapping=null")
    if not model_config.freeze_backbone:
        raise ValueError("SPSR preflight requires freeze_backbone=true")
    if model_config.patch_size != 16 or args.img_size % model_config.patch_size:
        raise ValueError("SPSR preflight expects a 512-like size divisible by patch_size=16")

    seed = int(raw.get("runtime", {}).get("seed", 20260901))
    _seed_all(seed)
    device = torch.device(
        args.device
        or raw.get("runtime", {}).get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model, backbone = build_model(model_config, str(device))
    model.lock_backbone()
    if not all(not parameter.requires_grad for parameter in backbone.parameters()):
        raise RuntimeError("DINO backbone is not fully frozen")
    patch_embed = getattr(backbone, "patch_embed", None)
    if patch_embed is None or not all(
        not parameter.requires_grad for parameter in patch_embed.parameters()
    ):
        raise RuntimeError("DINO patch_embed is not fully frozen")

    batch_size = int(args.batch_size)
    patch_h = args.img_size // model_config.patch_size
    patch_w = args.img_size // model_config.patch_size
    x = torch.randn(batch_size, 3, args.img_size, args.img_size, device=device)
    target = torch.randint(
        low=0,
        high=model_config.num_classes,
        size=(batch_size, args.img_size, args.img_size),
        device=device,
        dtype=torch.long,
    )

    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    eval_start = time.perf_counter()
    semantic_tokens, spatial_features, native_logits, trace = _run_decoder(
        model, backbone, x, patch_h, patch_w, diagnostics=True
    )
    outer_logits = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    eval_seconds = time.perf_counter() - eval_start

    fallback_native = trace["fallback_logits"]
    expected_outer = F.interpolate(
        native_logits, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False
    )
    degradation_diff_native = float(
        (native_logits - fallback_native).detach().abs().max().item()
    )
    degradation_diff_outer = float(
        (outer_logits - expected_outer).detach().abs().max().item()
    )
    fallback_outer = F.interpolate(
        fallback_native, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False
    )
    candidate_fallback_outer_diff = float(
        (outer_logits - fallback_outer).detach().abs().max().item()
    )

    decoder = model.decoder
    resampler = decoder.resampler
    pyramid = trace["pyramid"]
    outputs = trace["outputs"]
    resampler_traces = trace["resamplers"]
    gamma_init = [float(gamma.detach().item()) for gamma in decoder.gammas]
    max_offset_init = max(
        float(item["offset_tokens"].detach().abs().max().item())
        for item in resampler_traces
    )
    weight_sum_error = max(
        float((item["weights"].detach().sum(dim=1) - 1.0).abs().max().item())
        for item in resampler_traces
    )
    residual_norms = [
        float(item["resampled"].detach().sub(item["bilinear"]).float().norm().item())
        for item in resampler_traces
    ]

    # Count only the trainable components relevant to the prototype.
    total_params, backbone_params, non_backbone_params = summarize_parameters(model, backbone)
    component_params = {
        "lite_spm_stem_total": _count(model.spm_stem),
        "lite_spm_stem_trainable": _trainable_count(model.spm_stem),
        "semantic_projection": _count(decoder.semantic_projection),
        "spatial_projections": _count(decoder.spatial_projections),
        "shared_resampler": _count(resampler),
        "gamma_parameters": _count(decoder.gammas),
        "ms_mlp": _count(decoder.ms_mlp),
        "decoder_total": _count(decoder),
        "decoder_trainable": _trainable_count(decoder),
        "model_total": total_params,
        "backbone_total": backbone_params,
        "model_non_backbone_trainable": non_backbone_params,
    }

    flop_counts, flop_hooks = _flop_hooks(model.spm_stem)
    decoder_flops, decoder_hooks = _flop_hooks(decoder)
    # The hooks must observe a fresh forward.  The diagnostic forward above
    # happened before hooks were installed and therefore cannot be counted.
    model.eval()
    with torch.no_grad():
        _ = model.spm_stem(x)
        _ = decoder(semantic_tokens, spatial_features, patch_h, patch_w)
    flop_counts["conv_linear_flops"] += decoder_flops["conv_linear_flops"]
    flop_counts["conv2d_calls"] += decoder_flops["conv2d_calls"]
    flop_counts["linear_calls"] += decoder_flops["linear_calls"]
    _remove_hooks(flop_hooks)
    _remove_hooks(decoder_hooks)
    model.train()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    train_start = time.perf_counter()
    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    first_backward_seconds = time.perf_counter() - train_start

    gamma_grad_first = [_norm(gamma.grad) for gamma in decoder.gammas]
    offset_final_weight = resampler.offset_predictor[-1].weight
    offset_final_bias = resampler.offset_predictor[-1].bias
    first_backward = {
        "loss": float(loss.detach().item()),
        "gamma_grad_norm": gamma_grad_first,
        "gamma_grad_finite": [_finite(gamma.grad) for gamma in decoder.gammas],
        "offset_final_weight_grad_norm": _norm(offset_final_weight.grad),
        "offset_final_bias_grad_norm": _norm(offset_final_bias.grad),
        "query_grad_norm": _norm(resampler.query_projection.weight.grad),
        "key_grad_norm": _norm(resampler.key_projection.weight.grad),
    }

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(raw.get("optimizer", {}).get("lr", 1e-4)),
        weight_decay=float(raw.get("optimizer", {}).get("weight_decay", 1e-4)),
        betas=tuple(raw.get("optimizer", {}).get("betas", [0.9, 0.999])),
    )
    optimizer.step()
    gamma_after_step = [float(gamma.detach().item()) for gamma in decoder.gammas]

    model.zero_grad(set_to_none=True)
    second_start = time.perf_counter()
    logits_second = model(x)
    loss_second = F.cross_entropy(logits_second, target)
    loss_second.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    second_backward_seconds = time.perf_counter() - second_start
    second_backward = {
        "loss": float(loss_second.detach().item()),
        "gamma_grad_norm": [_norm(gamma.grad) for gamma in decoder.gammas],
        "offset_final_weight_grad_norm": _norm(offset_final_weight.grad),
        "offset_final_bias_grad_norm": _norm(offset_final_bias.grad),
        "offset_hidden_grad_norm": _norm(resampler.offset_predictor[0].weight.grad),
        "query_grad_norm": _norm(resampler.query_projection.weight.grad),
        "key_grad_norm": _norm(resampler.key_projection.weight.grad),
        "offset_branch_finite": _finite(offset_final_weight.grad)
        and _finite(offset_final_bias.grad),
        "compatibility_branch_finite": _finite(resampler.query_projection.weight.grad)
        and _finite(resampler.key_projection.weight.grad),
    }

    backbone_grad_names = [
        name for name, parameter in backbone.named_parameters() if parameter.grad is not None
    ]
    decoder_grad_count = sum(
        1
        for parameter in decoder.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    backbone_frozen = all(not parameter.requires_grad for parameter in backbone.parameters())
    patch_embed_frozen = all(
        not parameter.requires_grad for parameter in patch_embed.parameters()
    )
    first_gamma_ok = all(
        value is not None and value > 1e-12 and finite
        for value, finite in zip(
            first_backward["gamma_grad_norm"], first_backward["gamma_grad_finite"]
        )
    )
    second_gamma_opened = all(abs(value) > 1e-12 for value in gamma_after_step)
    second_branch_ok = bool(
        second_backward["offset_branch_finite"]
        and second_backward["compatibility_branch_finite"]
        # The first AdamW step only moves gamma by roughly 1e-5--1e-4 on
        # this random probe batch, so the newly opened branch gradients can
        # be very small.  The contract is finite and strictly non-zero;
        # their measured magnitudes remain in the report for review.
        and (second_backward["offset_final_weight_grad_norm"] or 0.0) > 0.0
        and (second_backward["query_grad_norm"] or 0.0) > 0.0
        and (second_backward["key_grad_norm"] or 0.0) > 0.0
    )
    finite_ok = bool(torch.isfinite(logits).all().item() and torch.isfinite(logits_second).all().item())
    sanity_checks = {
        "backbone_frozen": backbone_frozen,
        "patch_embed_frozen": patch_embed_frozen,
        "no_backbone_grad": not backbone_grad_names,
        "exact_gamma_zero_native": degradation_diff_native <= 1e-7,
        "exact_gamma_zero_outer": candidate_fallback_outer_diff <= 1e-7,
        "outer_shape_contract": tuple(outer_logits.shape)
        == (batch_size, model_config.num_classes, args.img_size, args.img_size),
        "initial_offsets_zero": max_offset_init <= 1e-7,
        "initial_weights_normalized": weight_sum_error <= 1e-6,
        "first_backward_gamma_open": first_gamma_ok,
        "gamma_left_zero_after_step": second_gamma_opened,
        "second_backward_branches_open": second_branch_ok,
        "forward_backward_finite": finite_ok,
        "decoder_received_gradients": decoder_grad_count > 0,
    }
    sanity_pass = all(sanity_checks.values())

    if device.type == "cuda":
        peak_memory = _memory(device)
    else:
        peak_memory = _memory(device)

    sampled_elements = sum(
        batch_size * resampler.num_samples * int(item["sampled_values"].shape[2])
        * int(item["sampled_values"].shape[3])
        * int(item["sampled_values"].shape[4])
        for item in resampler_traces
    )
    result = {
        "current_branch": _git("branch", "--show-current"),
        "current_commit": _git("rev-parse", "HEAD"),
        "config": str(config_path),
        "model": raw["name"],
        "decoder_variant": model_config.decoder_variant,
        "formal_training_started": False,
        "test_split_accessed": False,
        "seed": seed,
        "device": str(device),
        "input_shape": _shape(x),
        "l12_tokens_shape": _shape(semantic_tokens),
        "l12_source_map_shape": _shape(trace["semantic_map"]),
        "spatial_prior_shapes": [_shape(feature) for feature in spatial_features],
        "pyramid_shapes": [_shape(feature) for feature in pyramid],
        "spsr_output_shapes": [_shape(feature) for feature in outputs],
        "resampler_trace_shapes": {
            "offset_tokens": [_shape(item["offset_tokens"]) for item in resampler_traces],
            "grids": [_shape(item["grids"]) for item in resampler_traces],
            "sampled_values": [_shape(item["sampled_values"]) for item in resampler_traces],
            "compatibility": [_shape(item["compatibility"]) for item in resampler_traces],
            "weights": [_shape(item["weights"]) for item in resampler_traces],
            "resampled": [_shape(item["resampled"]) for item in resampler_traces],
        },
        "source_coordinate_contract": {
            "source_grid": [patch_h, patch_w],
            "num_samples": resampler.num_samples,
            "offset_limit_source_tokens": resampler.offset_limit,
            "align_corners": False,
            "padding_mode": "border",
            "anchor_offsets_source_tokens": resampler.anchor_offsets.detach().cpu().tolist(),
        },
        "logits_shape_native": _shape(native_logits),
        "logits_shape_outer": _shape(outer_logits),
        "gamma_initial": gamma_init,
        "gamma_after_one_optimizer_step": gamma_after_step,
        "max_learned_offset_at_init_source_tokens": max_offset_init,
        "max_weight_sum_error": weight_sum_error,
        "resampled_minus_bilinear_l2": residual_norms,
        "gamma_zero_max_abs_diff_native": degradation_diff_native,
        "gamma_zero_max_abs_diff_decoder_outer": degradation_diff_outer,
        "gamma_zero_max_abs_diff_outer_vs_fallback": candidate_fallback_outer_diff,
        "parameter_counts": component_params,
        "parameter_names": {
            "semantic_projection": [name for name, _ in decoder.semantic_projection.named_parameters()],
            "spatial_projections": [name for name, _ in decoder.spatial_projections.named_parameters()],
            "shared_resampler": [name for name, _ in resampler.named_parameters()],
            "gamma": [f"gammas.{index}" for index in range(len(decoder.gammas))],
        },
        "estimated_conv_linear_flops_per_forward": flop_counts,
        "grid_sample_calls_per_forward": 3,
        "sampled_value_elements_b2": sampled_elements,
        "timing_seconds": {
            "eval_decoder_plus_outer_forward": round(eval_seconds, 4),
            "first_forward_backward": round(first_backward_seconds, 4),
            "second_forward_backward": round(second_backward_seconds, 4),
        },
        "memory_after_preflight": peak_memory,
        "first_backward": first_backward,
        "second_backward": second_backward,
        "backbone_gradient_names": backbone_grad_names,
        "decoder_grad_tensors_after_second_backward": decoder_grad_count,
        "sanity_checks": sanity_checks,
        "sanity_check": "PASS" if sanity_pass else "FAIL",
    }
    args.output = _resolve(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    del optimizer, model, backbone, x, target, logits, logits_second
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not sanity_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
