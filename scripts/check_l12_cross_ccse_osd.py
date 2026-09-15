#!/usr/bin/env python3
"""Preflight CCSE against the fresh Cross-MSEF + original SAD/R parent.

This script performs architecture, initialization, gradient, shape, memory,
and 200-step latency checks only.  It never creates a data loader, reads a
checkpoint from an experiment run, accesses test data, or launches formal
training.
"""

from __future__ import annotations

import copy
import gc
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import ModelConfig  # noqa: E402
from dpt import CompactCrossScaleExchange, CrossMSEF, ResidualDepthwiseBlock  # noqa: E402
from runtime import build_model, summarize_parameters  # noqa: E402


CONFIG_PATH = ROOT_DIR / "configs/dino_l12_spm_cross_msef_ccse_001.json"
PARENT_CONFIG_PATH = ROOT_DIR / "configs/dino_l12_spm_cross_r_osd_512.json"
OUTPUT_PATH = ROOT_DIR / "work_dirs/dino_l12_cross_ccse_preflight.json"
SEED = 20260901
PARENT_VARIANT = "l12_a_cross_r"
CANDIDATE_VARIANT = "l12_a_cross_ccse"


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


def _norm(parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm().item())


def _finite(parameter) -> bool:
    return parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())


def _jsonable_diff(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left) | set(right))
        result = {}
        for key in keys:
            child_path = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                result[child_path] = {"left": left.get(key), "right": right.get(key)}
            else:
                result.update(_jsonable_diff(left[key], right[key], child_path))
        return result
    if isinstance(left, list) and isinstance(right, list):
        return {} if left == right else {path: {"left": left, "right": right}}
    return {} if left == right else {path: {"left": left, "right": right}}


def _config_diff(candidate_raw: dict, parent_raw: dict):
    candidate = copy.deepcopy(candidate_raw)
    parent = copy.deepcopy(parent_raw)
    # These fields are experiment metadata/runtime placement, not model or
    # optimization protocol variables.
    for raw in (candidate, parent):
        raw.pop("name", None)
        raw.pop("run_dir", None)
        raw.setdefault("runtime", {}).pop("device", None)
        raw.setdefault("model", {}).pop("decoder_variant", None)
    return _jsonable_diff(candidate, parent)


def _capture_ccse_input(model):
    captured = {}

    def hook(_module, inputs):
        captured["pyramid"] = tuple(inputs[0])

    handle = model.decoder.ccse.register_forward_pre_hook(hook)
    return captured, handle


def _benchmark(model, device: torch.device, x, target, steps: int, warmup: int):
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        F.cross_entropy(logits, target).backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        F.cross_entropy(logits, target).backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak = (
        float(torch.cuda.max_memory_allocated(device) / 2**20)
        if device.type == "cuda"
        else None
    )
    return {
        "steps": int(steps),
        "warmup_steps": int(warmup),
        "batch_size": int(x.shape[0]),
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(steps, 1),
        "estimated_5100_step_minutes": elapsed / max(steps, 1) * 5100 / 60,
        "peak_memory_mib": peak,
    }


def _gradient_preflight(raw: dict, device: torch.device):
    model, backbone = _build(raw, CANDIDATE_VARIANT, device)
    model.train()
    ccse = model.decoder.ccse
    x = torch.randn(1, 3, 512, 512, device=device)
    target = torch.randint(0, int(raw["num_classes"]), (1, 512, 512), device=device)
    ccse_optimizer = torch.optim.AdamW(ccse.parameters(), lr=1e-4, weight_decay=1e-4)

    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    first_gamma = [_norm(parameter) for parameter in ccse.gammas]
    first_internal = {
        "input_projections": [_norm(module.weight) for module in ccse.input_projections],
        "qkv": _norm(ccse.qkv.weight),
        "output_projection": _norm(ccse.output_projection.weight),
        "output_projections": [_norm(module.weight) for module in ccse.output_projections],
    }
    first_finite = all(
        _finite(parameter) for parameter in ccse.parameters() if parameter.requires_grad
    )
    if not first_finite:
        raise RuntimeError("CCSE first backward produced a non-finite gradient")
    if not all(value > 0.0 for value in first_gamma):
        raise RuntimeError(f"CCSE gamma gradients are not all nonzero: {first_gamma}")
    if any(value > 1e-10 for values in first_internal.values() for value in (values if isinstance(values, list) else [values])):
        raise RuntimeError(f"CCSE internal gradients were expected to be zero at gamma=0: {first_internal}")

    ccse_optimizer.step()
    gamma_after_step = [float(parameter.detach().item()) for parameter in ccse.gammas]
    ccse_optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), target).backward()
    second_internal = {
        "input_projections": [_norm(module.weight) for module in ccse.input_projections],
        "qkv": _norm(ccse.qkv.weight),
        "output_projection": _norm(ccse.output_projection.weight),
        "output_projections": [_norm(module.weight) for module in ccse.output_projections],
    }
    second_finite = all(
        _finite(parameter) for parameter in ccse.parameters() if parameter.requires_grad
    )
    if not second_finite:
        raise RuntimeError("CCSE second backward produced a non-finite gradient")
    if not all(
        value > 0.0
        for values in second_internal.values()
        for value in (values if isinstance(values, list) else [values])
    ):
        raise RuntimeError(f"CCSE internal branch did not open after gamma update: {second_internal}")

    result = {
        "first_backward_gamma_grad_norms": first_gamma,
        "first_backward_internal_grad_norms": first_internal,
        "gamma_after_one_ccse_step": gamma_after_step,
        "second_backward_internal_grad_norms": second_internal,
        "finite_gradients": second_finite,
    }
    del model, backbone, ccse, x, target, ccse_optimizer
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--parent-config", type=Path, default=PARENT_CONFIG_PATH)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--benchmark-steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=10)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT_DIR / args.config
    parent_config_path = (
        args.parent_config if args.parent_config.is_absolute() else ROOT_DIR / args.parent_config
    )
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    parent_raw = json.loads(parent_config_path.read_text(encoding="utf-8"))
    if raw["model"].get("decoder_variant") != CANDIDATE_VARIANT:
        raise RuntimeError("Unexpected CCSE decoder_variant in candidate config")
    if parent_raw["model"].get("decoder_variant") != PARENT_VARIANT:
        raise RuntimeError("Unexpected Cross-R decoder_variant in parent config")
    if raw.get("selection_split") != "val" or parent_raw.get("selection_split") != "val":
        raise RuntimeError("Both configs must select checkpoints on val")
    if raw.get("init_checkpoint") is not None or raw["model"].get("init_checkpoint") is not None:
        raise RuntimeError("CCSE preflight must not load an experiment checkpoint")
    unexpected_config_diff = _config_diff(raw, parent_raw)
    if unexpected_config_diff:
        raise RuntimeError(
            "Candidate config differs from the parent outside the allowed metadata and "
            f"decoder variant fields: {json.dumps(unexpected_config_diff, sort_keys=True)}"
        )

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    # Build fresh models for the initialization-equivalence and full forward
    # checks.  No experiment checkpoint is involved.
    candidate, candidate_backbone = _build(raw, CANDIDATE_VARIANT, device)
    parent, parent_backbone = _build(parent_raw, PARENT_VARIANT, device)
    candidate.eval()
    parent.eval()

    ccse_modules = [
        module for module in candidate.decoder.modules()
        if isinstance(module, CompactCrossScaleExchange)
    ]
    cross_modules = [
        module for module in candidate.decoder.modules()
        if isinstance(module, CrossMSEF)
    ]
    r_modules = [
        module for module in candidate.decoder.modules()
        if isinstance(module, ResidualDepthwiseBlock)
    ]
    if len(ccse_modules) != 1 or len(cross_modules) != 3 or len(r_modules) != 8:
        raise RuntimeError(
            f"Unexpected module counts: CCSE={len(ccse_modules)}, "
            f"Cross-MSEF={len(cross_modules)}, R={len(r_modules)}"
        )
    ccse = ccse_modules[0]

    candidate_backbone_frozen = all(not p.requires_grad for p in candidate_backbone.parameters())
    candidate_patch_frozen = all(
        not p.requires_grad for p in candidate_backbone.patch_embed.parameters()
    )
    parent_backbone_frozen = all(not p.requires_grad for p in parent_backbone.parameters())
    parent_patch_frozen = all(
        not p.requires_grad for p in parent_backbone.patch_embed.parameters()
    )
    if not all((candidate_backbone_frozen, candidate_patch_frozen, parent_backbone_frozen, parent_patch_frozen)):
        raise RuntimeError("Backbone or PatchEmbed is not frozen")

    common_init_diffs = []
    candidate_state = candidate.state_dict()
    parent_state = parent.state_dict()
    for name, value in parent_state.items():
        if name in candidate_state:
            common_init_diffs.append(
                float((candidate_state[name].detach().float() - value.detach().float()).abs().max().item())
            )
    common_init_max_abs_diff = max(common_init_diffs, default=0.0)

    x = torch.randn(1, 3, 512, 512, device=device)
    captured, capture_handle = _capture_ccse_input(candidate)
    with torch.no_grad():
        candidate_logits = candidate(x)
        parent_logits = parent(x)
    capture_handle.remove()
    if "pyramid" not in captured:
        raise RuntimeError("CCSE input hook did not capture the pyramid")
    identity_max_abs_diff = float(
        (candidate_logits - parent_logits).abs().max().item()
    )
    expected_logits_shape = [1, int(raw["num_classes"]), 512, 512]
    if list(candidate_logits.shape) != expected_logits_shape:
        raise RuntimeError(
            f"Unexpected candidate logits shape: {list(candidate_logits.shape)}"
        )
    if not torch.isfinite(candidate_logits).all().item():
        raise RuntimeError("Candidate forward produced non-finite logits")

    with torch.no_grad():
        _, ccse_trace = ccse(captured["pyramid"], return_trace=True)
    expected_pyramid_shapes = [
        [1, 256, 256, 256],
        [1, 256, 128, 128],
        [1, 256, 64, 64],
        [1, 256, 32, 32],
    ]
    if ccse_trace["input_shapes"] != expected_pyramid_shapes:
        raise RuntimeError(f"Unexpected CCSE input shapes: {ccse_trace['input_shapes']}")
    if ccse_trace["gathered_shapes"] != [[1, 256, 32, 32]] * 4:
        raise RuntimeError(f"Unexpected CCSE gathered shapes: {ccse_trace['gathered_shapes']}")
    if ccse_trace["projected_shapes"] != [[1, 64, 32, 32]] * 4:
        raise RuntimeError(f"Unexpected CCSE projected shapes: {ccse_trace['projected_shapes']}")
    if ccse_trace["token_shape"] != [1, 1024, 4, 64]:
        raise RuntimeError(f"Unexpected CCSE token shape: {ccse_trace['token_shape']}")
    if ccse_trace["attention_shape"] != [1, 1024, 4, 4, 4]:
        raise RuntimeError(f"Unexpected CCSE attention shape: {ccse_trace['attention_shape']}")
    if ccse_trace["output_shapes"] != expected_pyramid_shapes:
        raise RuntimeError(f"Unexpected CCSE output shapes: {ccse_trace['output_shapes']}")

    candidate_total, candidate_backbone_params, candidate_decoder_params = summarize_parameters(
        candidate, candidate_backbone
    )
    parent_total, parent_backbone_params, parent_decoder_params = summarize_parameters(
        parent, parent_backbone
    )
    candidate_trainable = _count(p for p in candidate.parameters() if p.requires_grad)
    parent_trainable = _count(p for p in parent.parameters() if p.requires_grad)
    ccse_params = _count(ccse.parameters())
    ccse_trainable = _count(p for p in ccse.parameters() if p.requires_grad)
    ccse_parameter_names = [
        name for name, parameter in candidate.named_parameters()
        if name.startswith("decoder.ccse.") and parameter.requires_grad
    ]

    # Release the two full models before the gradient and timing phases.
    del candidate, candidate_backbone, parent, parent_backbone
    del candidate_logits, parent_logits, captured, ccse_trace, ccse_modules, cross_modules, r_modules
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gradient_checks = _gradient_preflight(raw, device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    torch.manual_seed(314159)
    benchmark_x = torch.randn(args.batch_size, 3, 512, 512, device=device)
    benchmark_target = torch.randint(
        0, int(raw["num_classes"]), (args.batch_size, 512, 512), device=device
    )
    candidate_bench_model, candidate_bench_backbone = _build(raw, CANDIDATE_VARIANT, device)
    candidate_benchmark = _benchmark(
        candidate_bench_model,
        device,
        benchmark_x,
        benchmark_target,
        args.benchmark_steps,
        args.warmup_steps,
    )
    del candidate_bench_model, candidate_bench_backbone
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    parent_bench_model, parent_bench_backbone = _build(parent_raw, PARENT_VARIANT, device)
    parent_benchmark = _benchmark(
        parent_bench_model,
        device,
        benchmark_x,
        benchmark_target,
        args.benchmark_steps,
        args.warmup_steps,
    )
    del parent_bench_model, parent_bench_backbone, benchmark_x, benchmark_target
    gc.collect()

    latency_ratio = candidate_benchmark["seconds_per_step"] / max(
        parent_benchmark["seconds_per_step"], 1e-12
    )
    memory_delta_mib = candidate_benchmark["peak_memory_mib"] - parent_benchmark["peak_memory_mib"]
    latency_pass = latency_ratio <= 1.10
    memory_pass = memory_delta_mib <= 512.0
    identity_pass = identity_max_abs_diff < 1e-6
    init_pass = common_init_max_abs_diff < 1e-6
    overall_pass = all((init_pass, identity_pass, latency_pass, memory_pass))
    result = {
        "preflight": "PASS" if overall_pass else "NO-GO",
        "formal_training_started": False,
        "test_accessed": False,
        "config": str(config_path),
        "parent_config": str(parent_config_path),
        "candidate_variant": CANDIDATE_VARIANT,
        "parent_variant": PARENT_VARIANT,
        "device": str(device),
        "seed": SEED,
        "config_diff_allowed_only": True,
        "common_init_max_abs_diff": common_init_max_abs_diff,
        "identity_max_abs_diff_vs_fresh_parent": identity_max_abs_diff,
        "identity_threshold": 1e-6,
        "identity_pass": identity_pass,
        "shape_trace": {
            "ccse_input": expected_pyramid_shapes,
            "ccse_gathered": [[1, 256, 32, 32]] * 4,
            "ccse_projected": [[1, 64, 32, 32]] * 4,
            "tokens": [1, 1024, 4, 64],
            "qkv": [1, 1024, 4, 3, 4, 16],
            "attention": [1, 1024, 4, 4, 4],
            "distributed": [[1, 64, 32, 32]] * 4,
            "ccse_outputs": expected_pyramid_shapes,
            "logits": expected_logits_shape,
        },
        "module_counts": {
            "ccse": 1,
            "cross_msef": 3,
            "original_sad_r": 8,
        },
        "backbone_frozen": True,
        "patch_embed_frozen": True,
        "backbone_trainable_names": [],
        "params": {
            "candidate_total": candidate_total,
            "candidate_backbone": candidate_backbone_params,
            "candidate_decoder_and_stem": candidate_decoder_params,
            "candidate_trainable": candidate_trainable,
            "parent_total": parent_total,
            "parent_backbone": parent_backbone_params,
            "parent_decoder_and_stem": parent_decoder_params,
            "parent_trainable": parent_trainable,
            "ccse_total": ccse_params,
            "ccse_trainable": ccse_trainable,
            "candidate_minus_parent_total": candidate_total - parent_total,
            "candidate_minus_parent_trainable": candidate_trainable - parent_trainable,
            "ccse_trainable_parameter_names": ccse_parameter_names,
        },
        "gradient_checks": gradient_checks,
        "benchmark": {
            "candidate": candidate_benchmark,
            "parent": parent_benchmark,
            "candidate_over_parent_latency_ratio": latency_ratio,
            "candidate_minus_parent_peak_memory_mib": memory_delta_mib,
            "latency_threshold_ratio": 1.10,
            "memory_threshold_delta_mib": 512.0,
            "latency_pass": latency_pass,
            "memory_pass": memory_pass,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not overall_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
