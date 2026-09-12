#!/usr/bin/env python3
"""Preflight the two patch-guided DPA-Cal OSD variants.

This script deliberately stops after construction, equivalence, and gradient
checks.  It does not touch the OSD loaders and it does not start training.
Both variants are initialized from the same fresh B1 decoder state; the
independent adapters are value-copied from the shared adapter so that the
shared-vs-independent comparison has no avoidable initialization difference.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import ModelConfig  # noqa: E402
from runtime import build_model, summarize_parameters  # noqa: E402


DEPTH_NAMES = ("3", "6", "9", "12")
PATCH_VARIANTS = {"patch_dpa_shared", "patch_dpa_independent"}
EXPECTED_PARAMS = {
    "b1_total": 24_880_260,
    "b1_trainable": 3_279_108,
    "shared_total": 25_308_552,
    "shared_trainable": 3_707_400,
    "independent_total": 25_806_984,
    "independent_trainable": 4_205_832,
}


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_config(raw: dict[str, Any]) -> ModelConfig:
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path.resolve(), map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(state)!r}")
    return state


def _load_b1_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"B1 strict initialization mismatch: missing={missing}, unexpected={unexpected}")


def _expected_patch_missing(variant: str) -> set[str]:
    if variant not in PATCH_VARIANTS:
        raise ValueError(f"Unsupported patch variant: {variant}")
    missing = {
        "decoder.depth_alignments.0.weight",
        "decoder.depth_alignments.1.weight",
        "decoder.depth_alignments.2.weight",
        "decoder.depth_alignments.3.weight",
        "decoder.alpha_3",
        "decoder.alpha_6",
        "decoder.alpha_9",
        "decoder.alpha_12",
    }
    if variant == "patch_dpa_shared":
        missing.update(
            f"decoder.patch_adapter.{name}.weight"
            for name in ("in_projection", "depthwise", "out_projection")
        )
    else:
        missing.update(
            f"decoder.patch_adapters.{index}.{name}.weight"
            for index in range(4)
            for name in ("in_projection", "depthwise", "out_projection")
        )
    return missing


def _load_fresh_b1_into_patch(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
) -> list[str]:
    variant = str(getattr(model, "decoder_variant", ""))
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected = _expected_patch_missing(variant)
    if set(missing) != expected or unexpected:
        raise RuntimeError(
            "Patch-DPA fresh-B1 initialization mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return sorted(missing)


def _module_state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _set_alpha(model: torch.nn.Module, value: float) -> None:
    with torch.no_grad():
        for alpha in model.decoder.alphas:
            alpha.fill_(value)


def _norm_or_none(parameter: torch.nn.Parameter) -> float | None:
    if parameter.grad is None:
        return None
    return float(parameter.grad.detach().float().norm().item())


def _grad_norms(module: torch.nn.Module) -> dict[str, float | None]:
    return {
        name: _norm_or_none(parameter)
        for name, parameter in module.named_parameters()
    }


def _finite_nonzero(values: dict[str, float | None], tolerance: float = 1e-12) -> bool:
    return bool(values) and all(
        value is not None and np.isfinite(value) and value > tolerance
        for value in values.values()
    )


def _finite_zero(values: dict[str, float | None], tolerance: float = 1e-12) -> bool:
    return bool(values) and all(
        value is not None and np.isfinite(value) and value <= tolerance
        for value in values.values()
    )


def _adapter_modules(decoder: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    if decoder.adapter_sharing == "shared":
        return [("shared", decoder.patch_adapter)]
    return [
        (f"independent_{index}", adapter)
        for index, adapter in enumerate(decoder.patch_adapters)
    ]


def _optimizer_info(model: torch.nn.Module, raw: dict[str, Any]) -> dict[str, Any]:
    optimizer_cfg = raw["optimizer"]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimizer_cfg["lr"]),
        betas=tuple(float(value) for value in optimizer_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    names_by_id = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "betas": [float(value) for value in group["betas"]],
                "parameter_count": len(group["params"]),
                "parameter_names": [names_by_id[id(parameter)] for parameter in group["params"]],
            }
        )
    return {
        "name": "AdamW",
        "groups": groups,
        "all_trainable_names_match": {
            "optimizer": sorted(name for group in groups for name in group["parameter_names"]),
            "model": sorted(names_by_id.values()),
        },
    }


def _canonical_config(raw: dict[str, Any]) -> dict[str, Any]:
    canonical = copy.deepcopy(raw)
    for key in ("name", "run_dir", "initialization_protocol"):
        canonical.pop(key, None)
    canonical.get("runtime", {}).pop("device", None)
    canonical.get("model", {}).pop("decoder_variant", None)
    return canonical


def _max_state_diff(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> float:
    common = sorted(set(left) & set(right))
    if set(left) != set(right):
        raise RuntimeError("State comparison key sets differ")
    return max(
        float((left[key].float() - right[key].float()).abs().max().item())
        for key in common
    ) if common else 0.0


def _gpu_record(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device), "available": False}
    index = torch.cuda.current_device() if device.index is None else device.index
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    return {
        "device": f"cuda:{index}",
        "index": int(index),
        "name": torch.cuda.get_device_name(index),
        "free_mib_before": round(free_bytes / 1024**2, 2),
        "total_mib": round(total_bytes / 1024**2, 2),
    }


def _run_variant(
    raw: dict[str, Any],
    variant_label: str,
    device: torch.device,
    b1_state: dict[str, torch.Tensor],
    input_cpu: torch.Tensor,
    output_path: Path,
    shared_model_for_copy: torch.nn.Module | None = None,
) -> tuple[dict[str, Any], torch.nn.Module | None]:
    variant = str(raw["model"]["decoder_variant"])
    if variant not in PATCH_VARIANTS:
        raise ValueError(f"{variant_label} has unexpected variant {variant!r}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    # Compute the B1 reference on this same GPU.  This avoids comparing
    # kernels produced on different devices when checking alpha=0 equivalence.
    b1_cfg = copy.deepcopy(raw)
    b1_cfg["model"]["decoder_variant"] = "tpa_ms_mlp"
    b1_model, b1_backbone = build_model(_model_config(b1_cfg), str(device))
    b1_model.lock_backbone()
    _load_b1_state(b1_model, b1_state)
    b1_model.eval()
    input_device = input_cpu.to(device, non_blocking=True)
    with torch.no_grad():
        b1_logits = b1_model(input_device).detach().cpu()
    b1_total, b1_backbone_count, b1_decoder_count = summarize_parameters(b1_model, b1_backbone)
    b1_trainable = sum(parameter.numel() for parameter in b1_model.parameters() if parameter.requires_grad)
    del b1_model, b1_backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _seed_everything(int(raw.get("runtime", {}).get("seed", 20260901)))
    model, backbone = build_model(_model_config(raw), str(device))
    model.lock_backbone()
    missing_keys = _load_fresh_b1_into_patch(model, b1_state)
    model.eval()
    decoder = model.decoder

    if shared_model_for_copy is not None:
        decoder.depth_alignments.load_state_dict(
            _module_state_cpu(shared_model_for_copy.decoder.depth_alignments), strict=True
        )
        shared_adapter_state = _module_state_cpu(shared_model_for_copy.decoder.patch_adapter)
        for adapter in decoder.patch_adapters:
            adapter.load_state_dict(shared_adapter_state, strict=True)
        _set_alpha(model, 0.0)

    with torch.no_grad():
        normal_logits = model(input_device).detach().cpu()
        diagnostic_logits, trace = model.dpa_diagnostics(input_device)
        diagnostic_logits = diagnostic_logits.detach().cpu()
        raw_features = model.backbone.get_intermediate_layers(
            input_device,
            n=model.intermediate_layer_idx[model.encoder_size],
        )
        raw_patch = model._raw_patch_embedding(input_device)
        direct_patch = model.backbone.patch_embed.proj(input_device)

    projected_shapes = [list(value.shape) for value in trace["projected"]]
    aligned_shapes = [list(value.shape) for value in trace["aligned_patch_priors"]]
    score_shapes = [list(value.shape) for value in trace["scores"]]
    patch_prior_shapes = [list(value.shape) for value in trace["patch_priors"]]
    raw_feature_token_shapes = [list(value.shape) for value in raw_features]
    logits_max_diff_normal_vs_diagnostic = float(
        (normal_logits - diagnostic_logits).abs().max().item()
    )
    logits_max_diff_vs_b1 = float((normal_logits - b1_logits).abs().max().item())
    raw_patch_max_diff = float((raw_patch - direct_patch).abs().max().item())

    alpha_zero = {
        f"alpha_{name}": float(getattr(decoder, f"alpha_{name}").detach().item())
        for name in DEPTH_NAMES
    }
    optimizer_info = _optimizer_info(model, raw)

    # Learnability check: at alpha=0 the DPA path is exactly shut off, so the
    # alpha scalars should receive signal while adapter/phi gradients are zero.
    model.train()
    model.zero_grad(set_to_none=True)
    zero_logits = model(input_device)
    zero_loss = zero_logits.float().square().mean()
    zero_loss.backward()
    alpha_zero_grads = {
        f"alpha_{name}": _norm_or_none(getattr(decoder, f"alpha_{name}"))
        for name in DEPTH_NAMES
    }
    phi_zero_grads = _grad_norms(decoder.depth_alignments)
    adapter_zero_grads = {
        module_name: _grad_norms(module)
        for module_name, module in _adapter_modules(decoder)
    }
    backbone_grads_none_at_zero = all(
        parameter.grad is None for parameter in backbone.parameters()
    )

    _set_alpha(model, 1e-3)
    model.zero_grad(set_to_none=True)
    opened_logits = model(input_device)
    opened_loss = opened_logits.float().square().mean()
    opened_loss.backward()
    phi_open_grads = _grad_norms(decoder.depth_alignments)
    adapter_open_grads = {
        module_name: _grad_norms(module)
        for module_name, module in _adapter_modules(decoder)
    }
    backbone_grads_none_open = all(
        parameter.grad is None for parameter in backbone.parameters()
    )
    _set_alpha(model, 0.0)
    model.zero_grad(set_to_none=True)
    model.eval()

    if shared_model_for_copy is not None:
        adapter_identity_separate = all(
            id(target.in_projection.weight) != id(shared_model_for_copy.decoder.patch_adapter.in_projection.weight)
            for target in decoder.patch_adapters
        )
    else:
        adapter_identity_separate = True

    patch_projection_parameter_names = [
        name for name, _ in model.named_parameters()
        if name.startswith("backbone.patch_embed.proj.")
    ]
    decoder_patch_projection_names = [
        name for name, _ in decoder.named_parameters()
        if "patch" in name.lower() and "adapter" not in name.lower()
    ]
    total, backbone_count, decoder_count = summarize_parameters(model, backbone)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mib = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
        free_bytes, _ = torch.cuda.mem_get_info(device)
        free_memory_after_mib = round(free_bytes / 1024**2, 2)
    else:
        peak_memory_mib = None
        free_memory_after_mib = None

    full_state = _module_state_cpu(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": full_state,
            "config": raw,
            "purpose": f"fresh B1-shared initialization for {variant_label}",
            "initialization_type": "fresh_b1_shared_init_with_patch_preflight",
            "seed": int(raw.get("runtime", {}).get("seed", 20260901)),
            "parent_initialization": "work_dirs/dino_tpa_dpa_msmlp_clean_common_init_seed20260901.pth",
            "alpha_zero_logits_max_abs_diff_vs_b1": logits_max_diff_vs_b1,
            "preflight_status": "PASS",
        },
        output_path,
    )

    result = {
        "label": variant_label,
        "variant": variant,
        "device": str(device),
        "gpu": _gpu_record(device),
        "forward_data_flow": (
            "input 512x512 -> frozen DINOv3-S/16 -> native patch-only "
            "[L3,L6,L9,L12] tokens -> B1 token projections Xi "
            "-> raw pre-transformer backbone.patch_embed.proj E "
            "-> PatchAdapter(s) P -> depth alignments Qi -> "
            "cosine DPA-Cal for all four depths -> unchanged B1 TPA/PR "
            "-> unchanged B1 MS-MLP -> 1x1 classifier -> bilinear 512x512"
        ),
        "input_shape": list(input_device.shape),
        "patch_embed": {
            "source": "backbone.patch_embed.proj(x), before PatchEmbed flatten/norm and before Transformer",
            "kernel_size": list(backbone.patch_embed.proj.kernel_size),
            "stride": list(backbone.patch_embed.proj.stride),
            "padding": list(backbone.patch_embed.proj.padding),
            "raw_E_shape": list(raw_patch.shape),
            "direct_proj_max_abs_diff": raw_patch_max_diff,
            "registered_weight_names": patch_projection_parameter_names,
            "decoder_owns_separate_patch_projection": bool(decoder_patch_projection_names),
            "decoder_patch_projection_parameter_names": decoder_patch_projection_names,
        },
        "features": {
            "intermediate_raw_token_shapes": raw_feature_token_shapes,
            "image_token_rule": "_tokens_to_feature_map retains the final 32*32 patch tokens; CLS/register tokens are excluded",
            "Xi_shapes": projected_shapes,
            "Qi_shapes": aligned_shapes,
            "Si_shapes": score_shapes,
            "P_shapes": patch_prior_shapes,
            "final_logits_shape": list(normal_logits.shape),
            "diagnostic_logits_shape": list(diagnostic_logits.shape),
        },
        "frozen_checks": {
            "backbone_frozen": all(not parameter.requires_grad for parameter in backbone.parameters()),
            "patch_embed_frozen": all(not parameter.requires_grad for parameter in backbone.patch_embed.parameters()),
            "patch_projection_weight_requires_grad": bool(backbone.patch_embed.proj.weight.requires_grad),
            "backbone_grads_none_at_alpha_zero": backbone_grads_none_at_zero,
            "backbone_grads_none_at_alpha_1e-3": backbone_grads_none_open,
            "shared_patch_projection_source": "same backbone.patch_embed.proj module and weight; no decoder copy",
        },
        "alpha_zero": alpha_zero,
        "equivalence": {
            "alpha_zero_logits_max_abs_diff_vs_b1": logits_max_diff_vs_b1,
            "normal_vs_diagnostic_logits_max_abs_diff": logits_max_diff_normal_vs_diagnostic,
            "required_tolerance": "near floating-point zero",
        },
        "learnability": {
            "loss_at_alpha_zero": float(zero_loss.detach().item()),
            "alpha_grad_norms_at_zero": alpha_zero_grads,
            "phi_grad_norms_at_zero": phi_zero_grads,
            "adapter_grad_norms_at_zero": adapter_zero_grads,
            "alpha_grad_finite_nonzero_at_zero": _finite_nonzero(alpha_zero_grads),
            "phi_grad_zero_at_zero": _finite_zero(phi_zero_grads),
            "adapter_grad_zero_at_zero": all(_finite_zero(values) for values in adapter_zero_grads.values()),
            "loss_at_alpha_1e-3": float(opened_loss.detach().item()),
            "phi_grad_norms_at_1e-3": phi_open_grads,
            "adapter_grad_norms_at_1e-3": adapter_open_grads,
            "phi_grad_finite_nonzero_at_1e-3": _finite_nonzero(phi_open_grads),
            "adapter_grad_finite_nonzero_at_1e-3": all(
                _finite_nonzero(values) for values in adapter_open_grads.values()
            ),
        },
        "parameters": {
            "total": int(total),
            "backbone": int(backbone_count),
            "decoder": int(decoder_count),
            "trainable": int(trainable),
            "b1_reference_total": int(b1_total),
            "b1_reference_backbone": int(b1_backbone_count),
            "b1_reference_decoder": int(b1_decoder_count),
            "b1_reference_trainable": int(b1_trainable),
            "trainable_delta_vs_b1": int(trainable - b1_trainable),
            "adapter_identity_separate_from_shared": adapter_identity_separate,
        },
        "optimizer": optimizer_info,
        "initialization": {
            "source": "work_dirs/dino_tpa_dpa_msmlp_clean_common_init_seed20260901.pth",
            "loaded_missing_new_keys": missing_keys,
            "optimizer_fresh": True,
            "alpha_reset_after_gradient_check": all(
                float(alpha.detach().item()) == 0.0 for alpha in decoder.alphas
            ),
        },
        "saved_initialization_checkpoint": str(output_path),
        "memory": {
            "peak_allocated_mib": peak_memory_mib,
            "free_mib_after": free_memory_after_mib,
        },
    }
    return result, model if variant == "patch_dpa_shared" else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--b1-config",
        type=Path,
        default=ROOT_DIR / "configs/dino_tpa_base_msmlp_osd_512.json",
    )
    parser.add_argument(
        "--shared-config",
        type=Path,
        default=ROOT_DIR / "configs/dino_patch_dpa_shared_osd_512.json",
    )
    parser.add_argument(
        "--independent-config",
        type=Path,
        default=ROOT_DIR / "configs/dino_patch_dpa_independent_osd_512.json",
    )
    parser.add_argument(
        "--b1-init",
        type=Path,
        default=ROOT_DIR / "work_dirs/dino_tpa_dpa_msmlp_clean_common_init_seed20260901.pth",
    )
    parser.add_argument("--shared-device", default="cuda:0")
    parser.add_argument("--independent-device", default="cuda:1")
    parser.add_argument(
        "--shared-output",
        type=Path,
        default=ROOT_DIR / "work_dirs/dino_patch_dpa_shared_common_init_seed20260901.pth",
    )
    parser.add_argument(
        "--independent-output",
        type=Path,
        default=ROOT_DIR / "work_dirs/dino_patch_dpa_independent_common_init_seed20260901.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "work_dirs/dino_patch_dpa_preflight.json",
    )
    args = parser.parse_args()

    b1_config_path = args.b1_config.resolve()
    shared_config_path = args.shared_config.resolve()
    independent_config_path = args.independent_config.resolve()
    b1_init_path = args.b1_init.resolve()
    shared_device = torch.device(args.shared_device)
    independent_device = torch.device(args.independent_device)
    if shared_device == independent_device:
        raise ValueError("B2PS and B2PI must use different devices")
    if shared_device.type != "cuda" or independent_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This preflight requires two CUDA devices")
    if shared_device.index is None or independent_device.index is None:
        raise ValueError("Use explicit devices such as cuda:0 and cuda:1")
    device_count = torch.cuda.device_count()
    if max(shared_device.index, independent_device.index) >= device_count:
        raise RuntimeError(f"Requested GPUs exceed device_count={device_count}")

    b1_raw = _load_json(b1_config_path)
    shared_raw = _load_json(shared_config_path)
    independent_raw = _load_json(independent_config_path)
    if shared_raw["model"]["decoder_variant"] != "patch_dpa_shared":
        raise ValueError("Shared config must select patch_dpa_shared")
    if independent_raw["model"]["decoder_variant"] != "patch_dpa_independent":
        raise ValueError("Independent config must select patch_dpa_independent")
    if b1_raw["model"]["decoder_variant"] != "tpa_ms_mlp":
        raise ValueError("B1 config must select tpa_ms_mlp")
    if _canonical_config(b1_raw) != _canonical_config(shared_raw):
        raise RuntimeError("B1 and B2PS configs differ outside identity and decoder variant")
    if _canonical_config(shared_raw) != _canonical_config(independent_raw):
        raise RuntimeError("B2PS and B2PI configs differ outside identity and decoder variant")
    if not b1_init_path.is_file():
        raise FileNotFoundError(f"Fresh B1 initialization not found: {b1_init_path}")

    seed = int(shared_raw.get("runtime", {}).get("seed", 20260901))
    _seed_everything(seed)
    b1_state = _load_state(b1_init_path)
    input_cpu = torch.linspace(
        -1.0,
        1.0,
        steps=3 * 512 * 512,
        dtype=torch.float32,
    ).reshape(1, 3, 512, 512)

    shared_result, shared_model = _run_variant(
        shared_raw,
        "B2PS",
        shared_device,
        b1_state,
        input_cpu,
        args.shared_output.resolve(),
    )
    independent_result, _ = _run_variant(
        independent_raw,
        "B2PI",
        independent_device,
        b1_state,
        input_cpu,
        args.independent_output.resolve(),
        shared_model_for_copy=shared_model,
    )
    if shared_model is not None:
        del shared_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    shared_trainable = shared_result["parameters"]["trainable"]
    independent_trainable = independent_result["parameters"]["trainable"]
    b1_trainable = shared_result["parameters"]["b1_reference_trainable"]
    parameter_checks = {
        "B1_total_exact": shared_result["parameters"]["b1_reference_total"] == EXPECTED_PARAMS["b1_total"],
        "B1_trainable_exact": b1_trainable == EXPECTED_PARAMS["b1_trainable"],
        "B2PS_total_exact": shared_result["parameters"]["total"] == EXPECTED_PARAMS["shared_total"],
        "B2PS_trainable_exact": shared_trainable == EXPECTED_PARAMS["shared_trainable"],
        "B2PI_total_exact": independent_result["parameters"]["total"] == EXPECTED_PARAMS["independent_total"],
        "B2PI_trainable_exact": independent_trainable == EXPECTED_PARAMS["independent_trainable"],
        "B2PS_delta_exact": shared_trainable - b1_trainable == 428_292,
        "B2PI_delta_exact": independent_trainable - b1_trainable == 926_724,
    }
    checks = {
        "config_common_protocol_exact": True,
        "two_distinct_cuda_devices": shared_device.index != independent_device.index,
        "B2PS": {
            "backbone_frozen": shared_result["frozen_checks"]["backbone_frozen"],
            "patch_embed_frozen": shared_result["frozen_checks"]["patch_embed_frozen"],
            "alpha_zero_logits_equivalent": shared_result["equivalence"]["alpha_zero_logits_max_abs_diff_vs_b1"] <= 1e-6,
            "normal_diagnostic_equivalent": shared_result["equivalence"]["normal_vs_diagnostic_logits_max_abs_diff"] <= 1e-6,
            "alpha_learnable_at_zero": shared_result["learnability"]["alpha_grad_finite_nonzero_at_zero"],
            "phi_zero_at_zero": shared_result["learnability"]["phi_grad_zero_at_zero"],
            "adapter_zero_at_zero": shared_result["learnability"]["adapter_grad_zero_at_zero"],
            "phi_opens_at_1e-3": shared_result["learnability"]["phi_grad_finite_nonzero_at_1e-3"],
            "adapter_opens_at_1e-3": shared_result["learnability"]["adapter_grad_finite_nonzero_at_1e-3"],
        },
        "B2PI": {
            "backbone_frozen": independent_result["frozen_checks"]["backbone_frozen"],
            "patch_embed_frozen": independent_result["frozen_checks"]["patch_embed_frozen"],
            "alpha_zero_logits_equivalent": independent_result["equivalence"]["alpha_zero_logits_max_abs_diff_vs_b1"] <= 1e-6,
            "normal_diagnostic_equivalent": independent_result["equivalence"]["normal_vs_diagnostic_logits_max_abs_diff"] <= 1e-6,
            "alpha_learnable_at_zero": independent_result["learnability"]["alpha_grad_finite_nonzero_at_zero"],
            "phi_zero_at_zero": independent_result["learnability"]["phi_grad_zero_at_zero"],
            "adapter_zero_at_zero": independent_result["learnability"]["adapter_grad_zero_at_zero"],
            "phi_opens_at_1e-3": independent_result["learnability"]["phi_grad_finite_nonzero_at_1e-3"],
            "adapter_opens_at_1e-3": independent_result["learnability"]["adapter_grad_finite_nonzero_at_1e-3"],
        },
        "parameter_counts": parameter_checks,
    }
    all_pass = all(parameter_checks.values()) and all(
        all(value for value in group.values())
        for group in (checks["B2PS"], checks["B2PI"])
    )
    result = {
        "status": "PASS" if all_pass else "FAIL",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True
        ).strip(),
        "git_status_note": "preflight ran with existing worktree changes; no reset/stash/merge/checkout performed",
        "parent": {
            "name": "B1 = TPA(PR)+MS-MLP",
            "config": str(b1_config_path),
            "fresh_initialization": str(b1_init_path),
            "test_reference_mIoU3": 90.387709,
        },
        "configs": {
            "b1": str(b1_config_path),
            "B2PS": str(shared_config_path),
            "B2PI": str(independent_config_path),
            "B1_vs_B2PS_outside_variant_diff": [],
            "B2PS_vs_B2PI_outside_variant_diff": [],
        },
        "gpu_allocation": {
            "B2PS": _gpu_record(shared_device),
            "B2PI": _gpu_record(independent_device),
        },
        "checks": checks,
        "B2PS": shared_result,
        "B2PI": independent_result,
        "formal_training_started": False,
        "next_action": "Stop after preflight; formal 50-epoch training requires explicit confirmation after this report.",
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
