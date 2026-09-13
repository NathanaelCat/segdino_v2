#!/usr/bin/env python3
"""Preflight the OSD MSEF replacement for the CTRL-002 decoder.

The check is deliberately strict.  It verifies that the candidate keeps the
CTRL-002 data/model/training protocol, replaces exactly the eight SAD-R
locations, preserves the TPA pyramid and output contract, and is exactly
parent-equivalent when the new outer MSEF residual coefficients are zero.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import ModelConfig  # noqa: E402
from dpt import MSEFResidualBlock, ResidualDepthwiseBlock  # noqa: E402
from runtime import build_model  # noqa: E402


PARENT_CONFIG = ROOT_DIR / "configs/dino_layer_ctrl_002_l12x4_seed20260901.json"
CANDIDATE_CONFIG = ROOT_DIR / "configs/dino_tpa_sad_msef_001_l12x4_constant_seed20260901.json"
OUTPUT_PATH = ROOT_DIR / "work_dirs/dino_tpa_sad_msef_001_preflight.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def _model_config(raw: dict[str, Any]) -> ModelConfig:
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


def _protocol_without_variant(raw: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(raw)
    value.pop("name", None)
    value.pop("run_dir", None)
    value.get("runtime", {}).pop("device", None)
    value.setdefault("model", {}).pop("decoder_variant", None)
    return value


def _count(parameters) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _count_modules(modules) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


def _norm(parameter: torch.Tensor | None) -> float | None:
    if parameter is None:
        return None
    return float(parameter.detach().float().norm().item())


def _finite_nonzero(values: dict[str, float | None]) -> bool:
    return all(value is not None and math.isfinite(value) and value > 1e-12 for value in values.values())


def _capture_branch_shapes(model: torch.nn.Module, x: torch.Tensor):
    captured: dict[str, list[int]] = {}
    hooks = []
    for name in ("tpa_branch_1", "tpa_branch_2", "tpa_branch_3", "tpa_branch_4"):
        module = getattr(model.decoder, name)
        hooks.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, name=name: captured.__setitem__(
                    name, list(output.shape)
                )
            )
        )
    try:
        with torch.inference_mode():
            logits = model(x)
    finally:
        for hook in hooks:
            hook.remove()
    return logits, captured


def _copy_common_state(parent: torch.nn.Module, candidate: torch.nn.Module) -> list[str]:
    parent_state = parent.state_dict()
    candidate_state = candidate.state_dict()
    common: list[str] = []
    for name, value in candidate_state.items():
        if name in parent_state and tuple(parent_state[name].shape) == tuple(value.shape):
            candidate_state[name] = parent_state[name].detach().clone()
            common.append(name)
    missing, unexpected = candidate.load_state_dict(candidate_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"candidate common-state load failed: missing={missing}, unexpected={unexpected}"
        )
    return sorted(common)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    parent_raw = _load(PARENT_CONFIG)
    candidate_raw = _load(CANDIDATE_CONFIG)
    if _protocol_without_variant(parent_raw) != _protocol_without_variant(candidate_raw):
        raise RuntimeError("Candidate differs from CTRL-002 outside decoder_variant/name/run_dir/device")
    if parent_raw["model"].get("decoder_variant", "tpa_sad") != "tpa_sad":
        raise RuntimeError("CTRL-002 parent config is not the ordinary tpa_sad variant")
    if candidate_raw["model"].get("decoder_variant") != "tpa_sad_msef":
        raise RuntimeError("Candidate config is not tpa_sad_msef")
    if candidate_raw["model"].get("layer_mapping") != [3, 3, 3, 3]:
        raise RuntimeError("MSEF experiment must use CTRL-002 L12x4 routing")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    parent, parent_backbone = build_model(_model_config(parent_raw), str(device))
    candidate, candidate_backbone = build_model(_model_config(candidate_raw), str(device))
    parent.lock_backbone()
    candidate.lock_backbone()

    if not all(not parameter.requires_grad for parameter in candidate_backbone.parameters()):
        raise RuntimeError("candidate backbone is not fully frozen")
    if not all(not parameter.requires_grad for parameter in candidate_backbone.patch_embed.parameters()):
        raise RuntimeError("candidate PatchEmbed is not frozen")

    common_state = _copy_common_state(parent, candidate)
    new_state = sorted(
        name for name in candidate.state_dict() if name not in set(common_state)
    )

    candidate_msef = [module for module in candidate.decoder.modules() if isinstance(module, MSEFResidualBlock)]
    candidate_r = [module for module in candidate.decoder.modules() if isinstance(module, ResidualDepthwiseBlock)]
    parent_r = [module for module in parent.decoder.modules() if isinstance(module, ResidualDepthwiseBlock)]
    if len(candidate_msef) != 8 or candidate_r:
        raise RuntimeError(
            f"expected exactly eight MSEF blocks and no ordinary R; got MSEF={len(candidate_msef)}, R={len(candidate_r)}"
        )
    if len(parent_r) != 8:
        raise RuntimeError(f"CTRL-002 parent expected eight ordinary R blocks, got {len(parent_r)}")

    x = torch.linspace(-1.0, 1.0, steps=3 * 512 * 512, device=device).reshape(1, 3, 512, 512)
    parent.eval()
    candidate.eval()
    parent_logits, parent_shapes = _capture_branch_shapes(parent, x)
    candidate_logits, candidate_shapes = _capture_branch_shapes(candidate, x)
    zero_logits_max_abs_diff = float((parent_logits - candidate_logits).abs().max().item())

    with torch.inference_mode():
        features = candidate.backbone.get_intermediate_layers(
            x, n=candidate.intermediate_layer_idx[candidate.encoder_size]
        )
    feature_shapes = [list(feature.shape) for feature in features]

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    loss = candidate(x).float().square().mean()
    loss.backward()
    alpha_grad = {
        f"gamma_{index + 1}": _norm(module.gamma.grad)
        for index, module in enumerate(candidate_msef)
    }
    core_parameters = {
        f"block_{index + 1}": {
            "layer_norm": max(
                _norm(module.layer_norm.weight.grad) or 0.0,
                _norm(module.layer_norm.bias.grad) or 0.0,
            ),
            "depthwise": max(
                _norm(module.depthwise_conv.weight.grad) or 0.0,
                _norm(module.depthwise_conv.bias.grad) or 0.0,
            ),
            "se": max(
                _norm(module.se_fc1.weight.grad) or 0.0,
                _norm(module.se_fc1.bias.grad) or 0.0,
                _norm(module.se_fc2.weight.grad) or 0.0,
                _norm(module.se_fc2.bias.grad) or 0.0,
            ),
        }
        for index, module in enumerate(candidate_msef)
    }
    zero_core_max_grad = max(value for block in core_parameters.values() for value in block.values())

    with torch.no_grad():
        for module in candidate_msef:
            module.gamma.fill_(1e-3)
    candidate.zero_grad(set_to_none=True)
    candidate(x).float().square().mean().backward()
    opened_core_grad = {
        f"block_{index + 1}": max(
            _norm(module.layer_norm.weight.grad) or 0.0,
            _norm(module.layer_norm.bias.grad) or 0.0,
            _norm(module.depthwise_conv.weight.grad) or 0.0,
            _norm(module.depthwise_conv.bias.grad) or 0.0,
            _norm(module.se_fc1.weight.grad) or 0.0,
            _norm(module.se_fc1.bias.grad) or 0.0,
            _norm(module.se_fc2.weight.grad) or 0.0,
            _norm(module.se_fc2.bias.grad) or 0.0,
        )
        for index, module in enumerate(candidate_msef)
    }
    candidate.eval()

    trainable_named = [
        (name, parameter.numel())
        for name, parameter in candidate.named_parameters()
        if parameter.requires_grad
    ]
    parent_trainable = _count(parameter for parameter in parent.parameters() if parameter.requires_grad)
    candidate_trainable = _count(parameter for parameter in candidate.parameters() if parameter.requires_grad)
    parent_total = _count(parent.parameters())
    candidate_total = _count(candidate.parameters())
    msef_params = _count_modules(candidate_msef)
    # The explicit expected delta makes accidental inclusion/removal of a
    # decoder component visible before the long run starts.
    ordinary_r_params = _count_modules(parent_r)
    # Per block: LN weight/bias, depthwise weight/bias, two SE linear layers,
    # and the outer gamma.  Keep this expression readable in the JSON audit.
    expected_msef_params = 8 * (2 * 256 + (256 * 3 * 3 + 256) + (256 * 16 + 16) + (16 * 256 + 256) + 1)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in candidate.parameters() if parameter.requires_grad],
        lr=float(candidate_raw["optimizer"]["lr"]),
        betas=tuple(candidate_raw["optimizer"]["betas"]),
        weight_decay=float(candidate_raw["optimizer"]["weight_decay"]),
    )

    expected_shapes = {
        "tpa_branch_1": [1, 256, 256, 256],
        "tpa_branch_2": [1, 256, 128, 128],
        "tpa_branch_3": [1, 256, 64, 64],
        "tpa_branch_4": [1, 256, 32, 32],
    }
    if parent_shapes != expected_shapes or candidate_shapes != expected_shapes:
        raise RuntimeError(
            f"unexpected TPA shapes: parent={parent_shapes}, candidate={candidate_shapes}"
        )
    if tuple(parent_logits.shape) != (1, 4, 512, 512) or tuple(candidate_logits.shape) != (1, 4, 512, 512):
        raise RuntimeError(
            f"unexpected logits shape: parent={tuple(parent_logits.shape)}, candidate={tuple(candidate_logits.shape)}"
        )
    if zero_logits_max_abs_diff > 1e-6:
        raise RuntimeError(f"gamma=0 parent equivalence failed: max_abs_diff={zero_logits_max_abs_diff}")
    if not _finite_nonzero(alpha_grad):
        raise RuntimeError(f"gamma learnability failed at zero: {alpha_grad}")
    if zero_core_max_grad > 1e-10:
        raise RuntimeError(f"MSEF core should be closed at gamma=0, max_grad={zero_core_max_grad}")
    if not all(math.isfinite(value) and value > 1e-12 for value in opened_core_grad.values()):
        raise RuntimeError(f"MSEF core did not open at gamma=1e-3: {opened_core_grad}")
    if msef_params != expected_msef_params:
        raise RuntimeError(f"MSEF parameter mismatch: actual={msef_params}, expected={expected_msef_params}")

    result = {
        "status": "PASS",
        "candidate": candidate_raw["name"],
        "parent": parent_raw["name"],
        "device": str(device),
        "input_shape": list(x.shape),
        "feature_shapes_before_decoder": feature_shapes,
        "routing": candidate_raw["model"]["layer_mapping"],
        "backbone_frozen": all(not parameter.requires_grad for parameter in candidate_backbone.parameters()),
        "patch_embed_frozen": all(not parameter.requires_grad for parameter in candidate_backbone.patch_embed.parameters()),
        "parent_decoder_variant": parent.decoder_variant,
        "candidate_decoder_variant": candidate.decoder_variant,
        "parent_ordinary_sad_r_blocks": len(parent_r),
        "candidate_ordinary_sad_r_blocks": len(candidate_r),
        "candidate_msef_blocks": len(candidate_msef),
        "msef_reduction_ratio": 16,
        "common_state_keys_copied": len(common_state),
        "candidate_new_state_key_count": len(new_state),
        "candidate_new_state_prefixes": sorted({name.split(".")[1] for name in new_state if name.startswith("decoder.")}),
        "tpa_branch_shapes_parent": parent_shapes,
        "tpa_branch_shapes_candidate": candidate_shapes,
        "logits_shape": list(candidate_logits.shape),
        "gamma_zero_parent_logits_max_abs_diff": zero_logits_max_abs_diff,
        "gamma_zero_grad_norms": alpha_grad,
        "gamma_zero_msef_core_max_grad_norm": zero_core_max_grad,
        "gamma_1e-3_msef_core_grad_norms": opened_core_grad,
        "params": {
            "parent_total": parent_total,
            "candidate_total": candidate_total,
            "parent_trainable": parent_trainable,
            "candidate_trainable": candidate_trainable,
            "trainable_delta": candidate_trainable - parent_trainable,
            "ordinary_r_params_removed": ordinary_r_params,
            "msef_params_added": msef_params,
            "expected_msef_params_added": expected_msef_params,
        },
        "optimizer": {
            "name": "AdamW",
            "lr": float(candidate_raw["optimizer"]["lr"]),
            "betas": candidate_raw["optimizer"]["betas"],
            "weight_decay": float(candidate_raw["optimizer"]["weight_decay"]),
            "parameter_count": len(optimizer.param_groups[0]["params"]),
            "parameter_numel": _count(optimizer.param_groups[0]["params"]),
            "parameter_names": [name for name, _ in trainable_named],
        },
        "peak_memory_allocated_mib": (
            float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None
        ),
        "protocol_diff": "only model.decoder_variant changes; name/run_dir/device are metadata/runtime overrides",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("MSEF_PREFLIGHT=PASS")
    print(f"device={device}")
    print(f"parent={parent_raw['name']} variant={parent.decoder_variant}")
    print(f"candidate={candidate_raw['name']} variant={candidate.decoder_variant}")
    print(f"backbone_frozen={result['backbone_frozen']} patch_embed_frozen={result['patch_embed_frozen']}")
    print(f"features={feature_shapes}")
    print(f"TPA_shapes={candidate_shapes}")
    print(f"logits_shape={tuple(candidate_logits.shape)}")
    print(f"MSEF_blocks={len(candidate_msef)} ordinary_R_in_candidate={len(candidate_r)}")
    print(f"params_parent_total={parent_total:,} params_candidate_total={candidate_total:,}")
    print(f"params_parent_trainable={parent_trainable:,} params_candidate_trainable={candidate_trainable:,}")
    print(f"trainable_delta_vs_CTRL002={candidate_trainable - parent_trainable:+,}")
    print(f"ordinary_R_removed={ordinary_r_params:,} MSEF_added={msef_params:,}")
    print(f"gamma0_logits_max_abs_diff={zero_logits_max_abs_diff:.9g}")
    print(f"gamma0_grad_norms={alpha_grad}")
    print(f"gamma0_core_max_grad_norm={zero_core_max_grad:.9g}")
    print(f"gamma1e-3_core_grad_norms={opened_core_grad}")
    print(f"peak_memory_allocated_mib={result['peak_memory_allocated_mib']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
