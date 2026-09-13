#!/usr/bin/env python3
"""Preflight for the OSD-adapted ChangeViT cascade decoder.

This check never accesses OSD images.  It verifies that the candidate keeps
the B1 frozen DINO/TPA pyramid and changes only the decoder consumer to the
coarse-to-fine cascade adapted from the official ChangeViT implementation.
"""

from __future__ import annotations

import copy
import json
import random
import sys
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


PARENT_CONFIG = ROOT_DIR / "configs/dino_tpa_base_msmlp_osd_512.json"
CANDIDATE_CONFIG = ROOT_DIR / "configs/dino_tpa_change_cascade_osd_512.json"
DEFAULT_OUTPUT = ROOT_DIR / "work_dirs/dino_change_cascade_preflight.json"


def _load_raw(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


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


def _strip_identity_metadata(raw: dict) -> dict:
    value = copy.deepcopy(raw)
    for key in ("name", "run_dir", "adaptation_note", "reference"):
        value.pop(key, None)
    value.get("runtime", {}).pop("device", None)
    value.get("model", {}).pop("decoder_variant", None)
    return value


def _config_differences(left: dict, right: dict) -> list[tuple[str, object, object]]:
    differences: list[tuple[str, object, object]] = []

    def visit(a, b, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differences.append((f"{path}.{key}", a.get(key), b.get(key)))
                else:
                    visit(a[key], b[key], f"{path}.{key}")
            return
        if a != b:
            differences.append((path, a, b))

    visit(_strip_identity_metadata(left), _strip_identity_metadata(right), "config")
    return differences


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _shape(value: torch.Tensor) -> list[int]:
    return list(value.shape)


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _module_names(model: nn.Module, prefix: str) -> list[str]:
    return [name for name, _ in model.named_modules() if name.startswith(prefix)]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-config", type=Path, default=PARENT_CONFIG)
    parser.add_argument("--config", type=Path, default=CANDIDATE_CONFIG)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    parent_raw = _load_raw(args.parent_config.resolve())
    candidate_raw = _load_raw(args.config.resolve())
    parent_cfg = _model_config(parent_raw)
    candidate_cfg = _model_config(candidate_raw)

    if parent_cfg.decoder_variant != "tpa_ms_mlp":
        raise RuntimeError(
            f"Expected B1 parent decoder_variant='tpa_ms_mlp', got {parent_cfg.decoder_variant!r}"
        )
    if candidate_cfg.decoder_variant != "tpa_change_cascade":
        raise RuntimeError(
            "Expected candidate decoder_variant='tpa_change_cascade', "
            f"got {candidate_cfg.decoder_variant!r}"
        )
    if parent_cfg.layer_mapping is not None or candidate_cfg.layer_mapping is not None:
        raise RuntimeError("B1 and ChangeCascade must use native [L3,L6,L9,L12] routing")
    if not parent_cfg.freeze_backbone or not candidate_cfg.freeze_backbone:
        raise RuntimeError("B1 and ChangeCascade must keep the DINO backbone frozen")
    differences = _config_differences(parent_raw, candidate_raw)
    if differences:
        raise RuntimeError(f"Non-architecture config differences detected: {differences}")
    if args.img_size % candidate_cfg.patch_size:
        raise ValueError("img-size must be divisible by the DINO patch size")

    device = torch.device(args.device)
    seed = int(candidate_raw.get("runtime", {}).get("seed", 20260901))

    # Re-seeding before each construction makes the common B1 prefix
    # initialization directly comparable without loading a trained checkpoint.
    _seed(seed)
    parent, parent_backbone = build_model(parent_cfg, str(device))
    parent.lock_backbone()
    _seed(seed)
    candidate, candidate_backbone = build_model(candidate_cfg, str(device))
    candidate.lock_backbone()

    parent_state = parent.state_dict()
    candidate_state = candidate.state_dict()
    shared_prefixes = (
        "backbone.",
        "decoder.token_projections.",
        "decoder.tpa_branch_",
    )
    shared_keys = sorted(
        key
        for key in parent_state
        if key in candidate_state and key.startswith(shared_prefixes)
    )
    if not shared_keys:
        raise RuntimeError("No shared B1/ChangeCascade initialization keys found")
    shared_init_diff = max(
        _max_abs_diff(parent_state[key], candidate_state[key]) for key in shared_keys
    )

    candidate.eval()
    patch_h = args.img_size // candidate.patch_size
    patch_w = args.img_size // candidate.patch_size
    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    with torch.inference_mode():
        features = candidate_backbone.get_intermediate_layers(
            x, n=candidate.intermediate_layer_idx[candidate.encoder_size]
        )
        projected = candidate.decoder._project_tokens(features, patch_h, patch_w)
        pyramid = candidate.decoder._build_tpa_pyramid(projected)
        cascade_stages = candidate.decoder._cascade(pyramid)
        decoder_logits = candidate.decoder.classifier(cascade_stages[-1])
        candidate_logits = candidate(x)

    expected_pyramid = [
        (args.batch_size, candidate_cfg.decoder_dim, 256, 256),
        (args.batch_size, candidate_cfg.decoder_dim, 128, 128),
        (args.batch_size, candidate_cfg.decoder_dim, 64, 64),
        (args.batch_size, candidate_cfg.decoder_dim, 32, 32),
    ]
    if [tuple(value.shape) for value in pyramid] != expected_pyramid:
        raise RuntimeError(
            f"Pyramid shape mismatch: got {[tuple(value.shape) for value in pyramid]}, "
            f"expected {expected_pyramid}"
        )
    expected_cascade = [
        (args.batch_size, candidate_cfg.decoder_dim, 32, 32),
        (args.batch_size, candidate_cfg.decoder_dim, 64, 64),
        (args.batch_size, candidate_cfg.decoder_dim, 128, 128),
        (args.batch_size, candidate_cfg.decoder_dim, 256, 256),
    ]
    if [tuple(value.shape) for value in cascade_stages] != expected_cascade:
        raise RuntimeError(
            f"Cascade shape mismatch: got {[tuple(value.shape) for value in cascade_stages]}, "
            f"expected {expected_cascade}"
        )
    expected_logits = (args.batch_size, candidate_cfg.num_classes, args.img_size, args.img_size)
    if tuple(candidate_logits.shape) != expected_logits:
        raise RuntimeError(
            f"Candidate logits shape mismatch: got {tuple(candidate_logits.shape)}, "
            f"expected {expected_logits}"
        )

    # One real backward pass checks the trainable path without touching any
    # dataset split.  The outer DPT interpolation is part of the tested path.
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    logits = candidate(x)
    target = torch.zeros(
        args.batch_size, args.img_size, args.img_size, dtype=torch.long, device=device
    )
    F.cross_entropy(logits, target).backward()
    backbone_grad_names = [
        name for name, parameter in candidate_backbone.named_parameters() if parameter.grad is not None
    ]
    decoder_trainable = [
        (name, parameter)
        for name, parameter in candidate.decoder.named_parameters()
        if parameter.requires_grad
    ]
    decoder_grad_names = [name for name, parameter in decoder_trainable if parameter.grad is not None]
    if backbone_grad_names:
        raise RuntimeError(f"Frozen backbone received gradients: {backbone_grad_names[:5]}")
    if len(decoder_grad_names) != len(decoder_trainable):
        missing = sorted(set(name for name, _ in decoder_trainable) - set(decoder_grad_names))
        raise RuntimeError(f"Candidate decoder path has parameters without gradients: {missing}")

    total_parent, backbone_parent, decoder_parent = summarize_parameters(parent, parent_backbone)
    total_candidate, backbone_candidate, decoder_candidate = summarize_parameters(
        candidate, candidate_backbone
    )
    trainable_parent = sum(p.numel() for p in parent.parameters() if p.requires_grad)
    trainable_candidate = sum(p.numel() for p in candidate.parameters() if p.requires_grad)

    candidate_only_keys = sorted(set(candidate_state) - set(parent_state))
    parent_only_keys = sorted(set(parent_state) - set(candidate_state))
    disallowed_module_names = [
        name
        for name, module in candidate.decoder.named_modules()
        if isinstance(module, (nn.MultiheadAttention,))
        or module.__class__.__name__ in {"ResidualDepthwiseBlock", "FeatureInjector", "CrossAttention"}
    ]
    if disallowed_module_names:
        raise RuntimeError(f"Unexpected non-cascade modules found: {disallowed_module_names}")
    if _module_names(candidate.decoder, "ms_mlp"):
        raise RuntimeError("ChangeCascade unexpectedly contains an MS-MLP consumer")

    result = {
        "status": "PASS",
        "formal_training_started": False,
        "git_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True
        ).strip(),
        "parent_config": str(args.parent_config.resolve()),
        "candidate_config": str(args.config.resolve()),
        "parent_variant": parent_cfg.decoder_variant,
        "candidate_variant": candidate_cfg.decoder_variant,
        "device": str(device),
        "seed": seed,
        "protocol_config_diff_after_identity_fields": [],
        "shared_prefixes": list(shared_prefixes),
        "shared_initialization_keys": len(shared_keys),
        "shared_initialization_max_abs_diff": shared_init_diff,
        "backbone_frozen": all(not p.requires_grad for p in candidate_backbone.parameters()),
        "backbone_grad_tensors": len(backbone_grad_names),
        "intermediate_layers": ["L3", "L6", "L9", "L12"],
        "token_grid": [patch_h, patch_w],
        "pyramid_shapes_order_P2_P4_P8_P16": [_shape(value) for value in pyramid],
        "cascade_stage_shapes_order_P16_Y8_Y4_Y2": [_shape(value) for value in cascade_stages],
        "decoder_output_before_outer_dpt_resize": _shape(decoder_logits),
        "logits_shape": _shape(candidate_logits),
        "transposed_convolution_count": sum(
            isinstance(module, nn.ConvTranspose2d) for module in candidate.decoder.modules()
        ),
        "disallowed_modules": disallowed_module_names,
        "candidate_has_ms_mlp": bool(_module_names(candidate.decoder, "ms_mlp")),
        "candidate_only_state_keys": candidate_only_keys,
        "parent_only_state_keys": parent_only_keys,
        "parameters": {
            "parent_total": total_parent,
            "parent_backbone": backbone_parent,
            "parent_decoder": decoder_parent,
            "parent_trainable": trainable_parent,
            "candidate_total": total_candidate,
            "candidate_backbone": backbone_candidate,
            "candidate_decoder": decoder_candidate,
            "candidate_trainable": trainable_candidate,
            "candidate_minus_parent_total": total_candidate - total_parent,
            "candidate_minus_parent_trainable": trainable_candidate - trainable_parent,
        },
        "decoder_trainable_gradient_tensors": len(decoder_grad_names),
        "decoder_trainable_parameter_tensors": len(decoder_trainable),
        "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
        if device.type == "cuda"
        else None,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
