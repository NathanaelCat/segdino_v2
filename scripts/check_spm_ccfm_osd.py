#!/usr/bin/env python3
"""Preflight for the RGB Lite-SPM + four-scale CCFM OSD variants.

This script intentionally uses random input only.  It does not construct an
OSD data loader, read a split, load a training checkpoint, or run an optimizer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import ModelConfig  # noqa: E402
from runtime import build_model, summarize_parameters  # noqa: E402
from train_osd import _load_config, _resolve_path  # noqa: E402


EXPECTED_VARIANTS = {"spm_ccfm_sad", "spm_ccfm_ms_mlp"}
EXPECTED_INPUTS = [
    ("D2", (32, 256, 256)),
    ("D4", (64, 128, 128)),
    ("D8", (128, 64, 64)),
    ("D16", (384, 32, 32)),
]
EXPECTED_OUTPUTS = [
    ("C2", (256, 256, 256)),
    ("C4", (256, 128, 128)),
    ("C8", (256, 64, 64)),
    ("C16", (256, 32, 32)),
]


def _shape(value: torch.Tensor) -> list[int]:
    return list(value.shape)


def _make_model_config(config: dict) -> ModelConfig:
    model = config["model"]
    return ModelConfig(
        dino_size=model["dino_size"],
        dino_repo=str(_resolve_path(model["dino_repo"])),
        dino_ckpt=str(_resolve_path(model["dino_ckpt"])),
        decoder_dim=int(model["decoder_dim"]),
        use_bn=bool(model.get("use_bn", False)),
        num_classes=int(config["num_classes"]),
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


def _assert_shapes(actual, expected, context):
    expected_shapes = [tuple((2, *shape)) for _, shape in expected]
    actual_shapes = [tuple(value.shape) for value in actual]
    if actual_shapes != expected_shapes:
        raise RuntimeError(
            f"{context} shape mismatch: got {actual_shapes}, expected {expected_shapes}"
        )


def _run_variant(config_path: Path, device: torch.device) -> dict:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = _make_model_config(raw)
    variant = model_config.decoder_variant
    if variant not in EXPECTED_VARIANTS:
        raise ValueError(f"Unexpected decoder variant: {variant}")
    if raw["input_size"] != 512 or raw["num_classes"] != 4:
        raise ValueError("SPM/CCFM preflight expects input_size=512 and num_classes=4")
    if model_config.layer_mapping is not None:
        raise ValueError("SPM/CCFM must use layer_mapping=null; its only DINO source is L12")
    if not model_config.freeze_backbone:
        raise ValueError("SPM/CCFM preflight requires freeze_backbone=true")

    model, backbone = build_model(model_config, str(device))
    model.lock_backbone()
    model.eval()

    backbone_trainable = [name for name, parameter in backbone.named_parameters() if parameter.requires_grad]
    patch_trainable = [
        name for name, parameter in backbone.named_parameters()
        if "patch_embed" in name and parameter.requires_grad
    ]
    if backbone_trainable or patch_trainable:
        raise RuntimeError(
            f"Frozen-boundary failure: backbone={backbone_trainable[:5]}, patch={patch_trainable[:5]}"
        )

    batch_size = 2
    x = torch.randn(batch_size, 3, 512, 512, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode():
        spatial_features = model.spm_stem(x)
        semantic_tokens = backbone.get_intermediate_layers(
            x,
            n=[model.intermediate_layer_idx[model.encoder_size][-1]],
        )[-1]
        ccfm_features = model.decoder.build_ccfm_features(
            semantic_tokens,
            spatial_features,
            32,
            32,
        )
        logits = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    actual_input = [*spatial_features, model.decoder._tokens_to_feature_map(semantic_tokens, 32, 32)]
    _assert_shapes(actual_input, EXPECTED_INPUTS, f"{variant} D2/D4/D8/D16")
    _assert_shapes(ccfm_features, EXPECTED_OUTPUTS, f"{variant} C2/C4/C8/C16")
    if tuple(logits.shape) != (batch_size, 4, 512, 512):
        raise RuntimeError(f"{variant} logits mismatch: got {tuple(logits.shape)}")

    decoder_names = {name for name, _ in model.decoder.named_modules()}
    if any(name.startswith("tpa_") for name in decoder_names):
        raise RuntimeError(f"{variant} unexpectedly retains a TPA/PR module")
    if any(name.startswith("encoder") or "query" in name.lower() for name in decoder_names):
        raise RuntimeError(f"{variant} unexpectedly contains a detection/query encoder component")
    sad_top_level = [
        name for name in decoder_names
        if name.startswith("sad_") and "." not in name
    ]
    if variant == "spm_ccfm_sad":
        if len(sad_top_level) != 8:
            raise RuntimeError(f"Expected eight original SAD blocks, got {len(sad_top_level)}")
    else:
        if not hasattr(model.decoder, "ms_mlp"):
            raise RuntimeError("MS-MLP backend is missing")

    total_params, backbone_params, non_backbone_params = summarize_parameters(model, backbone)
    decoder_params = sum(parameter.numel() for parameter in model.decoder.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    spm_params = sum(parameter.numel() for parameter in model.spm_stem.parameters())
    ccfm_params = sum(parameter.numel() for parameter in model.decoder.ccfm.parameters())
    backend_params = decoder_params - ccfm_params
    unexpected_trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not (name.startswith("spm_stem.") or name.startswith("decoder."))
    ]
    if unexpected_trainable:
        raise RuntimeError(
            "Unexpected trainable parameters outside SPM/CCFM/backend: "
            f"{unexpected_trainable[:8]}"
        )
    if trainable_params != spm_params + decoder_params:
        raise RuntimeError(
            "Trainable parameter accounting mismatch: "
            f"trainable={trainable_params}, spm+decoder={spm_params + decoder_params}"
        )
    peak_memory = None
    if device.type == "cuda":
        peak_memory = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)

    result = {
        "config": str(config_path),
        "model": raw["name"],
        "decoder_variant": variant,
        "checkpoint_load": None,
        "dino_source": "Frozen DINOv3-S/16 L12 only",
        "branch_order": ["D2", "D4", "D8", "D16"],
        "semantic_layer_index": model.intermediate_layer_idx[model.encoder_size][-1],
        "spatial_input_shapes": [_shape(value) for value in spatial_features],
        "l12_tokens_shape": _shape(semantic_tokens),
        "ccfm_input_shapes": [_shape(value) for value in actual_input],
        "ccfm_output_shapes": [_shape(value) for value in ccfm_features],
        "logits_shape": _shape(logits),
        "backbone_frozen": not backbone_trainable,
        "patch_embed_frozen": not patch_trainable,
        "tpa_pr_present": any(name.startswith("tpa_") for name in decoder_names),
        "sad_blocks": len(sad_top_level),
        "total_params": total_params,
        "backbone_params": backbone_params,
        "non_backbone_params": non_backbone_params,
        "spm_params": spm_params,
        "decoder_params": decoder_params,
        "ccfm_params": ccfm_params,
        "backend_params": backend_params,
        "trainable_params": trainable_params,
        "forward_seconds": round(elapsed, 3),
        "peak_allocated_mib": peak_memory,
    }
    del logits, ccfm_features, semantic_tokens, spatial_features, x, model, backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        nargs=2,
        default=[
            ROOT_DIR / "configs/dino_spm_ccfm_sad_osd_512.json",
            ROOT_DIR / "configs/dino_spm_ccfm_msmlp_osd_512.json",
        ],
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")

    results = [_run_variant(path.resolve(), device) for path in args.config]
    payload = {
        "preflight": "PASS",
        "training_started": False,
        "test_accessed": False,
        "device": str(device),
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
