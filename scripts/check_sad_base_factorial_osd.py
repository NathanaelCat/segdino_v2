#!/usr/bin/env python3
"""Preflight and common-initialization export for the B3/B4 SAD-Base pair.

B3 is TPA PR plus the minimal progressive top-down consumer.  B4 adds the
already validated DPA calibration path.  This script does not train and does
not access OSD images; it verifies the structural 2x2 comparison and writes a
fresh B3 model state that both formal runs can use as their common
initialization.
"""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import ModelConfig  # noqa: E402
from dpt import ResidualDepthwiseBlock  # noqa: E402
from runtime import build_model, summarize_parameters  # noqa: E402


B3_CONFIG = ROOT_DIR / "configs/dino_tpa_sad_base_osd_512.json"
B4_CONFIG = ROOT_DIR / "configs/dino_dpa_sad_base_osd_512.json"
DEFAULT_OUTPUT = ROOT_DIR / "work_dirs/dino_sad_base_factorial_preflight.json"
DEFAULT_INIT = ROOT_DIR / "work_dirs/dino_sad_base_factorial_common_init.pth"
EXPECTED_DPA_KEYS = {
    "decoder.phi_3.weight",
    "decoder.phi_6.weight",
    "decoder.phi_9.weight",
    "decoder.alpha_3",
    "decoder.alpha_6",
    "decoder.alpha_9",
}


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
    for key in ("name", "run_dir"):
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


def _shape(value: torch.Tensor) -> list[int]:
    return list(value.shape)


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--b3-config", type=Path, default=B3_CONFIG)
    parser.add_argument("--b4-config", type=Path, default=B4_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--common-init", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    b3_path = args.b3_config.resolve()
    b4_path = args.b4_config.resolve()
    output_path = args.output.resolve()
    init_path = args.common_init.resolve()
    b3_raw = _load_raw(b3_path)
    b4_raw = _load_raw(b4_path)
    b3_cfg = _model_config(b3_raw)
    b4_cfg = _model_config(b4_raw)

    if b3_cfg.decoder_variant != "tpa_sad_base":
        raise RuntimeError(f"B3 must use tpa_sad_base, got {b3_cfg.decoder_variant!r}")
    if b4_cfg.decoder_variant != "dpa_sad_base":
        raise RuntimeError(f"B4 must use dpa_sad_base, got {b4_cfg.decoder_variant!r}")
    if b3_cfg.layer_mapping is not None or b4_cfg.layer_mapping is not None:
        raise RuntimeError("B3/B4 must use native [L3,L6,L9,L12] routing")
    if b3_cfg.wcf_enabled or b4_cfg.wcf_enabled:
        raise RuntimeError("B3/B4 must not enable WCF")
    if b3_cfg.adaptive_readout or b4_cfg.adaptive_readout:
        raise RuntimeError("B3/B4 must not enable ALSR")
    differences = _config_differences(b3_raw, b4_raw)
    if differences:
        raise RuntimeError(f"B3/B4 have non-structural config differences: {differences}")
    if args.img_size % b3_cfg.patch_size:
        raise ValueError("img-size must be divisible by patch size")

    device = torch.device(args.device)
    seed = int(b3_raw.get("runtime", {}).get("seed", 20260901))
    _set_seed(seed)
    b3, b3_backbone = build_model(b3_cfg, str(device))
    b3.lock_backbone()

    # Reset the RNG before constructing B4.  Its DPA-only modules are created
    # after the complete B3-compatible prefix, so all shared initial weights
    # are reproducible.  The explicit state copy below makes this invariant
    # independent of any future constructor initialization detail.
    _set_seed(seed)
    b4, b4_backbone = build_model(b4_cfg, str(device))
    b4.lock_backbone()

    b3_state = b3.state_dict()
    missing, unexpected = b4.load_state_dict(b3_state, strict=False)
    if set(missing) != EXPECTED_DPA_KEYS or unexpected:
        raise RuntimeError(
            f"B4 shared-state alignment mismatch: missing={missing}, unexpected={unexpected}"
        )
    if set(b4.state_dict()) - set(b3_state) != EXPECTED_DPA_KEYS:
        raise RuntimeError("B4 contains parameters beyond the expected six DPA keys")
    with torch.no_grad():
        for name in ("alpha_3", "alpha_6", "alpha_9"):
            getattr(b4.decoder, name).zero_()

    b3.eval()
    b4.eval()
    inputs = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    with torch.inference_mode():
        b3_logits = b3(inputs)
        b4_logits = b4(inputs)

        features = b3_backbone.get_intermediate_layers(
            inputs, n=b3.intermediate_layer_idx[b3.encoder_size]
        )
        projected = b3.decoder._project_tokens(
            features, args.img_size // b3.patch_size, args.img_size // b3.patch_size
        )
        pyramid = b3.decoder._build_tpa_pyramid(projected)

    if b3_logits.shape != (args.batch_size, b3_cfg.num_classes, args.img_size, args.img_size):
        raise RuntimeError(f"B3 output shape mismatch: {tuple(b3_logits.shape)}")
    if b4_logits.shape != b3_logits.shape:
        raise RuntimeError(f"B4 output shape mismatch: {tuple(b4_logits.shape)}")
    if any(isinstance(module, ResidualDepthwiseBlock) for module in b3.decoder.modules()):
        raise RuntimeError("SAD-Base unexpectedly contains ResidualDepthwiseBlock")
    if any(isinstance(module, ResidualDepthwiseBlock) for module in b4.decoder.modules()):
        raise RuntimeError("DPA+SAD-Base unexpectedly contains ResidualDepthwiseBlock")

    total_b3, _, _ = summarize_parameters(b3, b3_backbone)
    total_b4, _, _ = summarize_parameters(b4, b4_backbone)
    trainable_b3 = sum(p.numel() for p in b3.parameters() if p.requires_grad)
    trainable_b4 = sum(p.numel() for p in b4.parameters() if p.requires_grad)
    total_delta = total_b4 - total_b3
    trainable_delta = trainable_b4 - trainable_b3
    if total_delta != 196611 or trainable_delta != 196611:
        raise RuntimeError(
            f"Unexpected B4-B3 parameter delta: total={total_delta}, trainable={trainable_delta}"
        )

    result = {
        "status": "PASS",
        "purpose": "B3/B4 decoder-side factorial preflight",
        "b3": {"config": str(b3_path), "decoder_variant": b3_cfg.decoder_variant},
        "b4": {"config": str(b4_path), "decoder_variant": b4_cfg.decoder_variant},
        "seed": seed,
        "device": str(device),
        "common_initialization": {
            "source": str(init_path),
            "state_alignment": "B4 receives B3 state; only DPA keys are missing",
            "missing_dpa_keys": sorted(EXPECTED_DPA_KEYS),
            "unexpected_keys": [],
        },
        "backbone_frozen": {
            "b3": all(not p.requires_grad for p in b3_backbone.parameters()),
            "b4": all(not p.requires_grad for p in b4_backbone.parameters()),
        },
        "pyramid_shapes": [_shape(value) for value in pyramid],
        "lowres_logits_shape": [args.batch_size, b3_cfg.num_classes, args.img_size // 2, args.img_size // 2],
        "b3_logits_shape": _shape(b3_logits),
        "b4_logits_shape": _shape(b4_logits),
        "alpha_zero_logits_max_abs_diff": _max_abs_diff(b3_logits, b4_logits),
        "sad_base_has_refinement_blocks": False,
        "parameters": {
            "b3_total": total_b3,
            "b3_trainable": trainable_b3,
            "b4_total": total_b4,
            "b4_trainable": trainable_b4,
            "b4_minus_b3_total": total_delta,
            "b4_minus_b3_trainable": trainable_delta,
            "expected_dpa_delta": 196611,
        },
        "config_differences_allowed": [
            "name",
            "run_dir",
            "runtime.device",
            "model.decoder_variant",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    init_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": b3_state,
            "config": b3_raw,
            "purpose": "common initialization for B3/B4 SAD-Base factorial",
            "seed": seed,
        },
        init_path,
    )
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"common_init={init_path}")
    print(f"preflight_output={output_path}")


if __name__ == "__main__":
    main()
