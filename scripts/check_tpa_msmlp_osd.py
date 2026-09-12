#!/usr/bin/env python3
"""Sanity check for the TPA-base plus neutral multi-scale MLP path.

No OSD data or test split is touched.  The check verifies the native DINO
tokens, the four TPA spatial scales, the absence of SAD modules, the final
logit shape, and the frozen-backbone gradient boundary.
"""

from __future__ import annotations

import argparse
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
from runtime import build_model, summarize_parameters  # noqa: E402


def _resolve_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return str(path)


def _load_config(path: Path) -> tuple[dict, ModelConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    model = raw["model"]
    return raw, ModelConfig(
        dino_size=model["dino_size"],
        dino_repo=_resolve_path(model["dino_repo"]),
        dino_ckpt=_resolve_path(model["dino_ckpt"]),
        decoder_dim=int(model["decoder_dim"]),
        use_bn=bool(model.get("use_bn", False)),
        num_classes=int(raw["num_classes"]),
        patch_size=int(model.get("patch_size", 16)),
        decoder_variant=str(model.get("decoder_variant", "tpa_sad")),
        spatial_stride=int(model.get("spatial_stride", 4)),
        freeze_backbone=bool(model.get("freeze_backbone", True)),
        layer_mapping=model.get("layer_mapping"),
    )


def _shape(value: torch.Tensor) -> list[int]:
    return list(value.shape)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT_DIR, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "configs/dino_tpa_base_msmlp_osd_512.json",
    )
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    raw, model_config = _load_config(args.config.resolve())
    if model_config.decoder_variant != "tpa_ms_mlp":
        raise ValueError(
            f"Expected decoder_variant=tpa_ms_mlp, got {model_config.decoder_variant!r}"
        )
    if model_config.num_classes != 4:
        raise ValueError(f"Expected four OSD classes, got {model_config.num_classes}")
    if model_config.patch_size != 16:
        raise ValueError(f"Expected DINO patch size 16, got {model_config.patch_size}")
    if model_config.layer_mapping is not None:
        raise ValueError("TPA-base sanity expects native layer routing (layer_mapping=null)")
    if args.img_size % model_config.patch_size:
        raise ValueError("img-size must be divisible by patch size")

    device = torch.device(
        args.device
        or raw.get("runtime", {}).get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    model, backbone = build_model(model_config, str(device))
    model.lock_backbone()
    model.train()

    batch_size = 2
    patch_h = args.img_size // model_config.patch_size
    patch_w = args.img_size // model_config.patch_size
    x = torch.randn(batch_size, 3, args.img_size, args.img_size, device=device)
    decoder = model.decoder
    if any(name.startswith("sad_") for name, _ in decoder.named_modules()):
        raise RuntimeError("tpa_ms_mlp must not contain SAD modules")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        intermediate = backbone.get_intermediate_layers(
            x,
            n=[2, 5, 8, 11],
            reshape=False,
            return_class_token=True,
            return_extra_tokens=True,
        )
        logits_eval = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    eval_seconds = time.perf_counter() - start

    patch_tokens = [item[0] for item in intermediate]
    expected_patch = (batch_size, patch_h * patch_w, backbone.embed_dim)
    for tokens in patch_tokens:
        if tuple(tokens.shape) != expected_patch:
            raise RuntimeError(
                f"Patch token shape mismatch: got {tuple(tokens.shape)}, expected {expected_patch}"
            )

    with torch.inference_mode():
        projected = [
            projection(decoder._tokens_to_feature_map(tokens, patch_h, patch_w))
            for projection, tokens in zip(decoder.token_projections, patch_tokens)
        ]
        pyramid = (
            decoder.tpa_branch_1(projected[0]),
            decoder.tpa_branch_2(projected[1]),
            decoder.tpa_branch_3(projected[2]),
            decoder.tpa_branch_4(projected[3]),
        )
        target_size = pyramid[0].shape[-2:]
        aligned = tuple(
            feature
            if feature.shape[-2:] == target_size
            else F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            for feature in pyramid
        )

    expected_logits = (batch_size, model_config.num_classes, args.img_size, args.img_size)
    if tuple(logits_eval.shape) != expected_logits:
        raise RuntimeError(
            f"Logits shape mismatch: got {tuple(logits_eval.shape)}, expected {expected_logits}"
        )
    expected_pyramid = [(batch_size, model_config.decoder_dim, size, size) for size in (256, 128, 64, 32)]
    if [tuple(feature.shape) for feature in pyramid] != expected_pyramid:
        raise RuntimeError(
            f"TPA pyramid shape mismatch: got {[tuple(feature.shape) for feature in pyramid]}, "
            f"expected {expected_pyramid}"
        )
    if any(tuple(feature.shape) != (batch_size, model_config.decoder_dim, 256, 256) for feature in aligned):
        raise RuntimeError(f"MS-MLP alignment mismatch: {[tuple(feature.shape) for feature in aligned]}")

    model.train()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    logits = model(x)
    loss = F.cross_entropy(
        logits,
        torch.zeros(batch_size, args.img_size, args.img_size, dtype=torch.long, device=device),
    )
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_probe_seconds = time.perf_counter() - start

    backbone_grad_names = [
        name for name, parameter in backbone.named_parameters() if parameter.grad is not None
    ]
    decoder_grad_count = sum(
        1
        for parameter in model.decoder.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    decoder_param_count = sum(
        parameter.numel() for parameter in model.decoder.parameters() if parameter.requires_grad
    )
    total_params, backbone_params, non_backbone_params = summarize_parameters(model, backbone)
    if backbone_grad_names:
        raise RuntimeError(f"Frozen backbone received gradients: {backbone_grad_names[:5]}")
    if decoder_grad_count == 0:
        raise RuntimeError("No decoder parameter received a gradient")

    memory = {"peak_allocated_mib": None}
    if device.type == "cuda":
        memory["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)

    print(f"branch={_git('branch', '--show-current')}")
    print(f"commit={_git('rev-parse', 'HEAD')}")
    print(f"decoder_variant={model_config.decoder_variant}")
    print(f"backbone_frozen={all(not p.requires_grad for p in backbone.parameters())}")
    print("native_layers=[L3,L6,L9,L12]")
    print(f"patch_token_shapes={[ _shape(t) for t in patch_tokens ]}")
    print(f"tpa_pyramid_shapes={[ _shape(t) for t in pyramid ]}")
    print(f"ms_mlp_aligned_shapes={[ _shape(t) for t in aligned ]}")
    print(f"sad_modules_present={any(name.startswith('sad_') for name, _ in decoder.named_modules())}")
    print(f"logits_shape={_shape(logits_eval)}")
    print(f"total_params={total_params}")
    print(f"backbone_params={backbone_params}")
    print(f"decoder_params={non_backbone_params}")
    print(f"trainable_params={decoder_param_count}")
    print(f"decoder_grad_tensors={decoder_grad_count}")
    print(f"eval_seconds={eval_seconds:.3f}")
    print(f"train_probe_seconds={train_probe_seconds:.3f}")
    print(f"memory={json.dumps(memory, sort_keys=True)}")
    print("sanity=PASS")


if __name__ == "__main__":
    main()
