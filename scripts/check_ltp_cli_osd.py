#!/usr/bin/env python3
"""Preflight for DINO-LTP-CLI-001.

This script is intentionally data-free: it loads the local DINOv3 weights,
uses a random 512x512 tensor, and never builds an OSD loader or accesses a
split.  It checks the B1 identity path, the LTP/CLI tensor contracts, the
frozen-backbone boundary, and the initial gamma gradient before training is
allowed.
"""

from __future__ import annotations

import argparse
import gc
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
from ltp_cli import CrossLinearInteraction  # noqa: E402
from runtime import build_model, summarize_parameters  # noqa: E402


PARENT_CONFIG = ROOT_DIR / "configs/dino_tpa_base_msmlp_osd_512.json"
CANDIDATE_CONFIG = ROOT_DIR / "configs/dino_ltp_cli_osd_512.json"
EXPECTED_VARIANT = "ltp_cli"
EXPECTED_LAYERS = [2, 5, 8, 11]


def _resolve_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (ROOT_DIR / path).resolve())


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT_DIR, text=True).strip()


def _raw(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_config(raw: dict) -> ModelConfig:
    model = raw["model"]
    return ModelConfig(
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
        adaptive_readout=bool(model.get("adaptive_readout", False)),
        readout_mode=str(model.get("readout_mode", "matrix")),
        readout_init=str(model.get("readout_init", "uniform")),
        readout_temperature=float(model.get("readout_temperature", 1.0)),
        wcf_enabled=bool(model.get("wcf_enabled", False)),
        wcf_reduction=int(model.get("wcf_reduction", 4)),
        wcf_alpha_init=float(model.get("wcf_alpha_init", 1e-2)),
    )


def _shape(value: torch.Tensor) -> list[int]:
    return list(value.shape)


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _is_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def _load_shared_parent_state(parent, candidate) -> list[str]:
    parent_state = parent.state_dict()
    candidate_state = candidate.state_dict()
    common = {
        name: value
        for name, value in parent_state.items()
        if name in candidate_state and candidate_state[name].shape == value.shape
    }
    missing, unexpected = candidate.load_state_dict(common, strict=False)
    extra_prefixes = ("decoder.ltp.", "decoder.cli.")
    unexpected_allowed = not unexpected
    missing_allowed = all(name.startswith(extra_prefixes) for name in missing)
    if not unexpected_allowed or not missing_allowed:
        raise RuntimeError(
            "B1 shared initialization alignment failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return sorted(missing)


def _config_diff(parent: dict, candidate: dict) -> list[str]:
    """Return protocol differences after removing identity/metadata fields."""
    import copy

    left = copy.deepcopy(parent)
    right = copy.deepcopy(candidate)
    for raw in (left, right):
        raw.pop("name", None)
        raw.pop("run_dir", None)
        raw.setdefault("runtime", {}).pop("device", None)
        raw["model"].pop("decoder_variant", None)

    differences: list[str] = []

    def visit(a, b, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differences.append(f"{path}.{key}")
                else:
                    visit(a[key], b[key], f"{path}.{key}")
            return
        if a != b:
            differences.append(path)

    visit(left, right, "config")
    return differences


def _parameter_names(model) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _estimate_flops(img_size: int = 512) -> dict[str, int]:
    """Return a transparent multiply-add-to-FLOP estimate for one image."""
    s4 = (img_size // 4) ** 2
    s8 = (img_size // 8) ** 2
    s16 = (img_size // 16) ** 2

    def linear(tokens: int, in_dim: int, out_dim: int) -> int:
        return 2 * tokens * in_dim * out_dim

    def lite_block(tokens: int, channels: int, heads: int = 4) -> int:
        projections = linear(tokens, channels, channels) * 4
        ffn = linear(tokens, channels, 2 * channels) + linear(tokens, 2 * channels, channels)
        head_dim = channels // heads
        kernel = (
            2 * tokens * heads * head_dim * head_dim
            + 2 * tokens * heads * head_dim * head_dim
            + 2 * tokens * heads * head_dim
        )
        return projections + ffn + kernel

    patch_embed = 2 * s4 * 3 * 4 * 4 * 64
    merge_s8 = linear(s8, 4 * 64, 96)
    merge_s16 = linear(s16, 4 * 96, 128)
    ltp_blocks = lite_block(s4, 64) + lite_block(s8, 96) + lite_block(s16, 128)
    memory_projection = linear(s4, 64, 64) + linear(s8, 96, 64) + linear(s16, 128, 64)
    memory_tokens = s4 + s8 + s16
    cli_qo = 4 * (linear(s16, 256, 64) + linear(s16, 64, 256))
    cli_kv = linear(memory_tokens, 64, 64) * 2
    head_dim = 64 // 4
    cli_kernel = 4 * (
        2 * memory_tokens * 4 * head_dim * head_dim
        + 2 * s16 * 4 * head_dim * head_dim
        + 2 * s16 * 4 * head_dim
    )
    return {
        "ltp_cli_estimated_forward_flops": int(
            patch_embed
            + merge_s8
            + merge_s16
            + ltp_blocks
            + memory_projection
            + cli_qo
            + cli_kv
            + cli_kernel
        ),
        "memory_tokens": int(memory_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CANDIDATE_CONFIG)
    parser.add_argument("--parent-config", type=Path, default=PARENT_CONFIG)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    parent_raw = _raw(args.parent_config.resolve())
    candidate_raw = _raw(args.config.resolve())
    parent_cfg = _model_config(parent_raw)
    candidate_cfg = _model_config(candidate_raw)
    if candidate_cfg.decoder_variant != EXPECTED_VARIANT:
        raise ValueError(
            f"Expected decoder_variant={EXPECTED_VARIANT!r}, "
            f"got {candidate_cfg.decoder_variant!r}"
        )
    if candidate_cfg.num_classes != 4 or candidate_cfg.patch_size != 16:
        raise ValueError("DINO-LTP-CLI preflight requires four classes and patch_size=16")
    if candidate_cfg.layer_mapping is not None:
        raise ValueError("DINO-LTP-CLI must use native [L3,L6,L9,L12] routing")
    if _config_diff(parent_raw, candidate_raw):
        raise RuntimeError(
            "Candidate changes more than the decoder variant: "
            f"{_config_diff(parent_raw, candidate_raw)}"
        )
    if args.img_size % 16:
        raise ValueError("img-size must be divisible by 16")

    device = torch.device(
        args.device
        or candidate_raw.get("runtime", {}).get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")

    # The parent and candidate are loaded independently, then the complete B1
    # state is copied into the candidate.  This is an initialization audit,
    # not a trained-checkpoint warm start.
    torch.manual_seed(20260901)
    parent, parent_backbone = build_model(parent_cfg, str(device))
    parent.lock_backbone()
    parent_trainable = sum(
        parameter.numel() for parameter in parent.parameters() if parameter.requires_grad
    )
    torch.manual_seed(20260901)
    candidate, candidate_backbone = build_model(candidate_cfg, str(device))
    candidate.lock_backbone()
    missing_extra = _load_shared_parent_state(parent, candidate)

    if not all(not parameter.requires_grad for parameter in parent_backbone.parameters()):
        raise RuntimeError("B1 parent backbone is not fully frozen")
    if not all(not parameter.requires_grad for parameter in candidate_backbone.parameters()):
        raise RuntimeError("Candidate backbone is not fully frozen")
    if not isinstance(candidate.decoder.cli, CrossLinearInteraction):
        raise RuntimeError("Candidate decoder has no CrossLinearInteraction module")
    if any(float(gamma.detach().item()) != 0.0 for gamma in candidate.decoder.cli.gammas):
        raise RuntimeError("CLI gamma values are not zero-initialized")

    batch_size = 1
    patch_h = args.img_size // candidate_cfg.patch_size
    patch_w = args.img_size // candidate_cfg.patch_size
    x = torch.randn(batch_size, 3, args.img_size, args.img_size, device=device)
    parent.eval()
    candidate.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        parent_logits = parent(x)
        candidate_logits = candidate(x)
        features = candidate_backbone.get_intermediate_layers(
            x, n=EXPECTED_LAYERS, reshape=False, return_class_token=True, return_extra_tokens=True
        )
        patch_tokens = [item[0] if isinstance(item, (tuple, list)) else item for item in features]
        lowres_logits, trace = candidate.decoder.forward_with_diagnostics(
            patch_tokens, patch_h, patch_w, x
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - start

    logits_diff = _max_abs_diff(parent_logits, candidate_logits)
    if logits_diff > 1e-6:
        raise RuntimeError(f"gamma=0 parent equivalence failed: max_abs_diff={logits_diff}")

    expected = {
        "s4": (batch_size, 64, args.img_size // 4, args.img_size // 4),
        "s8": (batch_size, 96, args.img_size // 8, args.img_size // 8),
        "s16": (batch_size, 128, patch_h, patch_w),
        "memory": (batch_size, (args.img_size // 4) ** 2 + (args.img_size // 8) ** 2 + patch_h * patch_w, 64),
        "logits": (batch_size, candidate_cfg.num_classes, args.img_size, args.img_size),
    }
    actual = {name: tuple(trace[name].shape) for name in ("s4", "s8", "s16", "memory")}
    actual["logits"] = tuple(candidate_logits.shape)
    for name, shape in expected.items():
        if actual[name] != shape:
            raise RuntimeError(f"{name} shape mismatch: got {actual[name]}, expected {shape}")
    projected_shapes = [tuple(value.shape) for value in trace["projected"]]
    calibrated_shapes = [tuple(value.shape) for value in trace["calibrated"]]
    expected_x = [(batch_size, candidate_cfg.decoder_dim, patch_h, patch_w)] * 4
    if projected_shapes != expected_x or calibrated_shapes != expected_x:
        raise RuntimeError(
            f"Xi/Xi' shape mismatch: Xi={projected_shapes}, Xi'={calibrated_shapes}, expected={expected_x}"
        )
    pyramid_shapes = [tuple(value.shape) for value in trace["pyramid"]]
    expected_pyramid = [
        (batch_size, candidate_cfg.decoder_dim, 256, 256),
        (batch_size, candidate_cfg.decoder_dim, 128, 128),
        (batch_size, candidate_cfg.decoder_dim, 64, 64),
        (batch_size, candidate_cfg.decoder_dim, 32, 32),
    ]
    if pyramid_shapes != expected_pyramid:
        raise RuntimeError(f"B1 PR pyramid mismatch: got {pyramid_shapes}, expected {expected_pyramid}")
    if getattr(CrossLinearInteraction, "materializes_pairwise_attention", True):
        raise RuntimeError("CLI implementation declares pairwise attention materialization")

    # One real backward verifies that the zero gate can open.  The input and
    # target are synthetic; no dataset split is accessed.
    del parent_logits, candidate_logits, features, patch_tokens, lowres_logits, trace
    parent.cpu()
    parent_backbone.cpu()
    del parent, parent_backbone
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    start = time.perf_counter()
    logits = candidate(x)
    target = torch.zeros(batch_size, args.img_size, args.img_size, dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, target)
    if not _is_finite(logits) or not _is_finite(loss):
        raise RuntimeError("Candidate forward/loss is not finite")
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - start
    gamma_grad_norms = [
        None if gamma.grad is None else float(gamma.grad.detach().float().norm().item())
        for gamma in candidate.decoder.cli.gammas
    ]
    if any(value is None or not torch.isfinite(torch.tensor(value)) or value <= 0.0 for value in gamma_grad_norms):
        raise RuntimeError(f"Initial CLI gamma gradients are not finite/nonzero: {gamma_grad_norms}")
    nonfinite_grads = [
        name
        for name, parameter in candidate.named_parameters()
        if parameter.requires_grad and parameter.grad is not None and not _is_finite(parameter.grad)
    ]
    if nonfinite_grads:
        raise RuntimeError(f"Non-finite trainable gradients: {nonfinite_grads[:8]}")

    trainable_names = _parameter_names(candidate)
    candidate_total, candidate_backbone_params, candidate_non_backbone = summarize_parameters(
        candidate, candidate_backbone
    )
    candidate_trainable = sum(parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad)
    new_names = [name for name in trainable_names if name.startswith(("decoder.ltp.", "decoder.cli."))]
    new_trainable = sum(candidate.get_parameter(name).numel() for name in new_names)
    shared_trainable = candidate_trainable - new_trainable
    if shared_trainable != parent_trainable:
        raise RuntimeError(
            f"Shared B1 trainable count changed: candidate={shared_trainable}, "
            f"parent={parent_trainable}"
        )
    if new_trainable <= 0 or shared_trainable <= 0:
        raise RuntimeError("Candidate trainable-parameter partition is invalid")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in candidate.parameters() if parameter.requires_grad],
        lr=float(candidate_raw["optimizer"]["lr"]),
        betas=tuple(candidate_raw["optimizer"].get("betas", [0.9, 0.999])),
        weight_decay=float(candidate_raw["optimizer"]["weight_decay"]),
    )
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != {
        id(parameter) for parameter in candidate.parameters() if parameter.requires_grad
    }:
        raise RuntimeError("Optimizer parameter list is not exactly the trainable parameter set")

    peak_memory = None
    if device.type == "cuda":
        peak_memory = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)

    print(f"current_branch={_git('branch', '--show-current')}")
    print(f"current_commit={_git('rev-parse', 'HEAD')}")
    print(f"candidate_config={args.config.resolve()}")
    print(f"parent_config={args.parent_config.resolve()}")
    print(f"device={device}")
    print(f"backbone_frozen={all(not parameter.requires_grad for parameter in candidate_backbone.parameters())}")
    print(f"dino_patch_projection_frozen={not candidate_backbone.patch_embed.proj.weight.requires_grad}")
    print(f"native_layers={EXPECTED_LAYERS} (DINO blocks; semantic sources L3/L6/L9/L12)")
    print(f"b1_shared_state_missing_in_candidate={missing_extra}")
    print(f"patch_embed_new_trainable_shape={tuple(candidate.decoder.ltp.patch_embed.weight.shape)}")
    print(f"S4_shape={actual['s4']}")
    print(f"S8_shape={actual['s8']}")
    print(f"S16_shape={actual['s16']}")
    print(f"memory_shape={actual['memory']}")
    print(f"Xi_shapes={projected_shapes}")
    print(f"Xi_prime_shapes={calibrated_shapes}")
    print(f"B1_PR_pyramid_shapes={pyramid_shapes}")
    print(f"logits_shape={actual['logits']}")
    print("pairwise_attention_matrix_materialized=False")
    print(f"gamma_values={[float(value.detach().item()) for value in candidate.decoder.cli.gammas]}")
    print(f"gamma_grad_norms={gamma_grad_norms}")
    print(f"gamma_zero_parent_logits_max_abs_diff={logits_diff:.9g}")
    print(f"total_params={candidate_total:,}")
    print(f"backbone_params={candidate_backbone_params:,}")
    print(f"non_backbone_params={candidate_non_backbone:,}")
    print(f"trainable_params={candidate_trainable:,}")
    print(f"new_ltp_cli_trainable_params={new_trainable:,}")
    print(f"shared_b1_trainable_params={shared_trainable:,}")
    print(f"parent_b1_trainable_params={parent_trainable:,}")
    print(f"optimizer_param_groups={len(optimizer.param_groups)}")
    print(f"optimizer_parameter_names={json.dumps(trainable_names)}")
    print(f"forward_seconds={forward_seconds:.3f}")
    print(f"backward_seconds={backward_seconds:.3f}")
    print(f"peak_memory_mib={peak_memory}")
    print(f"flops_estimate={json.dumps(_estimate_flops(args.img_size), sort_keys=True)}")
    print("formal_training_started=False")
    print("sanity=PASS")


if __name__ == "__main__":
    main()
