#!/usr/bin/env python3
"""Preflight for DPA + neutral multi-scale MLP.

This script never builds an OSD data loader and never touches train/val/test
images.  It loads the completed TPA-base + MS-MLP checkpoint, aligns every
shared parameter into the DPA model, and checks that alpha=0 is an exact
parent-path identity before any formal training is allowed.
"""

from __future__ import annotations

import argparse
import copy
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


EXPECTED_PARENT_VARIANT = "tpa_ms_mlp"
EXPECTED_DPA_VARIANT = "dpa_ms_mlp"
EXPECTED_PARENT_CONFIG = ROOT_DIR / "configs/dino_tpa_base_msmlp_osd_512.json"
EXPECTED_PARENT_CHECKPOINT = (
    ROOT_DIR / "runs/dino_tpa_base_msmlp_osd_512_seed20260901/best.pth"
)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT_DIR, text=True).strip()


def _load_raw(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    return checkpoint


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _shape(value: torch.Tensor) -> list[int]:
    return list(value.shape)


def _stats(value: torch.Tensor) -> dict[str, float]:
    detached = value.detach().float()
    return {
        "mean": float(detached.mean().item()),
        "std": float(detached.std(unbiased=False).item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
    }


def _grad_norm(parameter: torch.nn.Parameter) -> float | None:
    if parameter.grad is None:
        return None
    return float(parameter.grad.detach().float().norm().item())


def _config_diff(parent: dict, dpa: dict) -> list[tuple[str, object, object]]:
    """Return differences after removing deliberate identity/variant metadata."""
    left = copy.deepcopy(parent)
    right = copy.deepcopy(dpa)
    for raw in (left, right):
        raw.pop("parent_experiment", None)
        raw.pop("name", None)
        raw.pop("run_dir", None)
    left["model"].pop("decoder_variant", None)
    right["model"].pop("decoder_variant", None)

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

    visit(left, right, "config")
    return differences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT_DIR / "configs/dino_dpa_base_msmlp_osd_512.json"
    )
    parser.add_argument("--parent-config", type=Path, default=EXPECTED_PARENT_CONFIG)
    parser.add_argument("--parent-checkpoint", type=Path, default=EXPECTED_PARENT_CHECKPOINT)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dpa_config_path = args.config.resolve()
    parent_config_path = args.parent_config.resolve()
    parent_checkpoint_path = args.parent_checkpoint.resolve()
    dpa_raw = _load_raw(dpa_config_path)
    parent_raw = _load_raw(parent_config_path)
    dpa_cfg = _model_config(dpa_raw)
    parent_cfg = _model_config(parent_raw)

    if parent_cfg.decoder_variant != EXPECTED_PARENT_VARIANT:
        raise RuntimeError(
            f"Parent decoder must be {EXPECTED_PARENT_VARIANT!r}, got {parent_cfg.decoder_variant!r}"
        )
    if dpa_cfg.decoder_variant != EXPECTED_DPA_VARIANT:
        raise RuntimeError(
            f"DPA decoder must be {EXPECTED_DPA_VARIANT!r}, got {dpa_cfg.decoder_variant!r}"
        )
    if dpa_cfg.layer_mapping is not None:
        raise RuntimeError("DPA preflight requires native layer_mapping=null")
    if not parent_checkpoint_path.is_file():
        raise FileNotFoundError(f"Parent checkpoint not found: {parent_checkpoint_path}")
    if args.img_size % dpa_cfg.patch_size:
        raise ValueError("img-size must be divisible by patch size")

    differences = _config_diff(parent_raw, dpa_raw)
    if differences:
        raise RuntimeError(f"Non-DPA config differences detected: {differences}")

    device = torch.device(
        args.device
        or dpa_raw.get("runtime", {}).get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    torch.manual_seed(int(dpa_raw.get("runtime", {}).get("seed", 20260901)))

    parent_model, parent_backbone = build_model(parent_cfg, str(device))
    dpa_model, dpa_backbone = build_model(dpa_cfg, str(device))
    parent_state = _checkpoint_state(parent_checkpoint_path)
    parent_missing, parent_unexpected = parent_model.load_state_dict(parent_state, strict=True)
    if parent_missing or parent_unexpected:
        raise RuntimeError(
            f"Parent checkpoint mismatch: missing={parent_missing}, unexpected={parent_unexpected}"
        )
    dpa_missing, dpa_unexpected = dpa_model.load_state_dict(parent_state, strict=False)
    expected_new_keys = {
        "decoder.phi_3.weight",
        "decoder.phi_6.weight",
        "decoder.phi_9.weight",
        "decoder.alpha_3",
        "decoder.alpha_6",
        "decoder.alpha_9",
    }
    if set(dpa_missing) != expected_new_keys or dpa_unexpected:
        raise RuntimeError(
            "DPA parent-weight alignment mismatch: "
            f"missing={dpa_missing}, unexpected={dpa_unexpected}"
        )

    parent_model.lock_backbone()
    dpa_model.lock_backbone()
    parent_model.eval()
    dpa_model.eval()
    dpa_decoder = dpa_model.decoder
    parent_decoder = parent_model.decoder

    parent_state_after = parent_model.state_dict()
    dpa_state_after = dpa_model.state_dict()
    shared_keys = sorted(set(parent_state_after) & set(dpa_state_after))
    shared_weight_max_abs_diff = max(
        _max_abs_diff(parent_state_after[key], dpa_state_after[key]) for key in shared_keys
    )

    batch_size = int(args.batch_size)
    patch_h = args.img_size // dpa_cfg.patch_size
    patch_w = args.img_size // dpa_cfg.patch_size
    x = torch.randn(batch_size, 3, args.img_size, args.img_size, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        features = parent_backbone.get_intermediate_layers(
            x, n=[2, 5, 8, 11]
        )
        parent_projected = parent_decoder._project_tokens(features, patch_h, patch_w)
        parent_pyramid = parent_decoder._build_tpa_pyramid(parent_projected)
        parent_lowres_logits = parent_decoder.ms_mlp(parent_pyramid)
        dpa_lowres_logits, trace = dpa_decoder.forward_with_diagnostics(
            features, patch_h, patch_w
        )
        parent_full_logits = parent_model(x)
        dpa_full_logits = dpa_model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - start

    patch_tokens = list(features)
    expected_patch_shape = (batch_size, patch_h * patch_w, parent_backbone.embed_dim)
    if any(tuple(item.shape) != expected_patch_shape for item in patch_tokens):
        raise RuntimeError(
            f"Patch-token shape mismatch: {[tuple(item.shape) for item in patch_tokens]} "
            f"vs expected {expected_patch_shape}"
        )

    calibrated = trace["calibrated"]
    scores = trace["scores"]
    aligned_l12 = trace["aligned_l12"]
    dpa_pyramid = trace["pyramid"]
    if not all(tuple(item.shape) == tuple(parent_projected[index].shape) for index, item in enumerate(calibrated)):
        raise RuntimeError("DPA calibrated branch shape mismatch")
    expected_score_shape = (batch_size, 1, patch_h, patch_w)
    if any(tuple(item.shape) != expected_score_shape for item in scores):
        raise RuntimeError(
            f"DPA score shape mismatch: {[tuple(item.shape) for item in scores]} "
            f"vs expected {expected_score_shape}"
        )
    expected_pyramid_shapes = [
        (batch_size, dpa_cfg.decoder_dim, size, size) for size in (256, 128, 64, 32)
    ]
    if [tuple(item.shape) for item in dpa_pyramid] != expected_pyramid_shapes:
        raise RuntimeError(
            f"DPA pyramid shape mismatch: {[tuple(item.shape) for item in dpa_pyramid]} "
            f"vs expected {expected_pyramid_shapes}"
        )
    expected_lowres_logits = (batch_size, dpa_cfg.num_classes, 256, 256)
    expected_full_logits = (batch_size, dpa_cfg.num_classes, args.img_size, args.img_size)
    if tuple(dpa_lowres_logits.shape) != expected_lowres_logits:
        raise RuntimeError(
            f"DPA low-resolution logits mismatch: {tuple(dpa_lowres_logits.shape)} "
            f"vs expected {expected_lowres_logits}"
        )
    if tuple(dpa_full_logits.shape) != expected_full_logits:
        raise RuntimeError(
            f"DPA full logits mismatch: {tuple(dpa_full_logits.shape)} vs expected {expected_full_logits}"
        )

    branch_input_diffs = [
        _max_abs_diff(parent_projected[index], calibrated[index]) for index in range(4)
    ]
    pyramid_diffs = [
        _max_abs_diff(parent_pyramid[index], dpa_pyramid[index]) for index in range(4)
    ]
    lowres_logits_diff = _max_abs_diff(parent_lowres_logits, dpa_lowres_logits)
    full_logits_diff = _max_abs_diff(parent_full_logits, dpa_full_logits)
    if max(branch_input_diffs + pyramid_diffs + [lowres_logits_diff, full_logits_diff]) > 1e-5:
        raise RuntimeError(
            "alpha=0 equivalence failed: "
            f"branch={branch_input_diffs}, pyramid={pyramid_diffs}, "
            f"lowres_logits={lowres_logits_diff}, full_logits={full_logits_diff}"
        )

    dpa_model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    probe_start = time.perf_counter()
    # ``inference_mode`` tensors from the equivalence pass cannot be saved by
    # autograd for a decoder weight-gradient probe.  Re-read the same frozen
    # backbone features under ordinary no_grad; this does not build a
    # backbone graph and leaves the model boundary unchanged.
    with torch.no_grad():
        probe_features = parent_backbone.get_intermediate_layers(
            x, n=[2, 5, 8, 11]
        )

    # Phase 1: the zero-init residual gate must leave the parent path intact,
    # while alpha itself must have a usable gradient to open the path.
    dpa_model.zero_grad(set_to_none=True)
    probe_logits = dpa_decoder(probe_features, patch_h, patch_w)
    probe_target = torch.zeros(
        batch_size, probe_logits.shape[-2], probe_logits.shape[-1], dtype=torch.long, device=device
    )
    probe_loss = F.cross_entropy(probe_logits, probe_target)
    probe_loss.backward()
    alpha_zero_grad_norms = {
        name: _grad_norm(getattr(dpa_decoder, name))
        for name in ("alpha_3", "alpha_6", "alpha_9")
    }
    phi_zero_grad_norms = {
        name: _grad_norm(getattr(dpa_decoder, name).weight)
        for name in ("phi_3", "phi_6", "phi_9")
    }
    alpha_zero_gradients = {
        name: float(getattr(dpa_decoder, name).grad.detach().item())
        for name in ("alpha_3", "alpha_6", "alpha_9")
    }
    alpha_names = ("alpha_3", "alpha_6", "alpha_9")
    phi_names = ("phi_3", "phi_6", "phi_9")
    if any(
        value is None or not torch.isfinite(torch.tensor(value)) or value <= 1e-12
        for value in alpha_zero_grad_norms.values()
    ):
        raise RuntimeError(
            f"DPA alpha=0 gate gradients are not finite/nonzero: {alpha_zero_grad_norms}"
        )
    if any(
        value is None or not torch.isfinite(torch.tensor(value)) or value > 1e-12
        for value in phi_zero_grad_norms.values()
    ):
        raise RuntimeError(
            f"DPA phi gradients should be zero at alpha=0: {phi_zero_grad_norms}"
        )

    # Phase 2: once alpha is infinitesimally opened, every phi branch must
    # receive a finite, nonzero gradient as well.
    with torch.no_grad():
        for name in alpha_names:
            getattr(dpa_decoder, name).fill_(1e-3)
    dpa_model.zero_grad(set_to_none=True)
    probe_logits_open = dpa_decoder(probe_features, patch_h, patch_w)
    probe_loss_open = F.cross_entropy(probe_logits_open, probe_target)
    probe_loss_open.backward()
    alpha_open_grad_norms = {
        name: _grad_norm(getattr(dpa_decoder, name)) for name in alpha_names
    }
    phi_open_grad_norms = {
        name: _grad_norm(getattr(dpa_decoder, name).weight) for name in phi_names
    }
    if any(
        value is None or not torch.isfinite(torch.tensor(value)) or value <= 1e-12
        for value in phi_open_grad_norms.values()
    ):
        raise RuntimeError(
            f"DPA phi gradients are not finite/nonzero at alpha=1e-3: {phi_open_grad_norms}"
        )
    if any(
        value is None or not torch.isfinite(torch.tensor(value))
        for value in alpha_open_grad_norms.values()
    ):
        raise RuntimeError(
            f"DPA alpha gradients are not finite at alpha=1e-3: {alpha_open_grad_norms}"
        )
    with torch.no_grad():
        for name in alpha_names:
            getattr(dpa_decoder, name).zero_()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    probe_seconds = time.perf_counter() - probe_start
    backbone_grad_tensors = sum(
        1 for parameter in dpa_backbone.parameters() if parameter.grad is not None
    )
    trainable_grad_tensors = sum(
        1
        for parameter in dpa_model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    dpa_model.eval()

    parent_total, parent_backbone_params, parent_decoder_params = summarize_parameters(
        parent_model, parent_backbone
    )
    dpa_total, dpa_backbone_params, dpa_decoder_params = summarize_parameters(
        dpa_model, dpa_backbone
    )
    parent_trainable = sum(
        parameter.numel() for parameter in parent_model.parameters() if parameter.requires_grad
    )
    dpa_trainable = sum(
        parameter.numel() for parameter in dpa_model.parameters() if parameter.requires_grad
    )
    expected_delta = 3 * dpa_cfg.decoder_dim * dpa_cfg.decoder_dim + 3
    actual_delta = dpa_total - parent_total
    actual_trainable_delta = dpa_trainable - parent_trainable
    if actual_delta != expected_delta or actual_trainable_delta != expected_delta:
        raise RuntimeError(
            f"Unexpected DPA parameter delta: total={actual_delta}, trainable={actual_trainable_delta}, "
            f"expected={expected_delta}"
        )

    peak_memory_mib = None
    if device.type == "cuda":
        peak_memory_mib = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)

    output = {
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "parent_config": str(parent_config_path),
        "parent_checkpoint": str(parent_checkpoint_path),
        "parent_checkpoint_alignment_missing_keys": sorted(dpa_missing),
        "non_dpa_config_differences": differences,
        "data_accessed": False,
        "test_accessed": False,
        "decoder_variant": dpa_cfg.decoder_variant,
        "parent_decoder_variant": parent_cfg.decoder_variant,
        "backbone_frozen": all(not p.requires_grad for p in dpa_backbone.parameters()),
        "native_layers": ["L3", "L6", "L9", "L12"],
        "patch_token_shapes": [_shape(item) for item in patch_tokens],
        "projected_shapes": [_shape(item) for item in trace["projected"]],
        "aligned_l12_shapes": [_shape(item) for item in aligned_l12],
        "score_shapes": [_shape(item) for item in scores],
        "score_stats": {name: _stats(score) for name, score in zip(("S3", "S6", "S9"), scores)},
        "calibrated_shapes": [_shape(item) for item in calibrated],
        "tpa_pyramid_shapes": [_shape(item) for item in dpa_pyramid],
        "lowres_logits_shape": _shape(dpa_lowres_logits),
        "logits_shape": _shape(dpa_full_logits),
        "alpha_init": {
            name: float(getattr(dpa_decoder, name).item())
            for name in ("alpha_3", "alpha_6", "alpha_9")
        },
        "alpha_probe_gradients": alpha_zero_gradients,
        "learnability": {
            "alpha_zero": {
                "alpha_grad_norms": alpha_zero_grad_norms,
                "phi_grad_norms": phi_zero_grad_norms,
            },
            "alpha_1e-3": {
                "alpha_grad_norms": alpha_open_grad_norms,
                "phi_grad_norms": phi_open_grad_norms,
            },
            "zero_tolerance": 1e-12,
            "sanity": "PASS",
        },
        "shared_weight_max_abs_diff": shared_weight_max_abs_diff,
        "alpha_zero_equivalence": {
            "branch_input_max_abs_diff": branch_input_diffs,
            "pyramid_max_abs_diff": pyramid_diffs,
            "lowres_logits_max_abs_diff": lowres_logits_diff,
            "full_logits_max_abs_diff": full_logits_diff,
            "tolerance": 1e-5,
        },
        "parent_params": {
            "total": parent_total,
            "backbone": parent_backbone_params,
            "decoder": parent_decoder_params,
            "trainable": parent_trainable,
        },
        "dpa_params": {
            "total": dpa_total,
            "backbone": dpa_backbone_params,
            "decoder": dpa_decoder_params,
            "trainable": dpa_trainable,
        },
        "new_parameter_names": sorted(
            key for key in dpa_state_after if key not in parent_state_after
        ),
        "parameter_delta": {
            "expected": expected_delta,
            "total": actual_delta,
            "trainable": actual_trainable_delta,
        },
        "backbone_grad_tensors": backbone_grad_tensors,
        "trainable_grad_tensors": trainable_grad_tensors,
        "inference_seconds": round(inference_seconds, 4),
        "train_probe_seconds": round(probe_seconds, 4),
        "peak_memory_mib": peak_memory_mib,
        "sanity": "PASS",
    }

    output_path = args.output
    if output_path is None:
        output_path = ROOT_DIR.parent / "work_dirs/dino_dpa_base_msmlp_osd_512_preflight.json"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"branch={output['branch']}")
    print(f"commit={output['commit']}")
    print(f"parent_decoder_variant={output['parent_decoder_variant']}")
    print(f"decoder_variant={output['decoder_variant']}")
    print(f"parent_checkpoint={output['parent_checkpoint']}")
    print(f"non_dpa_config_differences={output['non_dpa_config_differences']}")
    print(f"data_accessed={output['data_accessed']} test_accessed={output['test_accessed']}")
    print(f"backbone_frozen={output['backbone_frozen']}")
    print(f"native_layers={output['native_layers']}")
    print(f"patch_token_shapes={output['patch_token_shapes']}")
    print(f"projected_shapes={output['projected_shapes']}")
    print(f"aligned_l12_shapes={output['aligned_l12_shapes']}")
    print(f"score_shapes={output['score_shapes']}")
    print(f"score_stats={json.dumps(output['score_stats'], sort_keys=True)}")
    print(f"calibrated_shapes={output['calibrated_shapes']}")
    print(f"tpa_pyramid_shapes={output['tpa_pyramid_shapes']}")
    print(f"lowres_logits_shape={output['lowres_logits_shape']}")
    print(f"logits_shape={output['logits_shape']}")
    print(f"alpha_init={output['alpha_init']}")
    print(f"alpha_probe_gradients={output['alpha_probe_gradients']}")
    print(f"learnability={json.dumps(output['learnability'], sort_keys=True)}")
    print(f"shared_weight_max_abs_diff={output['shared_weight_max_abs_diff']}")
    print(f"alpha_zero_equivalence={json.dumps(output['alpha_zero_equivalence'], sort_keys=True)}")
    print(f"parent_params={output['parent_params']}")
    print(f"dpa_params={output['dpa_params']}")
    print(f"new_parameter_names={output['new_parameter_names']}")
    print(f"parameter_delta={output['parameter_delta']}")
    print(f"backbone_grad_tensors={output['backbone_grad_tensors']}")
    print(f"trainable_grad_tensors={output['trainable_grad_tensors']}")
    print(f"inference_seconds={output['inference_seconds']}")
    print(f"train_probe_seconds={output['train_probe_seconds']}")
    print(f"peak_memory_mib={output['peak_memory_mib']}")
    print(f"output={output_path}")
    print("sanity=PASS")


if __name__ == "__main__":
    main()
