"""Preflight the two decoder-side diagnostic candidates on OSD.

This script never trains and never touches the test split.  It loads the
Cross-MSEF+R Val-best checkpoint only to verify that the weighted-fusion
candidate is an exact parent at zero initialization and that the DySample
candidate changes only the three inter-scale samplers.
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
from runtime import build_model  # noqa: E402


PARENT_CONFIG = ROOT_DIR / "configs/dino_l12_spm_cross_r_osd_512.json"
PARENT_CHECKPOINT = ROOT_DIR / "runs/dino_l12_spm_cross_r_osd_512_seed20260901/best.pth"
CANDIDATE_CONFIGS = {
    "weighted": ROOT_DIR / "configs/dino_dec_fuse_weighted_osd_512.json",
    "dysample": ROOT_DIR / "configs/dino_dec_dysample_splus_osd_512.json",
}


def _model_config(raw):
    model = raw["model"]
    return ModelConfig(
        dino_size=model["dino_size"],
        dino_repo=str((ROOT_DIR / model["dino_repo"]).resolve()),
        dino_ckpt=str((ROOT_DIR / model["dino_ckpt"]).resolve()),
        decoder_dim=int(model["decoder_dim"]),
        use_bn=bool(model.get("use_bn", False)),
        num_classes=int(raw["num_classes"]),
        patch_size=int(model.get("patch_size", 16)),
        decoder_variant=str(model["decoder_variant"]),
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


def _load_state(model, checkpoint_path, allowed_missing):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = sorted(missing)
    unexpected = sorted(unexpected)
    if set(missing) != set(allowed_missing) or unexpected:
        raise RuntimeError(
            "checkpoint alignment mismatch: "
            f"missing={missing}, unexpected={unexpected}, expected_missing={sorted(allowed_missing)}"
        )
    return checkpoint


def _parameter_info(model, backbone):
    total = sum(p.numel() for p in model.parameters())
    backbone_total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "backbone": backbone_total,
        "decoder": total - backbone_total,
        "trainable": trainable,
    }


def _shape(value):
    return list(value.shape) if torch.is_tensor(value) else str(type(value))


def _capture_sampler_shapes(model):
    captured = {}
    handles = []
    for name in ("dysample_p8", "dysample_p4", "dysample_p2"):
        module = getattr(model.decoder, name)

        def hook(mod, inputs, output, name=name):
            captured[name] = {
                "input": _shape(inputs[0]),
                "output": _shape(output),
            }

        handles.append(module.register_forward_hook(hook))
    return captured, handles


def _check_finite_backward(model, device, candidate):
    model.train()
    inputs = torch.randn(1, 3, 512, 512, device=device)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    start = time.perf_counter()
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = logits.float().square().mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    finite = bool(torch.isfinite(logits).all() and torch.isfinite(loss))
    grad_norms = {}
    if candidate == "weighted":
        values = [model.decoder.fusion_logits.grad]
        names = ["fusion_logits"]
    else:
        values = [
            model.decoder.dysample_p8.offset.weight.grad,
            model.decoder.dysample_p4.offset.weight.grad,
            model.decoder.dysample_p2.offset.weight.grad,
        ]
        names = ["dysample_p8.offset", "dysample_p4.offset", "dysample_p2.offset"]
    for name, value in zip(names, values):
        grad_norms[name] = None if value is None else float(value.detach().norm().item())
        if value is None or not torch.isfinite(value).all():
            finite = False
    peak = (
        float(torch.cuda.max_memory_allocated(device) / 1024**2)
        if device.type == "cuda"
        else None
    )
    return {
        "finite": finite,
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu().item()),
        "grad_norms": grad_norms,
        "elapsed_seconds": elapsed,
        "peak_memory_mib": peak,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=tuple(CANDIDATE_CONFIGS), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    parent_raw = json.loads(PARENT_CONFIG.read_text(encoding="utf-8"))
    candidate_path = CANDIDATE_CONFIGS[args.candidate]
    candidate_raw = json.loads(candidate_path.read_text(encoding="utf-8"))

    parent, parent_backbone = build_model(_model_config(parent_raw), str(device))
    parent.lock_backbone()
    _load_state(parent, PARENT_CHECKPOINT, allowed_missing=[])
    parent.eval()

    candidate, candidate_backbone = build_model(_model_config(candidate_raw), str(device))
    candidate.lock_backbone()
    if args.candidate == "weighted":
        expected_missing = {"decoder.fusion_logits"}
    else:
        expected_missing = {
            f"decoder.{name}.{field}"
            for name in ("dysample_p8", "dysample_p4", "dysample_p2")
            for field in ("offset.weight", "offset.bias", "scope.weight", "init_pos")
        }
    _load_state(candidate, PARENT_CHECKPOINT, allowed_missing=expected_missing)
    candidate.eval()

    result = {
        "candidate": args.candidate,
        "device": str(device),
        "parent_config": str(PARENT_CONFIG),
        "candidate_config": str(candidate_path),
        "parent_checkpoint": str(PARENT_CHECKPOINT),
        "backbone_frozen": all(not p.requires_grad for p in candidate.backbone.parameters()),
        "patch_embed_frozen": all(
            not p.requires_grad for p in candidate.backbone.patch_embed.parameters()
        ),
        "parent_params": _parameter_info(parent, parent_backbone),
        "candidate_params": _parameter_info(candidate, candidate_backbone),
        "allowed_missing_on_parent_checkpoint": sorted(expected_missing),
    }
    result["trainable_delta_vs_parent"] = (
        result["candidate_params"]["trainable"] - result["parent_params"]["trainable"]
    )

    torch.manual_seed(20260901)
    inputs = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        parent_logits = parent(inputs)
        capture = {}
        handles = []
        if args.candidate == "dysample":
            capture, handles = _capture_sampler_shapes(candidate)
        candidate_logits = candidate(inputs)
        for handle in handles:
            handle.remove()
    result["logits_shape_parent"] = list(parent_logits.shape)
    result["logits_shape_candidate"] = list(candidate_logits.shape)
    result["parent_equivalence_max_abs_diff"] = float(
        (candidate_logits - parent_logits).abs().max().item()
    )
    if args.candidate == "weighted":
        result["fusion_weights_at_init"] = (
            (2.0 * torch.softmax(candidate.decoder.fusion_logits, dim=-1))
            .detach()
            .cpu()
            .tolist()
        )
    else:
        result["sampler_shapes"] = capture
        result["dysample_settings"] = {
            name: {
                "scale": int(getattr(candidate.decoder, name).scale),
                "style": getattr(candidate.decoder, name).style,
                "groups": int(getattr(candidate.decoder, name).groups),
                "dyscope": hasattr(getattr(candidate.decoder, name), "scope"),
            }
            for name in ("dysample_p8", "dysample_p4", "dysample_p2")
        }

    result["backward_check"] = _check_finite_backward(candidate, device, args.candidate)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
