"""Sanity and one-batch gradient checks for DINO-LAYER-WCF-001."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import ModelConfig
from runtime import build_model
from train_osd import _load_config, _make_loader, _normalize_input_size, _resolve_path


def _model_config(config: dict) -> ModelConfig:
    raw = config["model"]
    return ModelConfig(
        dino_size=raw["dino_size"],
        dino_repo=str(_resolve_path(raw["dino_repo"])),
        dino_ckpt=str(_resolve_path(raw["dino_ckpt"])),
        decoder_dim=int(raw["decoder_dim"]),
        use_bn=bool(raw.get("use_bn", False)),
        num_classes=int(config["num_classes"]),
        patch_size=int(raw.get("patch_size", 16)),
        layer_mapping=raw.get("layer_mapping"),
        adaptive_readout=bool(raw.get("adaptive_readout", False)),
        readout_mode=str(raw.get("readout_mode", "matrix")),
        readout_init=str(raw.get("readout_init", "uniform")),
        readout_temperature=float(raw.get("readout_temperature", 1.0)),
        wcf_enabled=bool(raw.get("wcf_enabled", False)),
        wcf_reduction=int(raw.get("wcf_reduction", 4)),
        wcf_alpha_init=float(raw.get("wcf_alpha_init", 1e-2)),
    )


def _nonzero(value: torch.Tensor | None) -> bool:
    return value is not None and bool(torch.isfinite(value).all()) and float(value.detach().abs().max()) > 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = _load_config(args.config)
    if not bool(config["model"].get("wcf_enabled", False)):
        raise ValueError("wcf_sanity requires model.wcf_enabled=true")
    device = torch.device(args.device)
    model, backbone = build_model(_model_config(config), str(device))
    model.lock_backbone()
    blocks = list(model.get_wcf_blocks())
    if len(blocks) != 3:
        raise AssertionError(f"expected three independent WCF blocks, got {len(blocks)}")

    channels = int(model.backbone.embed_dim)
    tokens = 32 * 32
    fi = torch.randn(2, tokens, channels, device=device)
    fl = torch.randn(2, tokens, channels, device=device)
    random_checks = []
    for index, block in enumerate(blocks):
        result, gate = block(fi, fl, return_gate=True)
        projection_error = float((block.projection(fi) - fi).abs().max().detach().cpu())
        random_checks.append({
            "block": index + 3,
            "input_shape": list(fi.shape),
            "output_shape": list(result.shape),
            "gate_mean": float(gate.mean().detach().cpu()),
            "gate_min": float(gate.min().detach().cpu()),
            "gate_max": float(gate.max().detach().cpu()),
            "projection_max_abs_error": projection_error,
            "alpha": float(block.alpha.detach().cpu()),
        })
        if result.shape != fi.shape:
            raise AssertionError("WCF shape check failed")
        if not torch.allclose(gate, torch.full_like(gate, 0.5), atol=1e-7, rtol=0.0):
            raise AssertionError("zero-initialized WCF gate is not 0.5")
        if projection_error > 1e-6:
            raise AssertionError(f"identity projection check failed: {projection_error}")
        if abs(float(block.alpha.detach().cpu()) - 0.01) > 1e-8:
            raise AssertionError("alpha initialization is not 0.01")

    data_root = _resolve_path(config["data_root"])
    input_size = _normalize_input_size(config["input_size"])
    loader = _make_loader(
        data_root,
        config.get("train_split", "train"),
        input_size,
        True,
        int(config["training"]["batch_size"]),
        0,
        config,
        device,
    )
    inputs, targets, _ = next(iter(loader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=int(config["ignore_index"]))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    model.train()
    logits = model(inputs)
    loss = criterion(logits, targets)
    if not torch.isfinite(loss):
        raise AssertionError("smoke loss is not finite")
    loss.backward()
    first_backward = {
        "loss": float(loss.detach().cpu()),
        "channel_last_grad_nonzero": [_nonzero(block.channel_mlp[-1].weight.grad) for block in blocks],
        "token_last_grad_nonzero": [_nonzero(block.token_mlp[-1].weight.grad) for block in blocks],
        "projection_grad_nonzero": [_nonzero(block.projection.weight.grad) for block in blocks],
        "alpha_grad_nonzero": [_nonzero(block.alpha.grad) for block in blocks],
        "channel_first_grad_nonzero": [_nonzero(block.channel_mlp[0].weight.grad) for block in blocks],
        "token_first_grad_nonzero": [_nonzero(block.token_mlp[0].weight.grad) for block in blocks],
        "backbone_has_grad": any(parameter.grad is not None for parameter in backbone.parameters()),
    }
    if not all(first_backward["channel_last_grad_nonzero"] + first_backward["token_last_grad_nonzero"]):
        raise AssertionError("WCF gate output layers did not receive gradients")
    if not all(first_backward["projection_grad_nonzero"] + first_backward["alpha_grad_nonzero"]):
        raise AssertionError("WCF projection/alpha did not receive gradients")
    if first_backward["backbone_has_grad"]:
        raise AssertionError("frozen backbone received a gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    logits_second = model(inputs)
    loss_second = criterion(logits_second, targets)
    if not torch.isfinite(loss_second):
        raise AssertionError("second smoke loss is not finite")
    loss_second.backward()
    second_backward = {
        "loss": float(loss_second.detach().cpu()),
        "channel_first_grad_nonzero": [_nonzero(block.channel_mlp[0].weight.grad) for block in blocks],
        "token_first_grad_nonzero": [_nonzero(block.token_mlp[0].weight.grad) for block in blocks],
        "backbone_has_grad": any(parameter.grad is not None for parameter in backbone.parameters()),
    }
    if not all(second_backward["channel_first_grad_nonzero"] + second_backward["token_first_grad_nonzero"]):
        raise AssertionError("WCF gate first layers did not receive gradients after one optimizer step")
    if second_backward["backbone_has_grad"]:
        raise AssertionError("frozen backbone received a gradient on second smoke pass")

    decoder_params = sum(parameter.numel() for parameter in model.decoder.parameters())
    wcf_params = sum(parameter.numel() for block in blocks for parameter in block.parameters())
    result = {
        "status": "passed",
        "device": str(device),
        "random_checks": random_checks,
        "first_backward": first_backward,
        "second_backward": second_backward,
        "params": {
            "baseline_ctrl003_trainable": 3300364,
            "wcf_added": wcf_params,
            "new_total_trainable": sum(parameter.numel() for parameter in trainable),
            "decoder_total": decoder_params,
        },
    }
    output = args.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
