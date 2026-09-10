"""Independent evaluation and gate-statistics report for WCF checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import ModelConfig
from osd_metrics import confusion_from_logits, l4_global_metrics, make_l4_payload, write_l4_payload
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = _load_config(args.config)
    raw_model = config["model"]
    if not bool(raw_model.get("wcf_enabled", False)):
        raise ValueError("evaluate_wcf requires model.wcf_enabled=true")
    device_name = args.device or config.get("runtime", {}).get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)
    model, _ = build_model(_model_config(config), str(device))
    if bool(raw_model.get("freeze_backbone", True)):
        model.lock_backbone()
    checkpoint = torch.load(_resolve_path(args.checkpoint), map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()

    input_size = _normalize_input_size(config["input_size"])
    loader = _make_loader(
        _resolve_path(config["data_root"]),
        config.get(f"{args.split}_split", args.split),
        input_size,
        False,
        int(config["training"].get("eval_batch_size", 1)),
        int(args.workers),
        config,
        device,
    )
    confusion = np.zeros((int(config["num_classes"]), int(config["num_classes"])), dtype=np.int64)
    gate_sum = np.zeros(3, dtype=np.float64)
    gate_sq_sum = np.zeros(3, dtype=np.float64)
    gate_count = np.zeros(3, dtype=np.int64)
    images = 0
    with torch.inference_mode():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits, _, gates = model(inputs, return_wcf_gates=True)
            if len(gates) != 3:
                raise AssertionError(f"expected three WCF gates, got {len(gates)}")
            confusion += confusion_from_logits(
                logits,
                targets,
                int(config["num_classes"]),
                int(config["ignore_index"]),
            )
            for index, gate in enumerate(gates):
                values = gate.detach().float().reshape(-1)
                gate_sum[index] += float(values.sum().cpu())
                gate_sq_sum[index] += float((values * values).sum().cpu())
                gate_count[index] += int(values.numel())
            images += int(inputs.shape[0])

    metrics = l4_global_metrics(confusion)
    gate_stats = {}
    for index, name in enumerate(("G3", "G6", "G9")):
        mean = gate_sum[index] / max(1, gate_count[index])
        variance = gate_sq_sum[index] / max(1, gate_count[index]) - mean * mean
        gate_stats[name] = {
            "mean": float(mean),
            "population_std": math.sqrt(max(0.0, variance)),
            "count": int(gate_count[index]),
        }
    payload = make_l4_payload(
        confusion,
        split=args.split,
        input_size=(input_size[0] if input_size[0] == input_size[1] else input_size),
        protocol_id=config.get("protocol_id", "OSD-EXP-v1.0/L4-global-512"),
        model_name=config.get("name"),
        evaluated_images=images,
    )
    payload["checkpoint"] = str(_resolve_path(args.checkpoint))
    payload["wcf_gate_stats"] = gate_stats
    payload["wcf_alpha"] = {
        f"alpha_{name}": float(block.alpha.detach().cpu())
        for name, block in zip(("3", "6", "9"), model.get_wcf_blocks())
    }
    payload["metrics_percent"] = metrics
    write_l4_payload(payload, args.output)
    print(json.dumps({"split": args.split, "images": images, "metrics": metrics, "wcf_gate_stats": gate_stats}, indent=2))


if __name__ == "__main__":
    main()
