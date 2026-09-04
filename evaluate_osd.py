"""Explicit split evaluation for a frozen SegDINO-v2 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config_loader import ModelConfig
from osd_metrics import make_l4_payload, write_l4_payload
from runtime import build_model
from train_osd import (
    _input_size_metadata,
    _load_config,
    _make_loader,
    _normalize_input_size,
    _resolve_path,
    evaluate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    input_size = _normalize_input_size(config["input_size"])
    runtime = config.get("runtime", {})
    device = torch.device(args.device or runtime.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model_config = config["model"]
    model_cfg = ModelConfig(
        dino_size=model_config["dino_size"],
        dino_repo=str(_resolve_path(model_config["dino_repo"])),
        dino_ckpt=str(_resolve_path(model_config["dino_ckpt"])),
        decoder_dim=int(model_config["decoder_dim"]),
        use_bn=bool(model_config.get("use_bn", False)),
        num_classes=int(config["num_classes"]),
        patch_size=int(model_config.get("patch_size", 16)),
    )
    model, _ = build_model(model_cfg, str(device))
    if bool(model_config.get("freeze_backbone", True)):
        model.lock_backbone()
    checkpoint = torch.load(_resolve_path(args.checkpoint), map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device)

    workers = int(args.workers if args.workers is not None else config["training"].get("workers", 4))
    loader = _make_loader(
        _resolve_path(config["data_root"]),
        config.get(f"{args.split}_split", args.split),
        input_size,
        False,
        int(config["training"].get("eval_batch_size", 1)),
        workers,
        config,
        device,
    )
    metrics, confusion, evaluated_images = evaluate(
        model,
        loader,
        device,
        int(config["num_classes"]),
        int(config["ignore_index"]),
    )
    payload = make_l4_payload(
        confusion,
        split=args.split,
        input_size=_input_size_metadata(input_size),
        protocol_id=config.get("protocol_id", "OSD-EXP-v1.0/L4-global-512"),
        model_name=config.get("name"),
        evaluated_images=evaluated_images,
    )
    payload["checkpoint"] = str(_resolve_path(args.checkpoint))
    output = args.output or (_resolve_path(config.get("run_dir", "./runs/osd")) / f"{args.split}_final.json")
    write_l4_payload(payload, output)
    print(f"split={args.split} images={evaluated_images} output={output}")
    print(metrics)


if __name__ == "__main__":
    main()
