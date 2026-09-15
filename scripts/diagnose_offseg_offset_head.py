"""Diagnose the official OffSeg offset-learning head on a frozen OSD model.

This is deliberately not a new full decoder experiment.  It loads the
Val-best ``DINO-L12-SPM-CROSSMSEF-R-001`` checkpoint, replaces only its final
1x1 classifier with the official OffSeg ``Offset_Learning`` head, freezes the
entire preceding representation, and trains/evaluates the replacement head
on the existing OSD train/val split.  Test is never constructed or accessed.

The OffSeg implementation was inspected from the official repository at
commit ``a203f52fb66399517c49f5acda3aaf931804036e``.  The local head mirrors
its global class representation, class-offset projection, feature-offset
projection, coupled softmaxes, and class-dimension LayerNorm.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Direct script execution puts ``scripts/`` rather than the project root on
# sys.path.  Add the root before importing the local OSD modules.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import ModelConfig
from dpt import OfficialOffsetLearningHead
from osd_metrics import confusion_from_logits, l4_global_metrics, make_l4_payload, write_l4_payload
from runtime import build_model, summarize_parameters
from train_osd import (
    _load_config,
    _make_loader,
    _normalize_input_size,
    _resolve_path,
    _seed_everything,
    evaluate,
)


DEFAULT_CONFIG = ROOT_DIR / "configs" / "dino_l12_spm_cross_r_osd_512.json"
DEFAULT_CHECKPOINT = ROOT_DIR / "runs" / "dino_l12_spm_cross_r_osd_512_seed20260901" / "best.pth"
DEFAULT_RUN_DIR = ROOT_DIR / "work_dirs" / "offseg_offset_learning_probe"
OFFSEG_COMMIT = "a203f52fb66399517c49f5acda3aaf931804036e"


def _model_config(raw: dict[str, Any], num_classes: int) -> ModelConfig:
    model = raw["model"]
    return ModelConfig(
        dino_size=model["dino_size"],
        dino_repo=str(_resolve_path(model["dino_repo"])),
        dino_ckpt=str(_resolve_path(model["dino_ckpt"])),
        decoder_dim=int(model["decoder_dim"]),
        use_bn=bool(model.get("use_bn", False)),
        num_classes=num_classes,
        patch_size=int(model.get("patch_size", 16)),
        decoder_variant=str(model.get("decoder_variant", "l12_a_cross_r")),
        spatial_stride=int(model.get("spatial_stride", 4)),
        freeze_backbone=True,
        layer_mapping=model.get("layer_mapping"),
        adaptive_readout=bool(model.get("adaptive_readout", False)),
        readout_mode=str(model.get("readout_mode", "matrix")),
        readout_init=str(model.get("readout_init", "uniform")),
        readout_temperature=float(model.get("readout_temperature", 1.0)),
        wcf_enabled=bool(model.get("wcf_enabled", False)),
        wcf_reduction=int(model.get("wcf_reduction", 4)),
        wcf_alpha_init=float(model.get("wcf_alpha_init", 1e-2)),
    )


def _load_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path.resolve(), map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(state)!r}")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def _count(parameters) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _finite_gradients(parameters) -> bool:
    for parameter in parameters:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            return False
    return True


def _make_loaders(raw: dict[str, Any], device: torch.device, workers: int):
    input_size = _normalize_input_size(raw["input_size"])
    training = raw["training"]
    train_loader = _make_loader(
        _resolve_path(raw["data_root"]),
        raw.get("train_split", "train"),
        input_size,
        True,
        int(training["batch_size"]),
        workers,
        raw,
        device,
        persistent_workers=workers > 0,
    )
    val_loader = _make_loader(
        _resolve_path(raw["data_root"]),
        raw.get("val_split", "val"),
        input_size,
        False,
        int(training.get("eval_batch_size", 1)),
        workers,
        raw,
        device,
        persistent_workers=workers > 0,
    )
    return train_loader, val_loader


def _benchmark(
    model: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    device: torch.device,
) -> dict[str, float | int | None]:
    model.eval()
    head.train()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        criterion(logits, targets).backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        criterion(logits, targets).backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    seconds_per_step = elapsed / max(1, steps)
    return {
        "steps": int(steps),
        "batch_size": int(inputs.shape[0]),
        "elapsed_seconds": float(elapsed),
        "seconds_per_step": float(seconds_per_step),
        "estimated_5100_step_minutes": float(seconds_per_step * 5100.0 / 60.0),
        "peak_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
    }


def _preflight(
    model: nn.Module,
    head: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    parent_total: int,
    parent_decoder: int,
    parent_classifier: int,
    num_classes: int,
    run_dir: Path,
    benchmark_steps: int,
) -> dict[str, Any]:
    model.eval()
    head.train()
    inputs, targets, _ = next(iter(val_loader))
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    with torch.no_grad():
        logits = model(inputs)
    if tuple(logits.shape[-2:]) != tuple(inputs.shape[-2:]):
        raise RuntimeError(f"Output/input spatial mismatch: {tuple(logits.shape)} vs {tuple(inputs.shape)}")
    if logits.shape[1] != num_classes or not torch.isfinite(logits).all():
        raise RuntimeError(f"Invalid diagnostic logits: shape={tuple(logits.shape)}")

    head_ids = {id(parameter) for parameter in head.parameters()}
    leaked = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    if leaked:
        raise RuntimeError(f"Frozen parent leaked trainable parameters: {leaked[:8]}")

    optimizer.zero_grad(set_to_none=True)
    train_logits = model(inputs)
    probe_loss = criterion(train_logits, targets)
    probe_loss.backward()
    head_parameters = list(head.parameters())
    if not _finite_gradients(head_parameters):
        raise RuntimeError("OffSeg head preflight produced missing/non-finite gradients")
    optimizer.zero_grad(set_to_none=True)

    train_inputs, train_targets, _ = next(iter(train_loader))
    train_inputs = train_inputs.to(device, non_blocking=True)
    train_targets = train_targets.to(device, non_blocking=True)
    initial_head_state = {
        name: value.detach().clone()
        for name, value in head.state_dict().items()
    }
    benchmark = _benchmark(
        model,
        head,
        optimizer,
        criterion,
        train_inputs,
        train_targets,
        benchmark_steps,
        device,
    )
    # The benchmark is a timing-only probe.  Restore the exact random
    # initialization and clear AdamW moments before the actual diagnostic
    # training starts.
    head.load_state_dict(initial_head_state)
    optimizer.state.clear()
    optimizer.zero_grad(set_to_none=True)

    head_params = _count(head.parameters())
    parent_total_after = parent_total - parent_classifier + head_params
    parent_decoder_after = parent_decoder - parent_classifier + head_params
    result = {
        "preflight": "PASS",
        "test_accessed": False,
        "offseg_source_commit": OFFSEG_COMMIT,
        "device": str(device),
        "input_shape": list(inputs.shape),
        "logits_shape": list(logits.shape),
        "backbone_frozen": all(not parameter.requires_grad for parameter in model.backbone.parameters()),
        "parent_decoder_variant": str(getattr(model, "decoder_variant", "")),
        "parent_classifier_params": parent_classifier,
        "official_offset_head_params": head_params,
        "parent_total_params": parent_total,
        "diagnostic_total_params": parent_total_after,
        "parent_decoder_params": parent_decoder,
        "diagnostic_decoder_params": parent_decoder_after,
        "diagnostic_trainable_params": head_params,
        "added_params_vs_parent": head_params - parent_classifier,
        "probe_loss": float(probe_loss.detach().item()),
        "benchmark": benchmark,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "preflight.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _save_probe_checkpoint(path: Path, model: nn.Module, optimizer, step: int, epoch: int, metrics, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "config": raw,
            "offseg_source_commit": OFFSEG_COMMIT,
        },
        path,
    )


def train_probe(
    raw: dict[str, Any],
    model: nn.Module,
    head: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    run_dir: Path,
    max_iters: int,
    log_interval: int,
) -> dict[str, Any]:
    num_classes = int(raw["num_classes"])
    ignore_index = int(raw["ignore_index"])
    model.eval()
    head.train()
    history: list[dict[str, Any]] = []
    best_value = float("-inf")
    best_path = run_dir / "best.pth"
    latest_path = run_dir / "latest.pth"
    train_iter = iter(train_loader)
    running_loss = 0.0
    running_count = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, max_iters + 1):
        try:
            inputs, targets, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets, _ = next(train_iter)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        if tuple(logits.shape) != (inputs.shape[0], num_classes, inputs.shape[2], inputs.shape[3]):
            raise RuntimeError(f"Diagnostic output shape mismatch: {tuple(logits.shape)}")
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite diagnostic loss at step {step}: {loss.item()}")
        loss.backward()
        if not _finite_gradients(list(head.parameters())):
            raise RuntimeError(f"Non-finite/missing OffSeg head gradients at step {step}")
        optimizer.step()
        running_loss += float(loss.detach().item())
        running_count += 1

        if step == 1 or step % log_interval == 0:
            elapsed = time.perf_counter() - start
            eta = max(0.0, (max_iters - step) * elapsed / step)
            print(
                f"step={step}/{max_iters} loss={float(loss.detach().item()):.5f} "
                f"step_sec={elapsed / step:.3f} eta_min={eta / 60.0:.1f}",
                flush=True,
            )

        if step == max_iters or step % len(train_loader) == 0:
            epoch = int((step - 1) // len(train_loader) + 1)
            metrics, confusion, images = evaluate(
                model,
                val_loader,
                device,
                num_classes,
                ignore_index,
            )
            record = {
                "step": step,
                "epoch": epoch,
                "train_loss": running_loss / max(1, running_count),
                "selection_split": "val",
                "selection_metrics": metrics,
                "evaluated_images": images,
            }
            history.append(record)
            running_loss = 0.0
            running_count = 0
            model.eval()
            head.train()
            print(
                f"[Epoch {epoch:02d} | Step {step:04d}/{max_iters}] "
                f"Mean Train Loss={record['train_loss']:.5f} | "
                f"val_mIoU3={metrics['mIoU3_report_only_global']:.4f}% | "
                f"Oil={metrics['IoU_oil_global']:.2f}% | "
                f"Water={metrics['IoU_water_global']:.2f}% | "
                f"Others={metrics['IoU_others_global']:.2f}%",
                flush=True,
            )
            _save_probe_checkpoint(latest_path, model, optimizer, step, epoch, metrics, raw)
            write_l4_payload(
                make_l4_payload(
                    confusion,
                    split="val",
                    input_size=raw["input_size"],
                    protocol_id=raw.get("protocol_id", "OSD-EXP-v1.0/L4-global-512"),
                    model_name=raw.get("name"),
                    evaluated_images=images,
                    selection_split="val",
                ),
                run_dir / f"val_step_{step:06d}.json",
            )
            if np.isfinite(metrics["mIoU3_report_only_global"]) and metrics["mIoU3_report_only_global"] > best_value:
                best_value = metrics["mIoU3_report_only_global"]
                _save_probe_checkpoint(best_path, model, optimizer, step, epoch, metrics, raw)

    elapsed = time.perf_counter() - start
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    summary = {
        "run_dir": str(run_dir),
        "max_iters": max_iters,
        "elapsed_seconds": elapsed,
        "best_val_mIoU3": best_value,
        "selection_split": "val",
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "latest_checkpoint": str(latest_path) if latest_path.exists() else None,
        "test_accessed": False,
        "offseg_source_commit": OFFSEG_COMMIT,
        "params_total": sum(parameter.numel() for parameter in model.parameters()),
        "params_trainable": _count(parameter for parameter in model.parameters() if parameter.requires_grad),
        "peak_memory_mib": (
            round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
            if device.type == "cuda"
            else None
        ),
        "config": raw,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Training completed in {elapsed:.2f}s ({elapsed / 60.0:.2f}min). "
        f"Best val mIoU3={best_value:.4f}%",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--benchmark-steps", type=int, default=30)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    config_path = args.config.resolve()
    raw = _load_config(config_path)
    device = torch.device(args.device)
    runtime = raw.get("runtime", {})
    _seed_everything(
        int(runtime.get("seed", 20260901)),
        bool(runtime.get("deterministic", False)),
        bool(runtime.get("cudnn_benchmark", False)),
    )
    num_classes = int(raw["num_classes"])
    model_cfg = _model_config(raw, num_classes)
    if model_cfg.decoder_variant != "l12_a_cross_r":
        raise ValueError(
            "OffSeg diagnostic must use the fixed Cross-MSEF + original R parent; "
            f"got {model_cfg.decoder_variant!r}"
        )
    workers = int(raw["training"].get("workers", 4)) if args.workers is None else int(args.workers)
    train_loader, val_loader = _make_loaders(raw, device, workers)
    model, backbone = build_model(model_cfg, str(device))
    model.lock_backbone()
    _load_checkpoint(model, args.checkpoint)
    model.eval()

    parent_total, _, parent_decoder = summarize_parameters(model, backbone)
    old_classifier = model.decoder.out_conv
    parent_classifier = _count(old_classifier.parameters())
    head = OfficialOffsetLearningHead(
        num_classes=num_classes,
        embed_dims=int(raw["model"]["decoder_dim"]),
        init_std=0.02,
    ).to(device)
    model.decoder.out_conv = head
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head.parameters():
        parameter.requires_grad = True
    model.eval()
    head.train()

    criterion = nn.CrossEntropyLoss(ignore_index=int(raw["ignore_index"]))
    optimizer = torch.optim.AdamW(
        list(head.parameters()),
        lr=float(raw["optimizer"]["lr"]),
        betas=tuple(float(value) for value in raw["optimizer"].get("betas", [0.9, 0.999])),
        weight_decay=float(raw["optimizer"]["weight_decay"]),
    )
    run_dir = args.run_dir.resolve()
    preflight = _preflight(
        model,
        head,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        parent_total,
        parent_decoder,
        parent_classifier,
        num_classes,
        run_dir,
        int(args.benchmark_steps),
    )
    print(json.dumps(preflight, indent=2), flush=True)
    if args.preflight_only:
        return

    max_iters = int(args.max_iters) if args.max_iters is not None else int(args.epochs) * len(train_loader)
    train_probe(
        raw,
        model,
        head,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        run_dir,
        max_iters,
        max(1, int(args.log_interval)),
    )


if __name__ == "__main__":
    main()
