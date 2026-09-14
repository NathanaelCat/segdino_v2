"""Train/evaluate the current SegDINO-v2 decoder on OSD.

This runner is intentionally separate from the upstream binary medical-data
runner.  It has an explicit four-class CE objective and an explicit OSD
L4-global evaluator so that input/training profiles can change without
changing the cross-model metric.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config_loader import ModelConfig
from osd_dataset import OSDDataset, build_osd_transform
from osd_metrics import confusion_from_logits, l4_global_metrics, make_l4_payload, write_l4_payload
from runtime import build_model, summarize_parameters


ROOT_DIR = Path(__file__).resolve().parent


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_input_size(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        size = (value, value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        size = (int(value[0]), int(value[1]))
    else:
        raise ValueError("input_size must be an integer or [height, width]")
    if min(size) <= 0:
        raise ValueError(f"Invalid input_size: {value}")
    return size


def _input_size_metadata(size: tuple[int, int]) -> int | list[int]:
    return size[0] if size[0] == size[1] else [size[0], size[1]]


def _summarize_dpa_trace(model: nn.Module, trace: dict[str, Any]) -> dict[str, Any]:
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        raise RuntimeError("DPA diagnostics require a model decoder")
    score_names = ("S3", "S6", "S9")
    scores = trace.get("scores")
    if scores is None or len(scores) != len(score_names):
        raise RuntimeError("DPA diagnostics did not return S3/S6/S9 scores")

    score_stats: dict[str, dict[str, float]] = {}
    for name, score in zip(score_names, scores):
        value = score.detach().float()
        score_stats[name] = {
            "mean": float(value.mean().item()),
            "std": float(value.std(unbiased=False).item()),
            "min": float(value.min().item()),
            "max": float(value.max().item()),
        }
    alpha = {
        name: float(getattr(decoder, name).detach().float().item())
        for name in ("alpha_3", "alpha_6", "alpha_9")
    }
    return {"alpha": alpha, "scores": score_stats}


def _summarize_patch_dpa_trace(model: nn.Module, trace: dict[str, Any]) -> dict[str, Any]:
    """Summarize the four-depth patch-guided DPA path without saving maps."""
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        raise RuntimeError("Patch-DPA diagnostics require a model decoder")
    depth_names = ("3", "6", "9", "12")
    scores = trace.get("scores")
    cosines = trace.get("cosines")
    projected = trace.get("projected")
    residuals = trace.get("residuals")
    expected_length = len(depth_names)
    if any(
        values is None or len(values) != expected_length
        for values in (scores, cosines, projected, residuals)
    ):
        raise RuntimeError(
            "Patch-DPA diagnostics must return four scores, cosines, projected maps, "
            "and residual maps"
        )

    def spatial_stats(value: torch.Tensor) -> dict[str, float]:
        value = value.detach().float()
        return {
            "mean": float(value.mean().item()),
            "std": float(value.std(unbiased=False).item()),
            "min": float(value.min().item()),
            "max": float(value.max().item()),
        }

    alpha = {
        f"alpha_{name}": float(getattr(decoder, f"alpha_{name}").detach().float().item())
        for name in depth_names
    }
    score_stats = {
        f"S{name}": spatial_stats(value)
        for name, value in zip(depth_names, scores)
    }
    cosine_stats = {
        f"cos_{name}": spatial_stats(value)
        for name, value in zip(depth_names, cosines)
    }
    residual_ratios = {}
    for name, feature, residual in zip(depth_names, projected, residuals):
        feature_flat = feature.detach().float().flatten(1)
        residual_flat = residual.detach().float().flatten(1)
        ratio = residual_flat.norm(dim=1) / feature_flat.norm(dim=1).clamp_min(1e-12)
        residual_ratios[f"R{name}"] = spatial_stats(ratio)
    return {
        "alpha": alpha,
        "scores": score_stats,
        "cosines": cosine_stats,
        "residual_ratios": residual_ratios,
    }


def _summarize_patch_sad_trace(model: nn.Module, trace: dict[str, Any]) -> dict[str, Any]:
    """Summarize the four scale-aligned PatchGuidedR SAD stages."""
    decoder = getattr(model, "decoder", None)
    if decoder is None or not hasattr(decoder, "named_patch_blocks"):
        raise RuntimeError("Patch-guided SAD diagnostics require named patch blocks")
    names = ("P16", "P8", "P4", "P2")
    scores = trace.get("scores")
    features = trace.get("projected")
    residuals = trace.get("residuals")
    if any(
        values is None or len(values) != len(names)
        for values in (scores, features, residuals)
    ):
        raise RuntimeError(
            "Patch-guided SAD diagnostics must return four scale scores, inputs, "
            "and residuals"
        )

    def spatial_stats(value: torch.Tensor) -> dict[str, float]:
        value = value.detach().float()
        return {
            "mean": float(value.mean().item()),
            "std": float(value.std(unbiased=False).item()),
            "min": float(value.min().item()),
            "max": float(value.max().item()),
        }

    residual_ratios = {}
    for name, feature, residual in zip(names, features, residuals):
        feature_flat = feature.detach().float().flatten(1)
        residual_flat = residual.detach().float().flatten(1)
        ratio = residual_flat.norm(dim=1) / feature_flat.norm(dim=1).clamp_min(1e-12)
        residual_ratios[f"R_{name}"] = spatial_stats(ratio)
    return {
        "alpha": {
            name: float(module.alpha.detach().float().item())
            for name, module in decoder.named_patch_blocks()
        },
        "scores": {
            f"S_{name}": spatial_stats(value)
            for name, value in zip(names, scores)
        },
        "cosines": {
            f"cos_{name}": spatial_stats(value)
            for name, value in zip(names, trace["cosines"])
        },
        "residual_ratios": residual_ratios,
        "native_patch_shape": list(trace["native_patch"].shape),
        "patch_pyramid_shapes": [list(value.shape) for value in trace["patch_priors"]],
    }


def _summarize_ltp_cli_gammas(model: nn.Module) -> dict[str, float]:
    """Read the four CLI residual scales without changing the forward path."""
    decoder = getattr(model, "decoder", None)
    cli = getattr(decoder, "cli", None)
    gammas = getattr(cli, "gammas", None)
    if gammas is None or len(gammas) != 4:
        raise RuntimeError("LTP-CLI gamma logging requires four CLI gamma parameters")
    names = ("gamma_3", "gamma_6", "gamma_9", "gamma_12")
    return {
        name: float(value.detach().float().item())
        for name, value in zip(names, gammas)
    }


def _spsr_learning_alive(event: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Apply the explicit early gate to the SPSR learning monitor.

    ``gamma`` is zero-initialized, so the first backward is expected to open
    only gamma.  By the gate step, at least one of the offset or compatibility
    statistics must have moved beyond numerical initialization and their
    gradients must remain finite/non-zero.  This is a conservative diagnostic
    gate, not a performance-based early-stopping rule.
    """
    gammas = [float(value) for value in event.get("gamma_after_step", [])]
    scales = event.get("scales", [])
    gradients = event.get("gradients_before_step", {})
    finite_gradients = all(np.isfinite(float(value)) for value in gradients.values())
    max_grad = max((abs(float(value)) for value in gradients.values()), default=0.0)
    max_gamma = max((abs(value) for value in gammas), default=0.0)
    max_offset = max(
        (abs(float(scale.get("offset_abs_max_source_tokens", 0.0))) for scale in scales),
        default=0.0,
    )
    max_weight_delta = max(
        (
            abs(float(scale.get("weight_abs_deviation_from_uniform_mean", 0.0)))
            for scale in scales
        ),
        default=0.0,
    )
    checks = {
        "gamma_moved": max_gamma > 1e-7,
        "offset_or_weight_moved": max_offset > 1e-6 or max_weight_delta > 1e-7,
        "branch_gradients_finite": finite_gradients,
        "branch_gradient_nonzero": max_grad > 0.0,
    }
    return all(checks.values()), {
        "checks": checks,
        "max_abs_gamma_after_step": max_gamma,
        "max_abs_offset_source_tokens": max_offset,
        "max_weight_deviation_from_uniform_mean": max_weight_delta,
        "max_branch_gradient_norm": max_grad,
    }


def _seed_everything(seed: int, deterministic: bool, cudnn_benchmark: bool) -> None:
    if deterministic and cudnn_benchmark:
        raise ValueError("deterministic=True and cudnn_benchmark=True are contradictory")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def _make_loader(
    data_root: Path,
    split: str,
    input_size: tuple[int, int],
    train: bool,
    batch_size: int,
    workers: int,
    config: dict[str, Any],
    device: torch.device,
    persistent_workers: bool = False,
) -> DataLoader:
    transform = build_osd_transform(
        input_size,
        train=train,
        augment=config.get("augmentation", {}) if train else None,
        resize_mode=config.get("preprocessing", {}).get("resize_mode", "stretch"),
        ignore_index=int(config["ignore_index"]),
    )
    dataset = OSDDataset(
        data_root,
        split=split,
        transform=transform,
        num_classes=int(config["num_classes"]),
        ignore_index=int(config["ignore_index"]),
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": train,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        # Do not fork workers after a CUDA model has been constructed.  A
        # forked CUDA context can corrupt memory statistics and is unsafe for
        # later CUDA work; spawn gives each worker a clean interpreter.
        loader_kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **loader_kwargs)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    max_batches: Optional[int] = None,
    collect_dpa_stats: bool = False,
) -> tuple[dict[str, float], np.ndarray, int] | tuple[dict[str, float], np.ndarray, int, dict[str, Any]]:
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    images = 0
    dpa_stats = None
    for batch_index, (inputs, targets, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs_device = inputs.to(device, non_blocking=True)
        if collect_dpa_stats and batch_index == 0:
            diagnostics_fn = getattr(model, "dpa_diagnostics", None)
            if diagnostics_fn is None:
                raise RuntimeError("DPA statistics requested but model has no diagnostics API")
            logits, trace = diagnostics_fn(inputs_device)
            if model.decoder_variant in {"patch_dpa_shared", "patch_dpa_independent"}:
                dpa_stats = _summarize_patch_dpa_trace(model, trace)
            elif model.decoder_variant == "patch_guided_sad":
                dpa_stats = _summarize_patch_sad_trace(model, trace)
            else:
                dpa_stats = _summarize_dpa_trace(model, trace)
        else:
            logits = model(inputs_device)
        confusion += confusion_from_logits(logits, targets, num_classes, ignore_index)
        images += inputs.shape[0]
    metrics = l4_global_metrics(confusion)
    if collect_dpa_stats:
        if dpa_stats is None:
            raise RuntimeError("DPA statistics requested but evaluation produced no batches")
        return metrics, confusion, images, dpa_stats
    return metrics, confusion, images


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    metrics: Optional[dict[str, float]],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def _load_init_checkpoint(model: nn.Module, path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Initialization checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported initialization checkpoint payload: {type(state)!r}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    variant = str(getattr(model, "decoder_variant", ""))
    if variant in {"dpa_ms_mlp", "dpa_sad_base"}:
        expected_missing = {
            "decoder.phi_3.weight",
            "decoder.phi_6.weight",
            "decoder.phi_9.weight",
            "decoder.alpha_3",
            "decoder.alpha_6",
            "decoder.alpha_9",
        }
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                "DPA initialization checkpoint alignment mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
    elif variant in {"patch_dpa_shared", "patch_dpa_independent"}:
        expected_missing = {
            "decoder.depth_alignments.0.weight",
            "decoder.depth_alignments.1.weight",
            "decoder.depth_alignments.2.weight",
            "decoder.depth_alignments.3.weight",
            "decoder.alpha_3",
            "decoder.alpha_6",
            "decoder.alpha_9",
            "decoder.alpha_12",
        }
        adapter_weight_names = (
            ("in_projection", "depthwise", "out_projection")
            if variant == "patch_dpa_shared"
            else None
        )
        if adapter_weight_names is not None:
            expected_missing.update(
                f"decoder.patch_adapter.{name}.weight" for name in adapter_weight_names
            )
        else:
            expected_missing.update(
                f"decoder.patch_adapters.{index}.{name}.weight"
                for index in range(4)
                for name in ("in_projection", "depthwise", "out_projection")
            )
        missing_set = set(missing)
        if missing_set not in (expected_missing, set()) or unexpected:
            raise RuntimeError(
                "Patch-DPA initialization checkpoint alignment mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
    elif variant == "patch_guided_sad":
        expected_missing = {
            name
            for name in model.state_dict()
            if name.startswith("decoder.patch_")
        }
        missing_set = set(missing)
        if missing_set not in (expected_missing, set()) or unexpected:
            raise RuntimeError(
                "Patch-guided SAD initialization checkpoint alignment mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
    elif variant == "tpa_sad":
        # B3R0 is initialized from the shared PR/SAD-Base state.  The ordinary
        # SAD R blocks are new, but all have gamma=0, so the active initial
        # function remains exactly the same as the neutral parent.
        expected_missing = {
            name
            for name in model.state_dict()
            if name.startswith("decoder.sad_intra_")
            or name.startswith("decoder.sad_inter_")
        }
        missing_set = set(missing)
        if missing_set not in (expected_missing, set()) or unexpected:
            raise RuntimeError(
                "SAD-R initialization checkpoint alignment mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
    elif missing or unexpected:
        raise RuntimeError(
            f"Initialization checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return {
        "path": str(path),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "model_variant": variant,
    }


def run(
    config: dict[str, Any],
    device: torch.device,
    smoke: bool = False,
    init_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    runtime = config.get("runtime", {})
    seed = int(runtime.get("seed", 20260901))
    deterministic = bool(runtime.get("deterministic", False))
    cudnn_benchmark = bool(runtime.get("cudnn_benchmark", False))
    if cudnn_benchmark and not bool(runtime.get("allow_large_cudnn_workspace", False)):
        raise ValueError(
            "cudnn_benchmark=True is blocked by default for SegDINO-v2 because "
            "it can select a multi-GiB convolution workspace. Set the explicit "
            "runtime.allow_large_cudnn_workspace=true only after a memory probe."
        )
    _seed_everything(seed, deterministic, cudnn_benchmark)

    data_root = _resolve_path(config["data_root"])
    input_size = _normalize_input_size(config["input_size"])
    num_classes = int(config["num_classes"])
    ignore_index = int(config["ignore_index"])
    training = config["training"]
    model_config = config["model"]
    patch_size = int(model_config.get("patch_size", 16))
    if any(dimension % patch_size != 0 for dimension in input_size):
        raise ValueError(
            f"input_size={input_size} must be divisible by DINO patch_size={patch_size}; "
            "otherwise border pixels are silently dropped by patch embedding"
        )
    model_cfg = ModelConfig(
        dino_size=model_config["dino_size"],
        dino_repo=str(_resolve_path(model_config["dino_repo"])),
        dino_ckpt=str(_resolve_path(model_config["dino_ckpt"])),
        decoder_dim=int(model_config["decoder_dim"]),
        use_bn=bool(model_config.get("use_bn", False)),
        num_classes=num_classes,
        patch_size=patch_size,
        decoder_variant=str(model_config.get("decoder_variant", "tpa_sad")),
        spatial_stride=int(model_config.get("spatial_stride", 4)),
        freeze_backbone=bool(model_config.get("freeze_backbone", True)),
        layer_mapping=model_config.get("layer_mapping"),
        adaptive_readout=bool(model_config.get("adaptive_readout", False)),
        readout_mode=str(model_config.get("readout_mode", "matrix")),
        readout_init=str(model_config.get("readout_init", "uniform")),
        readout_temperature=float(model_config.get("readout_temperature", 1.0)),
        wcf_enabled=bool(model_config.get("wcf_enabled", False)),
        wcf_reduction=int(model_config.get("wcf_reduction", 4)),
        wcf_alpha_init=float(model_config.get("wcf_alpha_init", 1e-2)),
    )

    worker_count = 0 if smoke else int(training.get("workers", 4))
    train_loader = _make_loader(
        data_root, config.get("train_split", "train"), input_size, True,
        int(training["batch_size"]), worker_count, config, device,
        persistent_workers=True,
    )
    val_loader = _make_loader(
        data_root, config.get("val_split", "val"), input_size, False,
        int(training.get("eval_batch_size", 1)), worker_count, config, device,
        persistent_workers=True,
    )
    test_loader = None
    selection_split = config.get("selection_split", "val")
    if selection_split not in {"val", "test"}:
        raise ValueError("selection_split must be 'val' or 'test'")
    selection_loader = val_loader
    if selection_split == "test":
        print("WARNING: selection_split=test; resulting metrics are development/test-tuned, not untouched-test results.")
        test_loader = _make_loader(
            data_root, config.get("test_split", "test"), input_size, False,
            int(training.get("eval_batch_size", 1)), worker_count, config, device,
            persistent_workers=True,
        )
        selection_loader = test_loader

    model, backbone = build_model(model_cfg, str(device))
    freeze_backbone = bool(model_config.get("freeze_backbone", True))
    if freeze_backbone:
        model.lock_backbone()
    spsr_enabled = model_cfg.decoder_variant == "l12_spsr_ms_mlp"
    if spsr_enabled:
        # The SPSR monitor is detached scalar bookkeeping only; it does not
        # change the forward graph or the optimization objective.
        model.set_spsr_monitor(True)
    init_checkpoint_info = None
    if init_checkpoint is not None:
        init_checkpoint_info = _load_init_checkpoint(model, _resolve_path(init_checkpoint))
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after applying freeze_backbone")

    optimizer_cfg = config["optimizer"]
    if optimizer_cfg.get("name", "AdamW") != "AdamW":
        raise ValueError("This runner currently supports only the explicit AdamW optimizer")

    base_lr = float(optimizer_cfg["lr"])
    backbone_lr = optimizer_cfg.get("backbone_lr")
    backbone_lr_mult = optimizer_cfg.get("backbone_lr_mult")
    if backbone_lr is None and backbone_lr_mult is not None:
        backbone_lr = base_lr * float(backbone_lr_mult)

    if not freeze_backbone and backbone_lr is not None:
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
        param_groups = [
            {"params": backbone_params, "lr": float(backbone_lr)},
            {"params": decoder_params, "lr": base_lr},
        ]
    else:
        param_groups = [{"params": trainable_params, "lr": base_lr}]

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=tuple(float(value) for value in optimizer_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    scheduler_cfg = config.get("scheduler", {"name": "constant"})
    scheduler_name = scheduler_cfg.get("name", "constant")
    if scheduler_name not in {"constant", "cosine"}:
        raise ValueError(
            f"Unsupported scheduler.name='{scheduler_name}'. Supported: 'constant', 'cosine'"
        )
    loss_cfg = config.get("loss")
    if not isinstance(loss_cfg, dict) or loss_cfg.get("name") != "CrossEntropyLoss":
        raise ValueError(
            "Set loss.name='CrossEntropyLoss' explicitly for mutually exclusive OSD labels"
        )
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    deep_supervision_weights = tuple(
        float(value)
        for value in training.get("deep_supervision_weights", [0.2, 0.2, 0.2])
    )
    if model_cfg.decoder_variant == "l12_a_cross_msef_ds":
        if len(deep_supervision_weights) != 3:
            raise ValueError(
                "l12_a_cross_msef_ds requires exactly three weights for inter3/inter2/inter1"
            )
        if any(value < 0 for value in deep_supervision_weights):
            raise ValueError("deep_supervision_weights must be non-negative")
    amp_enabled = bool(runtime.get("amp", False)) and device.type == "cuda"
    amp_dtype_name = runtime.get("amp_dtype", "float16")
    if amp_dtype_name not in {"float16", "bfloat16"}:
        raise ValueError("runtime.amp_dtype must be 'float16' or 'bfloat16'")
    amp_dtype = torch.float16 if amp_dtype_name == "float16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    total_params, backbone_params, decoder_params = summarize_parameters(model, backbone)
    trainable_count = sum(parameter.numel() for parameter in trainable_params)

    run_dir = _resolve_path(config.get("run_dir", "./runs/osd"))
    if smoke:
        run_dir = run_dir / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    log_file = run_dir / "train.log"
    def log_print(msg: str = "") -> None:
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log_print(
        f"model={config.get('name', 'unnamed')} input={input_size[0]}x{input_size[1]} "
        f"device={device} selection_split={selection_split}"
    )
    if init_checkpoint_info is not None:
        log_print(
            f"init_checkpoint={init_checkpoint_info['path']} "
            f"missing_keys={init_checkpoint_info['missing_keys']} "
            f"unexpected_keys={init_checkpoint_info['unexpected_keys']}"
        )
    log_print(
        f"params total={total_params / 1e6:.3f}M backbone={backbone_params / 1e6:.3f}M "
        f"decoder={decoder_params / 1e6:.3f}M trainable={trainable_count / 1e6:.3f}M"
    )
    if model_cfg.wcf_enabled:
        wcf_params = sum(parameter.numel() for block in model.get_wcf_blocks() for parameter in block.parameters())
        log_print(
            f"wcf=enabled blocks=3 reduction={model_cfg.wcf_reduction} "
            f"alpha_init={model_cfg.wcf_alpha_init} added_params={wcf_params}"
        )
    if not freeze_backbone and backbone_lr is not None:
        opt_info = f"optimizer=AdamW lr_decoder={base_lr} lr_backbone={float(backbone_lr)} weight_decay={optimizer_cfg['weight_decay']}"
    else:
        opt_info = f"optimizer=AdamW lr={optimizer_cfg['lr']} weight_decay={optimizer_cfg['weight_decay']}"
    log_print(
        f"{opt_info} scheduler={scheduler_name} "
        f"loss=CrossEntropyLoss ignore_index={ignore_index} freeze_backbone={freeze_backbone} "
        f"amp={amp_enabled} deterministic={deterministic} cudnn_benchmark={cudnn_benchmark}"
    )
    if model_config.get("layer_mapping") is not None:
        log_print(f"layer_mapping={model_config['layer_mapping']} (reversed/permuted routing)")
    elif model_cfg.wcf_enabled:
        log_print("layer_mapping=null (native [L3,L6,L9,L12] sources for L12-anchored WCF)")
    if model_cfg.adaptive_readout:
        log_print(
            f"adaptive_readout=True mode={model_cfg.readout_mode} "
            f"init={model_cfg.readout_init} tau={model_cfg.readout_temperature}"
        )
    if model_cfg.decoder_variant == "semantic_spatial":
        log_print(
            f"decoder_variant={model_cfg.decoder_variant} "
            f"spatial_stride={model_cfg.spatial_stride}"
        )
    elif model_cfg.decoder_variant == "dinov3_adapter":
        log_print(
            "decoder_variant=dinov3_adapter; official Meta DINOv3 Adapter with "
            "RGB SpatialPriorModule, four-even-interval interactions "
            "[2,5,8,11], multi-scale deformable attention, and official "
            "LinearHead adapted to OSD four-class output; no TPA/PR/SAD; "
            "official adapter internal autocast=bfloat16"
        )
    elif model_cfg.decoder_variant == "tpa_sad_msef":
        log_print(
            "decoder_variant=tpa_sad_msef; TPA produces "
            "256/128/64/32 branches; all eight SAD-R locations use the "
            "MSEF core (LayerNorm + depthwise 3x3 + SE), wrapped by "
            "zero-initialized residual gamma; reduction=16"
        )
    elif model_cfg.decoder_variant == "l12_a_msef":
        log_print(
            "decoder_variant=l12_a_msef; factorial-A shared 1x1 projection "
            "with bilinear-only L12 P2/P4/P8/P16; all eight SAD-R locations "
            "use the unwrapped official MSEF (LN + DWConv3x3 + full SE feature); "
            "no checkpoint initialization"
        )
    elif model_cfg.decoder_variant == "l12_a_cross_msef":
        log_print(
            "decoder_variant=l12_a_cross_msef; A-MSEF L12 semantic pyramid "
            "plus RGB Lite-SPM D2/D4/D8; Cross-MSEF at P2/P4/P8 with "
            "interaction_dim=64 and zero-initialized gamma; P16 unchanged; "
            "no checkpoint initialization"
        )
    elif model_cfg.decoder_variant == "l12_a_cross_r":
        log_print(
            "decoder_variant=l12_a_cross_r; A shared 1x1 + bilinear L12 pyramid "
            "plus RGB Lite-SPM D2/D4/D8; Cross-MSEF at P2/P4/P8 with "
            "interaction_dim=64; all eight SAD positions use the original "
            "ResidualDepthwiseBlock; no checkpoint initialization"
        )
    elif model_cfg.decoder_variant == "l12_a_cross_dweca":
        log_print(
            "decoder_variant=l12_a_cross_dweca; Cross-MSEF front-end is unchanged; "
            "all eight SAD positions use DW-ECA (GroupNorm + depthwise 3x3 + "
            "ECA kernel=5) with zero-initialized scalar gamma; no dense "
            "pointwise mixing and no checkpoint initialization"
        )
    elif model_cfg.decoder_variant == "l12_a_cross_bottleneck64":
        log_print(
            "decoder_variant=l12_a_cross_bottleneck64; Cross-MSEF front-end is "
            "unchanged; all eight SAD positions use nonlinear bottleneck "
            "refinement (DWConv + PW 256->64->256 + inner GELU + GroupNorm + "
            "GELU) with zero-initialized scalar gamma; no checkpoint "
            "initialization"
        )
    elif model_cfg.decoder_variant == "l12_a_cross_msef_cdr":
        log_print(
            "decoder_variant=l12_a_cross_msef_cdr; Cross-MSEF front-end is unchanged; "
            "all eight SAD positions use the same CDR block (LN + low/high split + "
            "context-selected detail); beta_init=1; no auxiliary loss"
        )
    elif model_cfg.decoder_variant == "l12_a_cross_msef_ds":
        log_print(
            "decoder_variant=l12_a_cross_msef_ds; Cross-MSEF + eight MSEF SAD blocks "
            "are unchanged; auxiliary CE heads follow inter3/inter2/inter1 with "
            f"weights={list(deep_supervision_weights)}; inference uses final head only"
        )
    elif model_cfg.decoder_variant in {"l12_a_cross_ee", "l12_a_cross_see"}:
        activation = "Sigmoid" if model_cfg.decoder_variant == "l12_a_cross_ee" else "Tanh"
        log_print(
            f"decoder_variant={model_cfg.decoder_variant}; Cross-MSEF front-end is unchanged; "
            f"all eight SAD positions use uniform EdgeEnhancer ({activation}); "
            "edge=x-AvgPool3x3(x), Conv1x1, GroupNorm, activation, residual add; "
            "count_include_pad=False; no R/MSEF/CDR/deep supervision; no checkpoint initialization"
        )
    elif model_cfg.decoder_variant == "l12_a_cross_r_weighted":
        log_print(
            "decoder_variant=l12_a_cross_r_weighted; Cross-MSEF front-end and all "
            "eight original SAD-R blocks are unchanged; P8/P4/P2 top-down fusion "
            "uses three pairs of zero-initialized scalar weights, initialized to "
            "exact P+Up; no checkpoint initialization"
        )
    elif model_cfg.decoder_variant == "l12_a_cross_r_dysample_splus":
        log_print(
            "decoder_variant=l12_a_cross_r_dysample_splus; Cross-MSEF front-end "
            "and all eight original SAD-R blocks are unchanged; only the three "
            "inter bilinear upsampling sites use official DySample-S+ "
            "(style=pl, groups=8, dyscope=True, scale=2); no checkpoint initialization"
        )
    elif model_cfg.decoder_variant == "mlp_same_scale":
        log_print(
            "decoder_variant=mlp_same_scale; four native [L3,L6,L9,L12] "
            "features remain at one 32x32 grid; no spatial decoder transform"
        )
    elif model_cfg.decoder_variant == "tpa_ms_mlp":
        log_print(
            "decoder_variant=tpa_ms_mlp; original TPA produces "
            "256/128/64/32 branches; neutral MS-MLP aligns to 256x256; no SAD"
        )
    elif model_cfg.decoder_variant == "tpa_change_cascade":
        log_print(
            "decoder_variant=tpa_change_cascade; existing TPA produces "
            "P2/P4/P8/P16=256/128/64/32; ChangeViT-style cascade uses "
            "1x1+4x4 stride-2 deconv and additive top-down fusion; no pairwise "
            "difference, ResNet detail branch, feature injector, or attention"
        )
    elif model_cfg.decoder_variant == "dpa_ms_mlp":
        log_print(
            "decoder_variant=dpa_ms_mlp; DPA after token_projection and before "
            "TPA reconstruction; TPA produces 256/128/64/32 branches; no SAD"
        )
    elif model_cfg.decoder_variant == "tpa_sad_base":
        log_print(
            "decoder_variant=tpa_sad_base; current TPA PR produces "
            "256/128/64/32 branches; neutral SAD-Base is progressive "
            "upsample+add only; no SAD refinement blocks"
        )
    elif model_cfg.decoder_variant == "dpa_sad_base":
        log_print(
            "decoder_variant=dpa_sad_base; DPA after token_projection and "
            "before TPA reconstruction; SAD-Base is progressive upsample+add "
            "only; no SAD refinement blocks"
        )
    elif model_cfg.decoder_variant == "patch_guided_sad":
        log_print(
            f"decoder_variant=patch_guided_sad; PR produces 256/128/64/32 branches; "
            f"raw K16/S{model_cfg.spatial_stride} patch prior is aligned to each scale; "
            "all eight SAD R locations are replaced by PatchGuidedR"
        )
    elif model_cfg.decoder_variant == "ltp_cli":
        log_print(
            "decoder_variant=ltp_cli; B1 PR+MS-MLP path is unchanged; "
            "RGB LTP produces S4/S8/S16 memory [16384+4096+1024,64]; "
            "four DINO levels read it with shared-K/V ReLU linear cross interaction"
        )
    elif model_cfg.decoder_variant in {"spm_ccfm_sad", "spm_ccfm_ms_mlp"}:
        backend = "original SAD" if model_cfg.decoder_variant == "spm_ccfm_sad" else "original MS-MLP"
        log_print(
            f"decoder_variant={model_cfg.decoder_variant}; RGB Lite-SPM produces "
            "D2/D4/D8=256/128/64; DINO L12 supplies D16=32; four-scale "
            f"RT-DETR CCFM outputs C2/C4/C8/C16; backend={backend}; "
            "no TPA/PR and no detection/query head"
        )
    elif model_cfg.decoder_variant in {
        "tpa_l12_shared_bilinear",
        "tpa_l12_shared_conv",
        "tpa_l12_independent_bilinear",
        "tpa_l12_independent_conv",
    }:
        projection = (
            "shared 1x1 projection"
            if "shared" in model_cfg.decoder_variant
            else "four independent 1x1 projections"
        )
        spatial = (
            "four independent dense 3x3 TPA branch convolutions"
            if model_cfg.decoder_variant.endswith("_conv")
            else "bilinear-only scale expansion"
        )
        log_print(
            f"decoder_variant={model_cfg.decoder_variant}; layer_mapping=[3,3,3,3] "
            f"(L12x4); {projection}; {spatial}; exact current SAD consumer"
        )

    epochs = training.get("epochs")
    max_iters = training.get("max_iters")
    if max_iters is None:
        if epochs is None:
            raise ValueError("Set either training.epochs or training.max_iters")
        max_iters = int(epochs) * len(train_loader)
    max_iters = int(max_iters)
    if smoke:
        max_iters = 1

    if scheduler_name == "cosine":
        warmup_epochs = float(scheduler_cfg.get("warmup_epochs", 0))
        warmup_steps = int(warmup_epochs * len(train_loader))
        min_lr = float(scheduler_cfg.get("min_lr", 1e-6))
        base_lr = float(optimizer_cfg["lr"])
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step + 1) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, max_iters - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = min_lr / base_lr
            return min_ratio + (1.0 - min_ratio) * cosine
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        log_print(f"scheduler=cosine warmup_steps={warmup_steps} ({warmup_epochs} epochs) min_lr={min_lr}")
    else:
        lr_scheduler = None

    val_interval = training.get("val_interval_steps")
    if val_interval is None:
        val_interval = len(train_loader)
    val_interval = 1 if smoke else int(val_interval)
    max_eval_batches = 1 if smoke else training.get("max_eval_batches")
    history: list[dict[str, Any]] = []
    spsr_monitor_history: list[dict[str, Any]] = []
    spsr_gate_step = int(training.get("spsr_learning_gate_step", 500))
    spsr_monitor_interval = int(training.get("spsr_monitor_interval_steps", 50))
    spsr_gate_passed = None
    spsr_stopped_early = False
    spsr_stop_reason = None
    best_value = float("-inf")
    best_routing_weights = None
    best_dpa_stats = None
    best_ltp_cli_gammas = None
    best_path = run_dir / "best.pth"
    latest_path = run_dir / "latest.pth"
    loader_iter = iter(train_loader)
    start_time = time.perf_counter()
    last_log_time = start_time
    log_interval = max(1, int(training.get("log_interval_steps", 50)))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    running_loss_sum = 0.0
    running_loss_count = 0
    running_final_loss_sum = 0.0
    running_aux_loss_sum = 0.0

    for step in range(1, max_iters + 1):
        try:
            inputs, targets, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            inputs, targets, _ = next(loader_iter)
        model.train()
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        final_loss_value = None
        aux_loss_value = None
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            if model_cfg.decoder_variant == "l12_a_cross_msef_ds":
                final_logits, aux_logits = model.forward_with_aux(inputs)
                logits = final_logits
                if len(aux_logits) != 3:
                    raise ValueError(
                        f"Expected 3 auxiliary logits, got {len(aux_logits)}"
                    )
                final_loss = criterion(final_logits, targets)
                aux_loss = sum(
                    weight * criterion(aux, targets)
                    for weight, aux in zip(deep_supervision_weights, aux_logits)
                )
                loss = final_loss + aux_loss
                final_loss_value = final_loss.detach()
                aux_loss_value = aux_loss.detach()
            else:
                logits = model(inputs)
                loss = criterion(logits, targets)
            if logits.shape[-2:] != targets.shape[-2:]:
                raise ValueError(f"Output/target shape mismatch: {logits.shape} vs {targets.shape}")
            if model_cfg.decoder_variant == "l12_a_cross_msef_ds":
                for aux in aux_logits:
                    if aux.shape[-2:] != targets.shape[-2:]:
                        raise ValueError(
                            f"Aux output/target shape mismatch: {aux.shape} vs {targets.shape}"
                        )
        if amp_enabled:
            spsr_event = model.get_spsr_learning_snapshot() if spsr_enabled else None
            scaler.scale(loss).backward()
            if spsr_enabled:
                # The snapshot is produced by the forward pass, while the
                # gradients are read immediately after backward below.
                spsr_event = model.get_spsr_learning_snapshot()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            spsr_event = model.get_spsr_learning_snapshot() if spsr_enabled else None
            optimizer.step()
        if spsr_enabled and spsr_event is not None:
            spsr_event = dict(spsr_event)
            spsr_event["step"] = step
            spsr_event["gamma_after_step"] = [
                float(value.detach().float().item())
                for value in model.decoder.gammas
            ]
            if step == 1 or step % max(1, spsr_monitor_interval) == 0 or step == spsr_gate_step:
                spsr_monitor_history.append(spsr_event)
            if step == spsr_gate_step:
                spsr_gate_passed, gate_summary = _spsr_learning_alive(spsr_event)
                log_print(
                    f"SPSR learning-alive gate step={step} "
                    f"passed={spsr_gate_passed} details={json.dumps(gate_summary, sort_keys=True)}"
                )
                if not spsr_gate_passed:
                    spsr_stopped_early = True
                    spsr_stop_reason = (
                        "SPSR branches remained inactive at the configured "
                        f"{spsr_gate_step}-step learning-alive gate"
                    )
                    log_print(f"SPSR early stop: {spsr_stop_reason}")
                    break
        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_val = float(loss.detach().cpu())
        running_loss_sum += loss_val
        running_loss_count += 1
        if final_loss_value is not None:
            running_final_loss_sum += float(final_loss_value.cpu())
        if aux_loss_value is not None:
            running_aux_loss_sum += float(aux_loss_value.cpu())

        if step == 1 or step % log_interval == 0:
            now = time.perf_counter()
            elapsed = now - start_time
            interval_seconds = now - last_log_time
            steps_since_log = 1 if step == 1 else log_interval
            step_seconds = interval_seconds / steps_since_log
            eta_seconds = max(0.0, (max_iters - step) * (elapsed / step))
            if len(optimizer.param_groups) > 1:
                lr_str = f"lr_dec={optimizer.param_groups[1]['lr']:.6g} lr_bb={optimizer.param_groups[0]['lr']:.6g}"
            else:
                lr_str = f"lr={optimizer.param_groups[0]['lr']:.8g}"
            log_print(
                f"step={step}/{max_iters} loss={loss_val:.5f} "
                f"{lr_str} "
                f"step_sec={step_seconds:.3f} eta_min={eta_seconds / 60.0:.1f}"
            )
            last_log_time = now

        should_validate = step == max_iters or step % val_interval == 0
        if should_validate:
            collect_dpa_stats = model_cfg.decoder_variant in {
                "dpa_ms_mlp",
                "dpa_sad_base",
                "patch_dpa_shared",
                "patch_dpa_independent",
                "patch_guided_sad",
            }
            evaluation = evaluate(
                model,
                selection_loader,
                device,
                num_classes,
                ignore_index,
                max_eval_batches,
                collect_dpa_stats=collect_dpa_stats,
            )
            if collect_dpa_stats:
                selection_metrics, confusion, evaluated_images, dpa_stats = evaluation
            else:
                selection_metrics, confusion, evaluated_images = evaluation
                dpa_stats = None
            epoch_mean_loss = running_loss_sum / max(1, running_loss_count)
            epoch_final_loss = (
                running_final_loss_sum / max(1, running_loss_count)
                if model_cfg.decoder_variant == "l12_a_cross_msef_ds"
                else None
            )
            epoch_aux_loss = (
                running_aux_loss_sum / max(1, running_loss_count)
                if model_cfg.decoder_variant == "l12_a_cross_msef_ds"
                else None
            )
            running_loss_sum = 0.0
            running_loss_count = 0
            running_final_loss_sum = 0.0
            running_aux_loss_sum = 0.0
            cur_lr = float(optimizer.param_groups[-1]["lr"])
            epoch_idx = int((step - 1) // len(train_loader) + 1)
            record = {
                "step": step,
                "epoch": epoch_idx,
                "train_loss": epoch_mean_loss,
                "last_step_loss": loss_val,
                "lr": cur_lr,
                "selection_split": selection_split,
                "selection_metrics": selection_metrics,
                "evaluated_images": evaluated_images,
            }
            if epoch_final_loss is not None:
                record["final_train_loss"] = epoch_final_loss
                record["aux_train_loss"] = epoch_aux_loss
            if dpa_stats is not None:
                record["dpa_stats"] = dpa_stats
            ltp_cli_gammas = None
            if model_cfg.decoder_variant == "ltp_cli":
                ltp_cli_gammas = _summarize_ltp_cli_gammas(model)
                record["ltp_cli_gammas"] = ltp_cli_gammas
            routing_w = model.get_routing_weights()
            if routing_w is not None:
                rw_np = routing_w.detach().cpu().numpy()
                record["routing_weights"] = rw_np.tolist()
            history.append(record)
            if len(optimizer.param_groups) > 1:
                epoch_lr_str = f"LR_dec={optimizer.param_groups[1]['lr']:.6g} LR_bb={optimizer.param_groups[0]['lr']:.6g}"
            else:
                epoch_lr_str = f"LR={cur_lr:.6g}"
            log_print(
                f"[Epoch {epoch_idx:02d} | Step {step:04d}/{max_iters}] {epoch_lr_str} | "
                f"Mean Train Loss={epoch_mean_loss:.5f} | "
                f"{selection_split}_mIoU3={selection_metrics['mIoU3_report_only_global']:.4f}% | "
                f"Oil={selection_metrics['IoU_oil_global']:.2f}% | "
                f"Water={selection_metrics['IoU_water_global']:.2f}% | "
                f"Others={selection_metrics['IoU_others_global']:.2f}%"
            )
            if epoch_final_loss is not None:
                log_print(
                    f"  Deep supervision losses: final={epoch_final_loss:.5f} "
                    f"aux_weighted={epoch_aux_loss:.5f} "
                    f"weights={list(deep_supervision_weights)}"
                )
            if dpa_stats is not None:
                alpha_str = " ".join(
                    f"{name}={value:.6f}" for name, value in dpa_stats["alpha"].items()
                )
                score_str = " | ".join(
                    f"{name}:mean={stats['mean']:.4f},std={stats['std']:.4f}"
                    for name, stats in dpa_stats["scores"].items()
                )
                log_print(f"  DPA alpha: {alpha_str}")
                log_print(f"  DPA scores: {score_str}")
                if "cosines" in dpa_stats:
                    cosine_str = " | ".join(
                        f"{name}:mean={stats['mean']:.4f},std={stats['std']:.4f}"
                        for name, stats in dpa_stats["cosines"].items()
                    )
                    residual_str = " | ".join(
                        f"{name}={stats['mean']:.6f}"
                        for name, stats in dpa_stats["residual_ratios"].items()
                    )
                    log_print(f"  DPA cosine: {cosine_str}")
                    log_print(f"  DPA residual_ratio: {residual_str}")
            if ltp_cli_gammas is not None:
                gamma_str = " ".join(
                    f"{name}={value:.6f}"
                    for name, value in ltp_cli_gammas.items()
                )
                log_print(f"  LTP-CLI gamma: {gamma_str}")
            if routing_w is not None and model_cfg.readout_mode in ("matrix", "uniform"):
                rw_str = " | ".join([f"s{s}:[" + " ".join([f"{w:.2f}" for w in rw_np[s]]) + "]" for s in range(rw_np.shape[0])])
                log_print(f"  ALSR Routing: {rw_str}")
            _save_checkpoint(
                latest_path, model, optimizer, step, record["epoch"], selection_metrics, config
            )
            score = selection_metrics["mIoU3_report_only_global"]
            if np.isfinite(score) and score > best_value:
                best_value = score
                best_dpa_stats = dpa_stats
                best_ltp_cli_gammas = ltp_cli_gammas
                if routing_w is not None:
                    best_routing_weights = rw_np.tolist()
                _save_checkpoint(
                    best_path, model, optimizer, step, record["epoch"], selection_metrics, config
                )
            payload = make_l4_payload(
                confusion,
                split=selection_split,
                input_size=_input_size_metadata(input_size),
                protocol_id=config.get("protocol_id", "OSD-EXP-v1.0/L4-global-512"),
                model_name=config.get("name"),
                evaluated_images=evaluated_images,
                selection_split=selection_split,
            )
            write_l4_payload(payload, run_dir / f"{selection_split}_step_{step:06d}.json")

    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - start_time
    summary = {
        "run_dir": str(run_dir),
        "max_iters": max_iters,
        "elapsed_seconds": elapsed,
        "best_selection_mIoU3_report_only_global": best_value,
        "selection_split": selection_split,
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "latest_checkpoint": str(latest_path) if latest_path.exists() else None,
        "spsr_learning_gate_step": spsr_gate_step if spsr_enabled else None,
        "spsr_learning_gate_passed": spsr_gate_passed if spsr_enabled else None,
        "spsr_stopped_early": spsr_stopped_early,
        "spsr_stop_reason": spsr_stop_reason,
        "spsr_monitor_trajectory": spsr_monitor_history if spsr_enabled else None,
        "params_total": total_params,
        "params_backbone": backbone_params,
        "params_decoder": decoder_params,
        "params_trainable": trainable_count,
        "init_checkpoint": init_checkpoint_info,
        "peak_memory_mib": round(
            torch.cuda.max_memory_allocated(device) / 1024**2, 2
        ) if device.type == "cuda" else None,
        "config": config,
    }
    if model_cfg.decoder_variant in {
        "dpa_ms_mlp",
        "dpa_sad_base",
        "patch_dpa_shared",
        "patch_dpa_independent",
        "patch_guided_sad",
    } and history:
        summary["dpa_diagnostics_last"] = history[-1].get("dpa_stats")
        summary["dpa_diagnostics_best"] = best_dpa_stats
    if model_cfg.decoder_variant == "ltp_cli" and history:
        summary["ltp_cli_gamma_trajectory"] = [
            {
                "epoch": item["epoch"],
                "step": item["step"],
                **item["ltp_cli_gammas"],
            }
            for item in history
            if "ltp_cli_gammas" in item
        ]
        summary["ltp_cli_gamma_best"] = best_ltp_cli_gammas
    if model_cfg.adaptive_readout:
        final_rw = model.get_routing_weights()
        if final_rw is not None:
            summary["final_routing_weights"] = final_rw.detach().cpu().numpy().tolist()
        if best_routing_weights is not None:
            summary["best_routing_weights"] = best_routing_weights
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log_print(f"Training completed in {elapsed:.2f}s ({elapsed/60.0:.2f}min). Best {selection_split}_mIoU3={best_value:.4f}% saved at {best_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None, help="e.g. cuda or cpu; defaults to config/auto")
    parser.add_argument("--smoke", action="store_true", help="one real OSD train batch plus one eval batch")
    parser.add_argument("--max-iters", type=int, default=None, help="explicit short-run override")
    parser.add_argument("--workers", type=int, default=None, help="explicit DataLoader worker override")
    parser.add_argument("--selection-split", choices=("val", "test"), default=None)
    parser.add_argument("--run-dir", default=None, help="explicit output directory override")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="optional model checkpoint used only to initialize weights; optimizer starts fresh",
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.max_iters is not None:
        config["training"]["max_iters"] = args.max_iters
        config["training"]["epochs"] = None
    if args.workers is not None:
        config["training"]["workers"] = args.workers
    if args.selection_split is not None:
        config["selection_split"] = args.selection_split
    if args.run_dir is not None:
        config["run_dir"] = args.run_dir
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config.get("runtime", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    run(config, device, smoke=args.smoke, init_checkpoint=args.init_checkpoint)


if __name__ == "__main__":
    main()
