#!/usr/bin/env python3
"""Export a fresh TPA+MS-MLP state for the clean B2 DPA rerun.

The exported state contains the frozen pretrained DINOv3 backbone and a
freshly initialized TPA+MS-MLP decoder.  It is deliberately not loaded from
the completed B1 checkpoint.  The B2 runner can load the shared keys from
this state; its six DPA-only parameters remain constructor-initialized.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import ModelConfig  # noqa: E402
from runtime import build_model  # noqa: E402


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "configs/dino_tpa_base_msmlp_osd_512.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR
        / "work_dirs/dino_tpa_dpa_msmlp_clean_common_init_seed20260901.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config_path = args.config.resolve()
    output_path = args.output.resolve()
    raw = _load_raw(config_path)
    model_cfg = _model_config(raw)
    if model_cfg.decoder_variant != "tpa_ms_mlp":
        raise RuntimeError(
            f"Clean shared init must be built from tpa_ms_mlp, got {model_cfg.decoder_variant!r}"
        )
    seed = int(raw.get("runtime", {}).get("seed", 20260901))
    _set_seed(seed)
    device = torch.device(args.device)
    model, backbone = build_model(model_cfg, str(device))
    model.lock_backbone()
    model.eval()

    if not all(not parameter.requires_grad for parameter in backbone.parameters()):
        raise RuntimeError("Backbone was not fully frozen")
    state = copy.deepcopy(model.state_dict())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": state,
            "config": raw,
            "purpose": "fresh shared initialization for clean B1/B2 comparison",
            "initialization_type": "fresh_model_seeded_decoder",
            "seed": seed,
            "source_config": str(config_path),
            "segdino_v2_commit": _git_commit(),
            "backbone_checkpoint": str(_resolve_path(raw["model"]["dino_ckpt"])),
        },
        output_path,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output_path),
                "source_config": str(config_path),
                "initialization_type": "fresh_model_seeded_decoder",
                "seed": seed,
                "device": str(device),
                "decoder_variant": model_cfg.decoder_variant,
                "backbone_frozen": True,
                "state_keys": len(state),
                "segdino_v2_commit": _git_commit(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
