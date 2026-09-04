import importlib
import sys
from pathlib import Path
from typing import Tuple

import torch

from config_loader import ModelConfig, resolve_encoder_size
from dpt import DPT


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_backbone(model_config: ModelConfig):
    """Load only the requested DINOv3 backbone from a local checkout.

    The official ``hubconf.py`` imports optional classifier/detector/segmentor
    modules as well as the backbone.  Those optional modules can require a
    newer PyTorch API than the backbone itself, so using ``torch.hub.load``
    makes an otherwise valid backbone environment fail during import.  Import
    the backbone factory directly and pass the local checkpoint to it instead.
    """
    repo = Path(model_config.dino_repo).expanduser().resolve()
    if not (repo / "dinov3").is_dir():
        raise FileNotFoundError(f"DINOv3 source directory not found: {repo}")
    checkpoint = Path(model_config.dino_ckpt).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint}")

    # The DINOv3 checkout is intentionally external to this repository.  Add
    # it only when needed so the same SegDINO source can point at Project2 or
    # another official checkout without copying model code into this repo.
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    backbones = importlib.import_module("dinov3.hub.backbones")
    try:
        constructor = {
            "s": backbones.dinov3_vits16,
            "b": backbones.dinov3_vitb16,
        }[model_config.dino_size]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dino_size '{model_config.dino_size}'. "
            "The current adapter supports 's' and 'b'."
        ) from exc

    return constructor(pretrained=True, weights=str(checkpoint))


def build_model(model_config: ModelConfig, device: str) -> Tuple[torch.nn.Module, torch.nn.Module]:
    backbone = load_backbone(model_config)
    model = DPT(
        encoder_size=resolve_encoder_size(model_config.dino_size),
        nclass=model_config.num_classes,
        decoder_channels=model_config.decoder_dim,
        use_bn=model_config.use_bn,
        patch_size=model_config.patch_size,
        backbone=backbone,
    ).to(device)
    return model, backbone


def summarize_parameters(model, backbone) -> tuple[int, int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in backbone.parameters())
    other_params = total_params - backbone_params
    return total_params, backbone_params, other_params
