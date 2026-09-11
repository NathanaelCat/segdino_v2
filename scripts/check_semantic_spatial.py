import argparse
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import DEFAULT_TRAIN_EXPERIMENT, get_train_config
from runtime import build_model, get_device, summarize_parameters


def main():
    parser = argparse.ArgumentParser(description="Sanity-check the semantic-spatial decoder path.")
    parser.add_argument("--experiment", default=DEFAULT_TRAIN_EXPERIMENT)
    parser.add_argument("--img-size", type=int, default=512)
    args = parser.parse_args()

    config = get_train_config(args.experiment)
    if config.model.decoder_variant != "semantic_spatial":
        raise ValueError(
            "This check expects decoder_variant='semantic_spatial'; "
            f"got {config.model.decoder_variant!r}."
        )

    device = get_device()
    model, backbone = build_model(config.model, device)
    model.eval()

    if config.model.freeze_backbone:
        trainable_backbone = [name for name, p in backbone.named_parameters() if p.requires_grad]
        if trainable_backbone:
            raise RuntimeError(f"Backbone is not fully frozen: {trainable_backbone[:5]}")

    img_size = args.img_size
    x = torch.randn(1, 3, img_size, img_size, device=device)
    patch_h = img_size // config.model.patch_size
    patch_w = img_size // config.model.patch_size

    final_layer_idx = model.intermediate_layer_idx[model.encoder_size][-1]
    with torch.no_grad():
        semantic_tokens = backbone.get_intermediate_layers(x, n=[final_layer_idx])[-1]
        spatial_prior = model._shared_highres_patch_projection(x)
        pyramid = model.decoder.build_pyramid(
            semantic_tokens,
            spatial_prior,
            patch_h,
            patch_w,
        )
        logits = model(x)

    p2, p4, p8, p16 = pyramid
    expected_hr = (img_size - config.model.patch_size) // config.model.spatial_stride + 1
    expected_shapes = {
        "P2": (patch_h * 8, patch_w * 8),
        "P4": (patch_h * 4, patch_w * 4),
        "P8": (patch_h * 2, patch_w * 2),
        "P16": (patch_h, patch_w),
    }
    actual_shapes = {
        "P2": p2.shape[-2:],
        "P4": p4.shape[-2:],
        "P8": p8.shape[-2:],
        "P16": p16.shape[-2:],
    }

    for name, expected in expected_shapes.items():
        if tuple(actual_shapes[name]) != tuple(expected):
            raise RuntimeError(f"{name} shape mismatch: got {actual_shapes[name]}, expected {expected}")

    if spatial_prior.shape[-2:] != (expected_hr, expected_hr):
        raise RuntimeError(
            "Unexpected high-resolution patch grid: "
            f"got {tuple(spatial_prior.shape[-2:])}, expected {(expected_hr, expected_hr)}"
        )
    if logits.shape[-2:] != (img_size, img_size):
        raise RuntimeError(
            f"Output shape mismatch: got {tuple(logits.shape[-2:])}, expected {(img_size, img_size)}"
        )

    total_params, backbone_params, other_params = summarize_parameters(model, backbone)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"device={device}")
    print(f"decoder_variant={config.model.decoder_variant}")
    print(f"backbone_frozen={config.model.freeze_backbone}")
    print(f"semantic_tokens={tuple(semantic_tokens.shape)}")
    print(f"spatial_prior={tuple(spatial_prior.shape)}")
    print(f"P2/P4/P8/P16={[tuple(t.shape) for t in pyramid]}")
    print(f"logits={tuple(logits.shape)}")
    print(f"params_total={total_params:,}")
    print(f"params_backbone={backbone_params:,}")
    print(f"params_non_backbone={other_params:,}")
    print(f"params_trainable={trainable_params:,}")
    print("sanity_check=PASS")


if __name__ == "__main__":
    main()
