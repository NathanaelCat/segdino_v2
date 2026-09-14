#!/usr/bin/env python3
"""Audit the direct SPSR parent/candidate pair before formal training.

The check is deliberately test-free and training-free.  It verifies that the
parent and candidate share the same initialization for every common state
key, that gamma=0 makes the candidate output equal the parent output, and
that the only additional trainable state is the SPSR core plus three gammas.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config_loader import ModelConfig  # noqa: E402
from runtime import build_model, summarize_parameters  # noqa: E402


PARENT_CONFIG = ROOT_DIR / "configs/dino_l12_spm_msmlp_osd_512.json"
CANDIDATE_CONFIG = ROOT_DIR / "configs/dino_l12_spm_spsr_msmlp_osd_512.json"
DEFAULT_OUTPUT = ROOT_DIR / "work_dirs/dino_l12_spsr_pair_preflight.json"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def _load(path: Path) -> tuple[dict, ModelConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    model = raw["model"]
    config = ModelConfig(
        dino_size=model["dino_size"],
        dino_repo=str(_resolve(model["dino_repo"])),
        dino_ckpt=str(_resolve(model["dino_ckpt"])),
        decoder_dim=int(model["decoder_dim"]),
        use_bn=bool(model.get("use_bn", False)),
        num_classes=int(raw["num_classes"]),
        patch_size=int(model.get("patch_size", 16)),
        decoder_variant=str(model.get("decoder_variant", "tpa_sad")),
        spatial_stride=int(model.get("spatial_stride", 4)),
        freeze_backbone=bool(model.get("freeze_backbone", True)),
        layer_mapping=model.get("layer_mapping"),
    )
    return raw, config


def _strip_identity(raw: dict) -> dict:
    value = copy.deepcopy(raw)
    value.pop("name", None)
    value.pop("run_dir", None)
    value.get("runtime", {}).pop("device", None)
    value.get("model", {}).pop("decoder_variant", None)
    return value


def _diff(left, right, path=""):
    if type(left) is not type(right):
        return [(path, left, right)]
    if isinstance(left, dict):
        output = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                output.append((f"{path}.{key}", left.get(key), right.get(key)))
            else:
                output.extend(_diff(left[key], right[key], f"{path}.{key}"))
        return output
    if isinstance(left, list):
        output = []
        for index in range(max(len(left), len(right))):
            if index >= len(left) or index >= len(right):
                output.append(
                    (
                        f"{path}.{index}",
                        left[index] if index < len(left) else None,
                        right[index] if index < len(right) else None,
                    )
                )
            else:
                output.extend(_diff(left[index], right[index], f"{path}.{index}"))
        return output
    return [] if left == right else [(path, left, right)]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _shape(value: torch.Tensor) -> list[int]:
    return [int(dim) for dim in value.shape]


def _count(parameters) -> int:
    return sum(parameter.numel() for parameter in parameters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-config", type=Path, default=PARENT_CONFIG)
    parser.add_argument("--candidate-config", type=Path, default=CANDIDATE_CONFIG)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    parent_raw, parent_cfg = _load(_resolve(args.parent_config))
    candidate_raw, candidate_cfg = _load(_resolve(args.candidate_config))
    if parent_cfg.decoder_variant != "l12_spm_ms_mlp":
        raise ValueError("Parent must use decoder_variant=l12_spm_ms_mlp")
    if candidate_cfg.decoder_variant != "l12_spsr_ms_mlp":
        raise ValueError("Candidate must use decoder_variant=l12_spsr_ms_mlp")
    config_diffs = _diff(_strip_identity(parent_raw), _strip_identity(candidate_raw))
    allowed = {".model.decoder_variant"}
    unexpected_config_diffs = [item for item in config_diffs if item[0] not in allowed]
    if unexpected_config_diffs:
        raise RuntimeError(f"Parent/candidate config drift: {unexpected_config_diffs}")

    seed = int(candidate_raw.get("runtime", {}).get("seed", 20260901))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    _seed_all(seed)
    candidate, candidate_backbone = build_model(candidate_cfg, str(device))
    candidate.lock_backbone()
    _seed_all(seed)
    parent, parent_backbone = build_model(parent_cfg, str(device))
    parent.lock_backbone()

    backbone_frozen = all(not p.requires_grad for p in candidate_backbone.parameters())
    parent_backbone_frozen = all(not p.requires_grad for p in parent_backbone.parameters())
    patch_frozen = all(
        not p.requires_grad for p in candidate_backbone.patch_embed.parameters()
    )
    parent_patch_frozen = all(
        not p.requires_grad for p in parent_backbone.patch_embed.parameters()
    )
    if not all((backbone_frozen, parent_backbone_frozen, patch_frozen, parent_patch_frozen)):
        raise RuntimeError("Parent/candidate backbone or PatchEmbed is not frozen")

    batch_size = int(args.batch_size)
    patch_h = args.img_size // candidate_cfg.patch_size
    patch_w = args.img_size // candidate_cfg.patch_size
    _seed_all(seed + 1)
    x = torch.randn(batch_size, 3, args.img_size, args.img_size, device=device)
    candidate.eval()
    parent.eval()
    with torch.no_grad():
        candidate_logits = candidate(x)
        parent_logits = parent(x)
    output_diff = float((candidate_logits - parent_logits).abs().max().item())

    candidate_state = candidate.state_dict()
    parent_state = parent.state_dict()
    parent_keys = set(parent_state)
    candidate_keys = set(candidate_state)
    common_keys = sorted(parent_keys & candidate_keys)
    extra_candidate_keys = sorted(candidate_keys - parent_keys)
    missing_candidate_keys = sorted(parent_keys - candidate_keys)
    shared_state_diff = 0.0
    shared_state_bad = []
    for key in common_keys:
        value_diff = float((candidate_state[key].detach().cpu() - parent_state[key].detach().cpu()).abs().max().item())
        shared_state_diff = max(shared_state_diff, value_diff)
        if value_diff > 0.0:
            shared_state_bad.append((key, value_diff))

    candidate_total, candidate_backbone_params, candidate_non_backbone = summarize_parameters(
        candidate, candidate_backbone
    )
    parent_total, parent_backbone_params, parent_non_backbone = summarize_parameters(
        parent, parent_backbone
    )
    candidate_trainable_names = [
        name for name, parameter in candidate.named_parameters() if parameter.requires_grad
    ]
    parent_trainable_names = [
        name for name, parameter in parent.named_parameters() if parameter.requires_grad
    ]
    candidate_extra_trainable = sorted(set(candidate_trainable_names) - set(parent_trainable_names))
    parent_extra_trainable = sorted(set(parent_trainable_names) - set(candidate_trainable_names))
    expected_extra = sorted(
        [
            f"decoder.resampler.{name}"
            for name, _ in candidate.decoder.resampler.named_parameters()
        ]
        + [f"decoder.gammas.{index}" for index in range(len(candidate.decoder.gammas))]
    )
    expected_parameter_delta = _count(candidate.decoder.resampler.parameters()) + _count(
        candidate.decoder.gammas
    )

    result = {
        "current_branch": subprocess_check_git("branch", "--show-current"),
        "current_commit": subprocess_check_git("rev-parse", "HEAD"),
        "parent_model": parent_raw["name"],
        "candidate_model": candidate_raw["name"],
        "parent_decoder_variant": parent_cfg.decoder_variant,
        "candidate_decoder_variant": candidate_cfg.decoder_variant,
        "formal_training_started": False,
        "test_split_accessed": False,
        "config_diffs": config_diffs,
        "unexpected_config_diffs": unexpected_config_diffs,
        "input_shape": _shape(x),
        "parent_logits_shape": _shape(parent_logits),
        "candidate_logits_shape": _shape(candidate_logits),
        "gamma_zero_candidate_vs_parent_max_abs_diff": output_diff,
        "shared_state_key_count": len(common_keys),
        "shared_state_max_abs_diff": shared_state_diff,
        "shared_state_mismatches": shared_state_bad[:20],
        "candidate_extra_state_keys": extra_candidate_keys,
        "missing_candidate_state_keys": missing_candidate_keys,
        "candidate_extra_trainable_names": candidate_extra_trainable,
        "parent_extra_trainable_names": parent_extra_trainable,
        "expected_spsr_extra_trainable_names": expected_extra,
        "parameter_counts": {
            "parent_total": parent_total,
            "candidate_total": candidate_total,
            "parent_backbone": parent_backbone_params,
            "candidate_backbone": candidate_backbone_params,
            "parent_non_backbone_trainable": parent_non_backbone,
            "candidate_non_backbone_trainable": candidate_non_backbone,
            "total_delta": candidate_total - parent_total,
            "trainable_delta": candidate_non_backbone - parent_non_backbone,
            "expected_spsr_delta": expected_parameter_delta,
        },
        "frozen_checks": {
            "candidate_backbone": backbone_frozen,
            "parent_backbone": parent_backbone_frozen,
            "candidate_patch_embed": patch_frozen,
            "parent_patch_embed": parent_patch_frozen,
        },
    }
    result["sanity_checks"] = {
        "only_decoder_variant_config_diff": not unexpected_config_diffs,
        "parent_candidate_common_initialization_equal": shared_state_diff == 0.0,
        "gamma_zero_exact_parent_output": output_diff <= 1e-7,
        "candidate_extra_state_is_spsr_only": set(extra_candidate_keys)
        == set(expected_extra),
        "candidate_extra_trainable_is_spsr_only": set(candidate_extra_trainable)
        == set(expected_extra),
        "no_parent_extra_trainable": not parent_extra_trainable,
        "parameter_delta_matches_spsr": candidate_total - parent_total == expected_parameter_delta,
        "logits_shape_512": tuple(candidate_logits.shape)
        == (batch_size, candidate_cfg.num_classes, args.img_size, args.img_size),
        "frozen_backbones": all(result["frozen_checks"].values()),
    }
    result["sanity_check"] = "PASS" if all(result["sanity_checks"].values()) else "FAIL"

    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    del candidate, parent, candidate_backbone, parent_backbone, x
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if result["sanity_check"] != "PASS":
        raise SystemExit(2)


def subprocess_check_git(*args: str) -> str:
    import subprocess

    return subprocess.check_output(["git", *args], cwd=ROOT_DIR, text=True).strip()


if __name__ == "__main__":
    main()
