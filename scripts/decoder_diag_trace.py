#!/usr/bin/env python3
"""Prepared Phase 2/3 trace interfaces for DEC-DIAG-001.

This module intentionally has no automatic experiment entry point.  It
provides a temporary decoder-forward trace that mirrors the current
TPA/SADDecoder call order without changing ``dpt.py`` or checkpoint weights:

* Phase 2 sources: decoder input tokens (raw L12 for CTRL-003), four TPA
  outputs after projection/resampling, and four SAD intra outputs.
* Phase 3 sources: the two tensors immediately before each top-down addition,
  ``Upsample(T_{i+1})`` and the lateral ``F_i``.

Future callers must explicitly install the trace and decide whether to train
or analyse a probe.  This file itself never loads data, a checkpoint, or the
test split.
"""

from __future__ import annotations

import json
import types
from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn.functional as F


def trace_decoder_forward(decoder: torch.nn.Module, features, patch_h: int, patch_w: int):
    """Run the current decoder order and return logits plus diagnostic tensors.

    The operation order is intentionally the same as ``TPASADDecoder.forward``:
    token projection -> optional ALSR -> TPA resampling/projection -> SAD intra
    -> top-down SAD.  No parameters or module definitions are changed.
    """
    projected = []
    for index, tokens in enumerate(features):
        feature_map = decoder._tokens_to_feature_map(tokens, patch_h, patch_w)
        projected.append(decoder.token_projections[index](feature_map))

    if decoder.alsr is not None:
        projected = decoder.alsr(projected)

    tpa_outputs = [
        decoder.tpa_branch_1(projected[0]),
        decoder.tpa_branch_2(projected[1]),
        decoder.tpa_branch_3(projected[2]),
        decoder.tpa_branch_4(projected[3]),
    ]
    sad_intra_outputs = [
        decoder.sad_intra_1(tpa_outputs[0]),
        decoder.sad_intra_2(tpa_outputs[1]),
        decoder.sad_intra_3(tpa_outputs[2]),
        decoder.sad_intra_4(tpa_outputs[3]),
    ]

    x4 = decoder.sad_inter_4(sad_intra_outputs[3])
    x3_up = F.interpolate(x4, size=sad_intra_outputs[2].shape[-2:], mode="bilinear", align_corners=False)
    x3_merge = x3_up + sad_intra_outputs[2]
    x3 = decoder.sad_inter_3(x3_merge)

    x2_up = F.interpolate(x3, size=sad_intra_outputs[1].shape[-2:], mode="bilinear", align_corners=False)
    x2_merge = x2_up + sad_intra_outputs[1]
    x2 = decoder.sad_inter_2(x2_merge)

    x1_up = F.interpolate(x2, size=sad_intra_outputs[0].shape[-2:], mode="bilinear", align_corners=False)
    x1_merge = x1_up + sad_intra_outputs[0]
    x1 = decoder.sad_inter_1(x1_merge)

    trace = {
        "decoder_input_features": list(features),
        "projected_features_before_tpa_resampling": projected,
        "tpa_outputs_after_projection_and_resampling": tpa_outputs,
        "sad_intra_outputs": sad_intra_outputs,
        "topdown_merges": [
            {"scale": "x2", "upsampled": x3_up, "lateral": sad_intra_outputs[2], "fused": x3_merge},
            {"scale": "x4", "upsampled": x2_up, "lateral": sad_intra_outputs[1], "fused": x2_merge},
            {"scale": "x8", "upsampled": x1_up, "lateral": sad_intra_outputs[0], "fused": x1_merge},
        ],
    }
    return decoder.out_conv(x1), trace


@contextmanager
def install_decoder_trace(model: torch.nn.Module) -> Iterator[None]:
    """Temporarily capture the last decoder trace during ``model(...)``.

    The trace is available as ``model.decoder._decoder_diag_last_trace`` while
    the context is active.  It is removed and the original forward method is
    restored on exit.
    """
    decoder = model.decoder
    original_forward = decoder.forward

    def traced_forward(self, features, patch_h, patch_w):
        logits, trace = trace_decoder_forward(self, features, patch_h, patch_w)
        self._decoder_diag_last_trace = trace
        return logits

    decoder.forward = types.MethodType(traced_forward, decoder)
    decoder._decoder_diag_last_trace = None
    try:
        yield
    finally:
        decoder.forward = original_forward
        if hasattr(decoder, "_decoder_diag_last_trace"):
            delattr(decoder, "_decoder_diag_last_trace")


def phase2_sources(trace: dict[str, Any]) -> dict[str, Any]:
    """Return the named tensors intended for future frozen task probes."""
    return {
        "raw_decoder_input": trace["decoder_input_features"],
        "tpa_outputs": trace["tpa_outputs_after_projection_and_resampling"],
        "sad_intra_outputs": trace["sad_intra_outputs"],
    }


def _safe_pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    if x.numel() < 2 or float(x.std(unbiased=False)) == 0.0 or float(y.std(unbiased=False)) == 0.0:
        return None
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1].detach().cpu())


def topdown_conflict_statistics(
    trace: dict[str, Any],
    logits: torch.Tensor | None = None,
    targets: torch.Tensor | None = None,
    *,
    low_similarity_threshold: float = 0.2,
    ignore_index: int = 255,
) -> list[dict[str, Any]]:
    """Summarize pre-add cosine conflict for each top-down merge.

    ``low_similarity_threshold`` is an analysis parameter, not a network
    change.  Error correlation is included only when logits and targets are
    supplied; all tensors are reduced immediately so callers need not retain
    full feature maps.
    """
    if not 0.0 <= low_similarity_threshold <= 1.0:
        raise ValueError("low_similarity_threshold must be in [0, 1]")
    rows = []
    for merge in trace["topdown_merges"]:
        upsampled = merge["upsampled"]
        lateral = merge["lateral"]
        cosine = F.cosine_similarity(upsampled.float(), lateral.float(), dim=1, eps=1e-8)
        row: dict[str, Any] = {
            "scale": merge["scale"],
            "positions": int(cosine.numel()),
            "cosine_mean": float(cosine.mean().detach().cpu()),
            "cosine_std_population": float(cosine.std(unbiased=False).detach().cpu()),
            "cosine_min": float(cosine.min().detach().cpu()),
            "cosine_max": float(cosine.max().detach().cpu()),
            "negative_cosine_ratio": float((cosine < 0).float().mean().detach().cpu()),
            "low_similarity_threshold": float(low_similarity_threshold),
            "low_similarity_ratio": float((cosine < low_similarity_threshold).float().mean().detach().cpu()),
        }
        if logits is not None and targets is not None:
            target = targets.to(logits.device)
            if target.ndim == 4 and target.shape[1] == 1:
                target = target[:, 0]
            pred = F.interpolate(logits, size=cosine.shape[-2:], mode="bilinear", align_corners=False).argmax(dim=1)
            target = F.interpolate(target.unsqueeze(1).float(), size=cosine.shape[-2:], mode="nearest").squeeze(1).long()
            valid = target != ignore_index
            error = (pred != target) & valid
            row["valid_positions"] = int(valid.sum().detach().cpu())
            row["segmentation_error_ratio"] = float(error.sum().detach().cpu() / max(1, int(valid.sum().detach().cpu())))
            row["cosine_error_pearson"] = _safe_pearson(cosine[valid], error[valid].float()) if bool(valid.any()) else None
        rows.append(row)
    return rows


def prepared_plan() -> dict[str, Any]:
    return {
        "status": "prepared_only",
        "experiment_id": "DINO-DEC-DIAG-001",
        "phase2": {
            "sources": [
                "raw decoder input tokens (CTRL-003: four L12 inputs)",
                "four TPA outputs after projection/resampling",
                "four SAD intra outputs",
            ],
            "probe": "future frozen light segmentation probe; execution requires research-side approval",
        },
        "phase3": {
            "sources": "Upsample(T_{i+1}) and F_i immediately before each unconditional top-down addition",
            "statistics": ["cosine distribution", "negative cosine ratio", "low-similarity ratio", "optional error correlation"],
            "threshold": "explicit analysis argument; no network change",
            "execution": "not run automatically",
        },
        "test_accessed": False,
        "training_started": False,
    }


if __name__ == "__main__":
    print(json.dumps(prepared_plan(), indent=2, ensure_ascii=False))
