# SegDINO-v2 on OSD

This directory is an independent working copy of `script-Yang/segdino_v2`.
The original binary-dataset entry points are kept intact.  The OSD runner is
`train_osd.py` and uses the four indexed labels
`background / oil / others / water`.

## Environment

The tested environment is the conda environment `segdino_osd`:

```bash
conda activate segdino_osd
```

The DINOv3 source and checkpoints are intentionally external.  The supplied
configs point to the existing Project2 checkout and weights; change those two
paths in a copy of the config when moving machines.

## Smoke and training profiles

Run a real OSD train batch, including backward, followed by one validation
batch:

```bash
CUDA_VISIBLE_DEVICES=1 python train_osd.py \
  --config configs/osd_v2_paper256.json --smoke
```

The smoke path intentionally uses zero workers.  Normal runs use spawned
workers rather than forking after CUDA initialization, so memory and ETA
measurements are not polluted by a forked CUDA context.

The profiles differ only in explicitly recorded training/preprocessing choices:

- `osd_v2_paper256.json`: SegDINO paper-like 256 input, batch 4, 50 epochs.
- `osd_v2_512_dev.json`: OSD 512 development profile, batch 8, 50 epochs.
- `osd_v2_native_512x896.json`: aspect-preserving 512x896 letterbox profile,
  batch 8, with padding excluded from the metric via `ignore_index=255`.

Both use the current v2 decoder, frozen DINOv3-S, AdamW with explicit
`lr=1e-4` and `weight_decay=1e-4`, an explicitly constant scheduler, and
four-class cross entropy.  Input dimensions must be divisible by the DINOv3
patch size (16); the runner fails instead of silently dropping border pixels.
They select
on validation by default.  `selection_split=test` is supported for an
explicit development run, but its output is marked test-tuned in the saved
metadata and must not be presented as an untouched test result.

The default `cudnn_benchmark=false` is deliberate.  With the current v2
decoder, enabling cuDNN benchmarking can select a convolution algorithm with
an unexpectedly large workspace (tens of GiB at the supplied batch sizes).
The runner blocks that switch unless the config explicitly sets
`allow_large_cudnn_workspace=true`; any such speed-profile change must be
re-probed for peak memory before a long run.

The metric implementation in `osd_metrics.py` mirrors the OSD project's
`OSDL4GlobalMetric`: one split-global 4x4 confusion matrix, background kept in
the error terms, and Oil/Water/Others averaged for the primary score.

The first controlled OSD screen is recorded in
`reports/OSD_v2_screen_20260904.md`.  At equal 500-step budgets, 512 square and
512x896 letterbox differed by only 0.15 mIoU points, while the rectangle cost
about 1.7x wall time and memory.  The current recommendation is therefore the
512 square profile for the main run; the rectangular profile remains an
explicit input-strategy ablation.

After a checkpoint/config is frozen, evaluate either split explicitly; this
does not select a checkpoint:

```bash
python evaluate_osd.py --config configs/osd_v2_512_dev.json \
  --checkpoint runs/osd_v2_512_dev/best.pth --split test
```
