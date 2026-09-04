# SegDINO-v2 OSD controlled screen — 2026-09-04

## Scope

This is a development-screen result, not a final test result.  All checkpoint
selection and numbers below use the 203-image OSD validation split.  No test
images were evaluated.

The model is the current SegDINO-v2 release with a frozen DINOv3-S/16
backbone.  Total parameters are 24,901,516; only the 3,300,364 decoder
parameters are trainable.  Both controlled runs use batch 8, four-class cross
entropy, AdamW (`lr=1e-4`, `betas=[0.9,0.999]`, `weight_decay=1e-4`), a
constant learning rate, FP32, seed 20260901, and `cudnn_benchmark=false`.

## Input audit

The local OSD archive contains 811/203/254 train/val/test images.  File-header
inspection found that 1,263 of all 1,268 images are approximately 16:9; the
remaining five training images are 4:3.

Project2 commit `a262e96` records the earlier DINOv3-Mask2Former input fix.  It
changed forced 512x512 crops and `SIZE_DIVISIBILITY=512` to aspect-preserving
short-edge 480, no crop, and `SIZE_DIVISIBILITY=32`; ViT-S AMP was enabled in
the same commit.  That was a dataset/pipeline correction, not a requirement
that DINOv3 itself use 480.

DINOv3 ViT checkpoints use patch size 16 and support varying spatial grids.
Official configurations use different resolutions for different tasks (for
example 256 for the main pretraining recipe, 512 for ADE20K linear training,
and 896 for released Mask2Former ADE20K inference), so there is no single
universal "DINOv3 optimal size".  For this decoder, each input dimension is
required to be divisible by 16.

## Equal-budget 500-step comparison

Both runs evaluate the full validation split every 102 steps.

| profile | best step | L4 mIoU3 | Oil | Water | Others | background | elapsed incl. 5 val | peak allocated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512x512 stretch | 408 | 85.2120 | 82.0441 | 77.3387 | 96.2532 | 16.8163 | 138.64 s | 11,316 MiB |
| 512x896 letterbox | 500 | 85.3641 | 82.0863 | 76.9902 | 97.0157 | 21.4725 | 240.44 s | 19,707 MiB |

The rectangle gains only 0.1521 mIoU points under this short budget, while
using about 1.73x wall time and 1.74x peak allocated memory.  That difference
is too small to claim an accuracy gain from aspect preservation.

## 1,000-step 512 screen

The 512x512 run was repeated from initialization for 1,000 steps.  Its first
408 steps reproduced the 500-step run's validation trajectory.  The best
checkpoint was step 918:

| metric | value |
|---|---:|
| `mIoU3_report_only_global` | 87.426981% |
| Oil IoU | 84.836287% |
| Water IoU | 80.642037% |
| Others IoU | 96.802620% |
| Background IoU | 31.396336% |
| mIoU4 | 73.419320% |
| elapsed incl. 10 full validations | 274.99 s |
| peak allocated memory | 11,316 MiB |

Validation mIoU3 by step was 78.5969, 78.2155, 82.2928, 85.2120, 83.5994,
86.1865, 85.5157, 85.0852, 87.4270, and 84.6427.  The oscillation makes
validation-best checkpointing important, but there is no divergence or
learning-rate anomaly.  The actual logged learning rate remained exactly
`1e-4`.

For context, the existing fully trained SegFormer-B0 seed-20260901 checkpoint
at iteration 38,000 has 91.3700% under the same L4-global validation
calculation.  The 1,000-step SegDINO result is 3.9430 points lower, but it has
only seen about ten epochs and should not be treated as the converged result.

## Decision

Use `configs/osd_v2_512_dev.json` for the first full 50-epoch run.  It is the
better current accuracy/efficiency choice.  Keep
`configs/osd_v2_native_512x896.json` as a controlled input-resolution
ablation, not as the default.  Do not copy Project2's Mask2Former
`weight_decay=0.05` and warmup/poly schedule into this frozen-backbone,
3.3M-parameter SegDINO decoder without a separate ablation.

## Primary references

- DINOv3 pretraining configuration (256 global crop):
  <https://github.com/facebookresearch/dinov3/blob/main/dinov3/configs/train/dinov3_vit7b16_pretrain.yaml>
- DINOv3 ADE20K linear training configuration (512 crop):
  <https://github.com/facebookresearch/dinov3/blob/main/dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml>
- DINOv3 released ADE20K Mask2Former inference configuration (896 crop):
  <https://github.com/facebookresearch/dinov3/blob/main/dinov3/eval/segmentation/configs/config-ade20k-m2f-inference.yaml>
- Current SegDINO-v2 source:
  <https://github.com/script-Yang/segdino_v2>
