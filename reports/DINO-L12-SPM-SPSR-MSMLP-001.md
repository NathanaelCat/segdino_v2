# DINO-L12-SPM-SPSR-MSMLP-001 — preflight

> Status: `preflight_passed`; formal 50-epoch training has **not** started.
> Date: 2026-09-14

## Purpose

This is the first implementation check for the spatial-prior-guided semantic
resampler (SPSR) prototype.  It tests whether RGB Lite-SPM can guide sparse
cross-position sampling and compatibility weighting on the projected frozen
DINOv3-S L12 map while retaining the existing bilinear semantic pyramid as a
zero-initialized residual fallback.

## Fixed prototype

```text
Frozen DINOv3-S/16 -> L12 -> shared 1x1 384->256 -> Z [B,256,32,32]
RGB -> Lite-SPM -> S2 [B,32,256,256], S4 [B,64,128,128], S8 [B,128,64,64]

For P2/P4/P8 (shared K=4 resampler core):
  four fixed +/-0.5 source-token anchors
  learned offset = tanh(f_off(S)) * 1.0 source token
  V_k = grid_sample(Z, source-coordinate grid + anchor + learned offset)
  w_k = softmax(Q(S)^T K(V_k) / sqrt(64))
  R = sum_k w_k V_k
  Y = P + gamma_s * (R - P), gamma_s=0

[Y2, Y4, Y8, Z] -> existing neutral MS-MLP fusion
```

No output projection, DCNv4, MSDA, dense attention matrix, or formal training
was added.

## Preflight result

| Check | Result |
|---|---:|
| Branch / implementation commit | `feat/osd-semantic-spatial-decoupling` / `3e3ba08` |
| Input | `[2,3,512,512]` |
| L12 patch tokens | `[2,1024,384]` |
| Projected source map | `[2,256,32,32]` |
| S2 / S4 / S8 | `[2,32,256,256]` / `[2,64,128,128]` / `[2,128,64,64]` |
| P2 / P4 / P8 / P16 | `[2,256,256,256]` / `[2,256,128,128]` / `[2,256,64,64]` / `[2,256,32,32]` |
| K / source offset limit | `4` / `±1.0` source token |
| Learned offset at init | exactly `0` |
| Weight normalization error | `1.19e-7` max |
| Native logits | `[2,4,256,256]` |
| Final logits | `[2,4,512,512]` |
| Backbone / PatchEmbed frozen | PASS |
| Backbone gradients | none |
| gamma=0 native max abs diff | `0.0` |
| gamma=0 final-logit max abs diff | `0.0` |
| First backward gamma gradients | finite and non-zero for P2/P4/P8 |
| After one optimizer step | gamma leaves zero at all three scales |
| Second backward offset / Q-K gradients | finite and non-zero |
| Forward/backward finite | PASS |
| Formal training / test access | not started / none |

The measured second-backward gradients are small (`offset final weight`
`3.36e-11`, `Q` `2.87e-13`, `K` `6.14e-13`) because the prototype starts
with near-uniform compatibility and gamma has only taken one small optimizer
step.  This is recorded as a learnability observation, not hidden by the
PASS status.  The zero-initialized final offset head means its hidden layer is
expected to receive gradient only after that head has itself moved.

## Resource profile (preflight batch 2)

| Component | Parameters |
|---|---:|
| Lite-SPM stem | 93,472 |
| Shared L12 projection | 98,304 |
| Three spatial projections | 14,336 |
| Shared SPSR core | 25,160 |
| Three gamma scalars | 3 |
| MS-MLP | 526,596 |
| Decoder total | 664,399 |
| Total trainable outside frozen backbone | 757,871 |
| Total model parameters | 22,359,023 |

The hook-based Conv/Linear estimate is `121,332,826,112` FLOPs for batch 2
(`60,666,413,056` per image), excluding the arithmetic cost of
`grid_sample`, softmax, and elementwise operations.  Three K-fold sampling
calls are used per forward.  Peak allocated memory in the batch-2
forward/backward probe was `10,704.69 MiB`; this is a diagnostic batch and is
not a formal training memory claim.

## Decision

The implementation satisfies the requested preflight contract.  It is ready
for research-end review, but **formal 50-epoch training remains blocked until
that review explicitly approves this exact implementation**.

Full machine-readable output:
`work_dirs/dino_l12_spsr_preflight.json`
