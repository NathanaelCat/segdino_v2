import torch
import torch.nn as nn
import torch.nn.functional as F

from ltp_cli import CrossLinearInteraction, LightweightTokenPyramid


PATCH_DPA_VARIANTS = {"patch_dpa_shared", "patch_dpa_independent"}
PATCH_GUIDED_SAD_VARIANTS = {"patch_guided_sad"}
PATCH_PRIOR_VARIANTS = PATCH_DPA_VARIANTS | PATCH_GUIDED_SAD_VARIANTS
LTP_CLI_VARIANTS = {"ltp_cli"}


class L12AnchoredWCF(nn.Module):
    """L12-anchored token/channel selective shallow supplementation.

    The module deliberately stays in token space.  ``fi`` is one auxiliary
    DINO patch-token sequence and ``fl`` is the final-layer patch-token
    sequence.  The two MLPs produce a per-token, per-channel gate; the
    projected auxiliary feature is then injected as a small residual into the
    final-layer anchor.
    """

    def __init__(self, channels: int, reduction: int = 4, alpha_init: float = 1e-2):
        super().__init__()
        if channels <= 0 or reduction <= 0:
            raise ValueError("channels and reduction must be positive")
        hidden = channels // reduction
        if hidden <= 0:
            raise ValueError(f"reduction={reduction} is too large for channels={channels}")

        self.channels = int(channels)
        self.reduction = int(reduction)
        self.channel_mlp = nn.Sequential(
            nn.Linear(4 * channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )
        self.token_mlp = nn.Sequential(
            nn.Linear(2 * channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )
        self.projection = nn.Linear(channels, channels)
        nn.init.zeros_(self.channel_mlp[-1].weight)
        nn.init.zeros_(self.channel_mlp[-1].bias)
        nn.init.zeros_(self.token_mlp[-1].weight)
        nn.init.zeros_(self.token_mlp[-1].bias)
        nn.init.eye_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, fi: torch.Tensor, fl: torch.Tensor, return_gate: bool = False):
        if fi.ndim != 3 or fl.ndim != 3:
            raise ValueError(
                f"WCF expects token tensors [B,N,C], got {tuple(fi.shape)} and {tuple(fl.shape)}"
            )
        if fi.shape != fl.shape:
            raise ValueError(
                f"WCF auxiliary/anchor shapes must match, got {tuple(fi.shape)} and {tuple(fl.shape)}"
            )
        if fi.shape[-1] != self.channels:
            raise ValueError(
                f"WCF expected channel dimension {self.channels}, got {fi.shape[-1]}"
            )

        avg_i = fi.mean(dim=1)
        max_i = fi.max(dim=1).values
        avg_l = fl.mean(dim=1)
        max_l = fl.max(dim=1).values
        lc = self.channel_mlp(torch.cat((avg_i, max_i, avg_l, max_l), dim=-1)).unsqueeze(1)
        lt = self.token_mlp(torch.cat((fi, fl), dim=-1))
        gate = torch.sigmoid(lt + lc)
        delta = gate * self.projection(fi)
        result = fl + self.alpha * delta
        if return_gate:
            return result, gate
        return result


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels, use_group_norm=True):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = (
            nn.GroupNorm(min(32, channels), channels)
            if use_group_norm
            else nn.BatchNorm2d(channels)
        )
        self.act = nn.GELU()
        self.gamma = nn.Parameter(torch.zeros(1)) 

    def forward(self, x):
        residual = self.act(self.norm(self.pointwise(self.depthwise(x))))
        return x + self.gamma * residual


class TPAResampleProject(nn.Module):
    def __init__(self, channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        if self.scale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode="bilinear",
                align_corners=False,
            )
        return self.conv(x)


class SpatialScaleProject(nn.Module):
    """Resize a projected spatial prior to a decoder scale, then refine it."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x, size):
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.conv(x)


class AdaptiveLayerScaleReadout(nn.Module):
    """Adaptive Layer-Scale Readout (ALSR).

    Instead of a fixed 1-to-1 mapping from intermediate DINO layers to decoder
    pyramid scales, ALSR aggregates features across all layers for each scale:
        F_s = sum_{i=0}^{N-1} alpha_{s, i} * P_i(feat_i)
    where alpha_{s, :} = softmax(W_{s, :} / tau).
    """

    def __init__(
        self,
        num_layers: int = 4,
        num_scales: int = 4,
        mode: str = "matrix",
        init_mode: str = "uniform",
        temperature: float = 1.0,
        channels: int = 128,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_scales = num_scales
        self.mode = mode
        self.temperature = float(temperature)

        if mode == "uniform":
            self.register_buffer("weight_logits", torch.zeros(num_scales, num_layers))
        elif mode == "matrix":
            if init_mode == "identity":
                weights = torch.eye(num_scales, num_layers) * 2.0
            else:
                weights = torch.zeros(num_scales, num_layers)
            self.weight_logits = nn.Parameter(weights)
        elif mode == "dynamic":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.mlp = nn.Sequential(
                nn.Linear(num_layers * channels, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_scales * num_layers),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            raise ValueError(f"Unknown ALSR mode: {mode}")

    def get_routing_weights(self, projected_feats=None) -> torch.Tensor:
        if self.mode in ("matrix", "uniform"):
            return F.softmax(self.weight_logits / self.temperature, dim=-1)
        elif self.mode == "dynamic":
            if projected_feats is None:
                raise ValueError("Dynamic ALSR requires projected_feats to compute weights")
            B = projected_feats[0].shape[0]
            pooled = torch.cat([self.pool(f).flatten(1) for f in projected_feats], dim=1)
            logits = self.mlp(pooled).view(B, self.num_scales, self.num_layers)
            return F.softmax(logits / self.temperature, dim=-1)

    def forward(self, projected_feats: list) -> list:
        assert len(projected_feats) == self.num_layers
        weights = self.get_routing_weights(projected_feats)
        stacked = torch.stack(projected_feats, dim=1)

        out_scales = []
        if self.mode in ("matrix", "uniform"):
            for s in range(self.num_scales):
                w_s = weights[s].view(1, self.num_layers, 1, 1, 1)
                f_s = (stacked * w_s).sum(dim=1)
                out_scales.append(f_s)
        else:
            for s in range(self.num_scales):
                w_s = weights[:, s, :].view(-1, self.num_layers, 1, 1, 1)
                f_s = (stacked * w_s).sum(dim=1)
                out_scales.append(f_s)
        return out_scales


class TPASADDecoder(nn.Module):
    def __init__(
        self,
        in_dims,
        decoder_channels=128,
        num_classes=2,
        use_group_norm=True,
        adaptive_readout=False,
        readout_mode="matrix",
        readout_init="uniform",
        readout_temperature=1.0,
    ):
        super().__init__()
        assert len(in_dims) == 4

        # TPA: project backbone tokens into decoder channels and align them to
        # the four spatial branches used by the decoder.
        self.token_projections = nn.ModuleList(
            [nn.Conv2d(channels, decoder_channels, 1, bias=False) for channels in in_dims]
        )
        if adaptive_readout:
            self.alsr = AdaptiveLayerScaleReadout(
                num_layers=len(in_dims),
                num_scales=4,
                mode=readout_mode,
                init_mode=readout_init,
                temperature=readout_temperature,
                channels=decoder_channels,
            )
        else:
            self.alsr = None

        self.tpa_branch_1 = TPAResampleProject(decoder_channels, scale_factor=8)
        self.tpa_branch_2 = TPAResampleProject(decoder_channels, scale_factor=4)
        self.tpa_branch_3 = TPAResampleProject(decoder_channels, scale_factor=2)
        self.tpa_branch_4 = TPAResampleProject(decoder_channels, scale_factor=1)

        # SAD: refine each branch independently, then fuse them from coarse to fine.
        self.sad_intra_1 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_intra_2 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_intra_3 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_intra_4 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)

        self.sad_inter_4 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_inter_3 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_inter_2 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_inter_1 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)

        self.out_conv = nn.Conv2d(decoder_channels, num_classes, 1)

    def _tokens_to_feature_map(self, x, patch_h, patch_w):
        if isinstance(x, (list, tuple)):
            x = x[0]

        num_patches = patch_h * patch_w
        if x.ndim != 3:
            raise ValueError(f"Expected token tensor with 3 dims, got shape {tuple(x.shape)}")
        if x.shape[1] < num_patches:
            raise ValueError(
                f"Token count {x.shape[1]} is smaller than expected patch grid {num_patches}"
            )
        if x.shape[1] != num_patches:
            x = x[:, -num_patches:, :]

        return x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)

    def forward(self, features, patch_h, patch_w):
        # TPA starts here: token sequences become a four-branch feature pyramid.
        branches = []
        for index, tokens in enumerate(features):
            feature_map = self._tokens_to_feature_map(tokens, patch_h, patch_w)
            feature_map = self.token_projections[index](feature_map)
            branches.append(feature_map)

        if self.alsr is not None:
            branches = self.alsr(branches)

        branch_1 = self.tpa_branch_1(branches[0])
        branch_2 = self.tpa_branch_2(branches[1])
        branch_3 = self.tpa_branch_3(branches[2])
        branch_4 = self.tpa_branch_4(branches[3])

        # SAD starts here: each branch is refined, then merged top-down.
        level_1 = self.sad_intra_1(branch_1)
        level_2 = self.sad_intra_2(branch_2)
        level_3 = self.sad_intra_3(branch_3)
        level_4 = self.sad_intra_4(branch_4)

        x4 = self.sad_inter_4(level_4)
        x3_up = F.interpolate(x4, size=level_3.shape[-2:], mode="bilinear", align_corners=False)
        x3 = self.sad_inter_3(x3_up + level_3)

        x2_up = F.interpolate(x3, size=level_2.shape[-2:], mode="bilinear", align_corners=False)
        x2 = self.sad_inter_2(x2_up + level_2)

        x1_up = F.interpolate(x2, size=level_1.shape[-2:], mode="bilinear", align_corners=False)
        x1 = self.sad_inter_1(x1_up + level_1)

        return self.out_conv(x1)


class SemanticSpatialSADDecoder(nn.Module):
    """Use L12 semantics and a shared overlapping patch prior before SAD.

    The semantic path intentionally keeps the current OSD SAD implementation:
    the only new inputs are a projected L12 P16 map and projected/resized
    spatial-prior maps for P8/P4/P2.  The backbone patch projection itself is
    owned by ``DPT.backbone`` and is passed functionally by the caller; this
    module does not register a cloned patch projection parameter.
    """

    def __init__(self, backbone_channels, decoder_channels=128, num_classes=2, use_group_norm=True):
        super().__init__()
        self.semantic_projection = nn.Conv2d(backbone_channels, decoder_channels, 1, bias=False)
        self.spatial_projection = nn.Conv2d(backbone_channels, decoder_channels, 1, bias=False)

        self.p2_project = SpatialScaleProject(decoder_channels)
        self.p4_project = SpatialScaleProject(decoder_channels)
        self.p8_project = SpatialScaleProject(decoder_channels)
        self.p16_project = SpatialScaleProject(decoder_channels)

        # This is the current OSD SAD organization, copied without structural
        # changes: four intra refinements followed by four top-down inter blocks.
        self.sad_intra_1 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_intra_2 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_intra_3 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_intra_4 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)

        self.sad_inter_4 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_inter_3 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_inter_2 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_inter_1 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)

        self.out_conv = nn.Conv2d(decoder_channels, num_classes, 1)

    @staticmethod
    def _tokens_to_feature_map(tokens, patch_h, patch_w):
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]

        num_patches = patch_h * patch_w
        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor with 3 dims, got shape {tuple(tokens.shape)}")
        if tokens.shape[1] < num_patches:
            raise ValueError(
                f"Token count {tokens.shape[1]} is smaller than expected patch grid {num_patches}"
            )
        if tokens.shape[1] != num_patches:
            tokens = tokens[:, -num_patches:, :]

        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w)

    def build_pyramid(self, semantic_tokens, spatial_prior, patch_h, patch_w):
        # L12 is the only semantic source.  The other three scales are derived
        # from the overlapping spatial prior, after projection to decoder width.
        semantic_map = self._tokens_to_feature_map(semantic_tokens, patch_h, patch_w)
        semantic_map = self.semantic_projection(semantic_map)
        spatial_map = self.spatial_projection(spatial_prior)

        p16_size = (patch_h, patch_w)
        p8_size = (patch_h * 2, patch_w * 2)
        p4_size = (patch_h * 4, patch_w * 4)
        p2_size = (patch_h * 8, patch_w * 8)

        p2 = self.p2_project(spatial_map, p2_size)
        p4 = self.p4_project(spatial_map, p4_size)
        p8 = self.p8_project(spatial_map, p8_size)
        p16 = self.p16_project(semantic_map, p16_size)
        return p2, p4, p8, p16

    def forward(self, semantic_tokens, spatial_prior, patch_h, patch_w):
        p2, p4, p8, p16 = self.build_pyramid(
            semantic_tokens,
            spatial_prior,
            patch_h,
            patch_w,
        )

        level_1 = self.sad_intra_1(p2)
        level_2 = self.sad_intra_2(p4)
        level_3 = self.sad_intra_3(p8)
        level_4 = self.sad_intra_4(p16)

        x4 = self.sad_inter_4(level_4)
        x3_up = F.interpolate(x4, size=level_3.shape[-2:], mode="bilinear", align_corners=False)
        x3 = self.sad_inter_3(x3_up + level_3)

        x2_up = F.interpolate(x3, size=level_2.shape[-2:], mode="bilinear", align_corners=False)
        x2 = self.sad_inter_2(x2_up + level_2)

        x1_up = F.interpolate(x2, size=level_1.shape[-2:], mode="bilinear", align_corners=False)
        x1 = self.sad_inter_1(x1_up + level_1)

        return self.out_conv(x1)


class SameScaleMLPDecoder(nn.Module):
    """Minimal same-scale DINO feature fusion for the DINOv3-MLP baseline.

    All four inputs are semantic-depth variants of the same 32x32 patch grid.
    This decoder deliberately contains no resize, pyramid, spatial convolution,
    TPA, SAD, FPN, or transposed convolution.  The only spatial operation in
    the full model is the final output interpolation performed by ``DPT``.
    """

    def __init__(self, backbone_channels, decoder_channels=256, num_classes=4):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(backbone_channels, decoder_channels, 1, bias=True) for _ in range(4)]
        )
        self.fusion = nn.Conv2d(4 * decoder_channels, decoder_channels, 1, bias=True)
        self.activation = nn.GELU()
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1, bias=True)

    @staticmethod
    def _tokens_to_feature_map(tokens, patch_h, patch_w):
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected patch tokens [B,N,C], got shape {tuple(tokens.shape)}"
            )
        expected_tokens = int(patch_h * patch_w)
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                "DINOv3-MLP requires patch-only tokens with exactly one token per "
                f"grid cell: got N={tokens.shape[1]}, expected {expected_tokens}. "
                "CLS/register tokens must be removed before reshape."
            )
        return tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[-1], patch_h, patch_w
        )

    def forward(self, features, patch_h, patch_w):
        if len(features) != 4:
            raise ValueError(f"DINOv3-MLP requires four intermediate features, got {len(features)}")
        maps = [self._tokens_to_feature_map(feature, patch_h, patch_w) for feature in features]
        if any(feature.shape[-2:] != maps[0].shape[-2:] for feature in maps[1:]):
            raise ValueError("DINOv3-MLP received feature maps with different spatial grids")
        projected = [projection(feature) for projection, feature in zip(self.projections, maps)]
        fused = self.activation(self.fusion(torch.cat(projected, dim=1)))
        return self.classifier(fused)


class MultiScaleMLPFusion(nn.Module):
    """Neutral consumer for four already-constructed spatial scales.

    This is the multi-scale extension of ``SameScaleMLPDecoder``: each scale
    is independently projected, all projected maps are aligned to the highest
    available resolution, and only then are they concatenated and fused.  It
    deliberately contains no intra-scale refinement or top-down path, so it
    can be used to isolate the effect of a TPA/DPA feature pyramid.
    """

    def __init__(self, input_channels, decoder_channels=256, num_classes=4):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(input_channels, decoder_channels, 1, bias=True)
                for _ in range(4)
            ]
        )
        self.fusion = nn.Conv2d(4 * decoder_channels, decoder_channels, 1, bias=True)
        self.activation = nn.GELU()
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1, bias=True)

    def forward(self, features):
        if len(features) != 4:
            raise ValueError(f"MultiScaleMLPFusion requires four feature maps, got {len(features)}")
        target_size = features[0].shape[-2:]
        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        aligned = [
            feature
            if feature.shape[-2:] == target_size
            else F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            for feature in projected
        ]
        fused = self.activation(self.fusion(torch.cat(aligned, dim=1)))
        return self.classifier(fused)


class TPAMultiScaleMLPDecoder(nn.Module):
    """Original TPA pyramid followed by the neutral multi-scale MLP head.

    The TPA part is intentionally the existing token projection plus
    scale-specific resample/project path.  The consumer after TPA is only
    projection -> resize-to-highest -> concat -> 1x1 fusion -> GELU ->
    classifier; SAD is not present in this decoder.
    """

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__()
        assert len(in_dims) == 4
        self.token_projections = nn.ModuleList(
            [nn.Conv2d(channels, decoder_channels, 1, bias=False) for channels in in_dims]
        )
        self.tpa_branch_1 = TPAResampleProject(decoder_channels, scale_factor=8)
        self.tpa_branch_2 = TPAResampleProject(decoder_channels, scale_factor=4)
        self.tpa_branch_3 = TPAResampleProject(decoder_channels, scale_factor=2)
        self.tpa_branch_4 = TPAResampleProject(decoder_channels, scale_factor=1)
        self.ms_mlp = MultiScaleMLPFusion(
            input_channels=decoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )

    @staticmethod
    def _tokens_to_feature_map(x, patch_h, patch_w):
        if isinstance(x, (list, tuple)):
            x = x[0]
        num_patches = patch_h * patch_w
        if x.ndim != 3:
            raise ValueError(f"Expected token tensor with 3 dims, got shape {tuple(x.shape)}")
        if x.shape[1] < num_patches:
            raise ValueError(
                f"Token count {x.shape[1]} is smaller than expected patch grid {num_patches}"
            )
        if x.shape[1] != num_patches:
            x = x[:, -num_patches:, :]
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)

    def forward(self, features, patch_h, patch_w):
        if len(features) != 4:
            raise ValueError(f"TPAMultiScaleMLPDecoder requires four features, got {len(features)}")
        projected = self._project_tokens(features, patch_h, patch_w)
        pyramid = self._build_tpa_pyramid(projected)
        return self.ms_mlp(pyramid)

    def _project_tokens(self, features, patch_h, patch_w):
        if len(features) != 4:
            raise ValueError(f"TPAMultiScaleMLPDecoder requires four features, got {len(features)}")
        return [
            projection(self._tokens_to_feature_map(tokens, patch_h, patch_w))
            for projection, tokens in zip(self.token_projections, features)
        ]

    def _build_tpa_pyramid(self, projected):
        if len(projected) != 4:
            raise ValueError(f"TPA pyramid requires four projected maps, got {len(projected)}")
        return (
            self.tpa_branch_1(projected[0]),
            self.tpa_branch_2(projected[1]),
            self.tpa_branch_3(projected[2]),
            self.tpa_branch_4(projected[3]),
        )


class LTPCLIMultiScaleMLPDecoder(TPAMultiScaleMLPDecoder):
    """B1 plus an RGB token pyramid and cross-linear interaction.

    ``TPAMultiScaleMLPDecoder`` is initialized first and inherited unchanged:
    its token projections, PR branches, and MS-MLP are the complete B1 path.
    The new LTP/CLI path is inserted between B1 token projection and B1 PR.
    With all CLI ``gamma`` values at zero, the output is therefore exactly the
    B1 output for the same shared weights and input.
    """

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__(
            in_dims,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )
        if decoder_channels != 256:
            raise ValueError(
                "DINO-LTP-CLI-001 fixes the B1 decoder width and CLI query width at 256; "
                f"got decoder_channels={decoder_channels}"
            )
        self.ltp = LightweightTokenPyramid(
            s4_channels=64,
            s8_channels=96,
            s16_channels=128,
            memory_channels=64,
            num_heads=4,
        )
        self.cli = CrossLinearInteraction(
            query_channels=decoder_channels,
            memory_channels=64,
            interaction_channels=64,
            num_heads=4,
            num_queries=4,
        )

    def _forward_impl(self, features, patch_h, patch_w, image, diagnostics=False):
        if image is None:
            raise ValueError("DINO-LTP-CLI requires the original RGB image")
        projected = self._project_tokens(features, patch_h, patch_w)
        memory, ltp_trace = self.ltp(image, return_trace=True)
        calibrated, cli_trace = self.cli(projected, memory, return_trace=True)
        pyramid = self._build_tpa_pyramid(calibrated)
        logits = self.ms_mlp(pyramid)
        if not diagnostics:
            return logits
        return logits, {
            "projected": tuple(projected),
            "calibrated": tuple(calibrated),
            "pyramid": tuple(pyramid),
            "s4": ltp_trace["s4"],
            "s8": ltp_trace["s8"],
            "s16": ltp_trace["s16"],
            "memory": memory,
            "interaction_outputs": cli_trace["interaction_outputs"],
        }

    def forward(self, features, patch_h, patch_w, image=None):
        return self._forward_impl(features, patch_h, patch_w, image, diagnostics=False)

    def forward_with_diagnostics(self, features, patch_h, patch_w, image=None):
        return self._forward_impl(features, patch_h, patch_w, image, diagnostics=True)


class DPAMultiScaleMLPDecoder(TPAMultiScaleMLPDecoder):
    """DPA inserted between TPA token projection and spatial reconstruction.

    The parent decoder and all of its parameters are inherited unchanged.
    DPA adds only three bias-free 1x1 projections and three zero-initialized
    scalar residual coefficients.  With all coefficients at zero, this path
    reduces exactly to ``TPAMultiScaleMLPDecoder``.
    """

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__(
            in_dims,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )
        self.phi_3 = nn.Conv2d(decoder_channels, decoder_channels, 1, bias=False)
        self.phi_6 = nn.Conv2d(decoder_channels, decoder_channels, 1, bias=False)
        self.phi_9 = nn.Conv2d(decoder_channels, decoder_channels, 1, bias=False)
        self.alpha_3 = nn.Parameter(torch.zeros(()))
        self.alpha_6 = nn.Parameter(torch.zeros(()))
        self.alpha_9 = nn.Parameter(torch.zeros(()))

    def _calibrate_projected(self, projected):
        if len(projected) != 4:
            raise ValueError(f"DPA requires four projected maps, got {len(projected)}")
        x3, x6, x9, g = projected
        calibrated = []
        scores = []
        aligned = []
        for x_i, phi_i, alpha_i in (
            (x3, self.phi_3, self.alpha_3),
            (x6, self.phi_6, self.alpha_6),
            (x9, self.phi_9, self.alpha_9),
        ):
            g_i = phi_i(g)
            s_i = F.cosine_similarity(x_i, g_i, dim=1, eps=1e-6).unsqueeze(1)
            s_i = (s_i + 1.0) * 0.5
            calibrated.append(x_i + alpha_i * s_i * g_i)
            scores.append(s_i)
            aligned.append(g_i)
        calibrated.append(g)
        return calibrated, scores, aligned

    def forward(self, features, patch_h, patch_w):
        projected = self._project_tokens(features, patch_h, patch_w)
        calibrated, _, _ = self._calibrate_projected(projected)
        pyramid = self._build_tpa_pyramid(calibrated)
        return self.ms_mlp(pyramid)

    def forward_with_diagnostics(self, features, patch_h, patch_w):
        """Return the normal logits plus tensors needed by the preflight check."""
        projected = self._project_tokens(features, patch_h, patch_w)
        calibrated, scores, aligned = self._calibrate_projected(projected)
        pyramid = self._build_tpa_pyramid(calibrated)
        logits = self.ms_mlp(pyramid)
        return logits, {
            "projected": tuple(projected),
            "calibrated": tuple(calibrated),
            "aligned_l12": tuple(aligned),
            "scores": tuple(scores),
            "pyramid": tuple(pyramid),
        }


class PatchAdapter(nn.Module):
    """Small patch-local adapter used by the Patch-guided DPA variants."""

    def __init__(self, in_channels=384, out_channels=256):
        super().__init__()
        self.in_projection = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.depthwise = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            groups=out_channels,
            bias=False,
        )
        self.out_projection = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.activation = nn.GELU()

    def forward(self, patch_embedding):
        if patch_embedding.ndim != 4:
            raise ValueError(
                "PatchAdapter expects raw patch embedding [B,C,H,W], got "
                f"{tuple(patch_embedding.shape)}"
            )
        u = self.in_projection(patch_embedding)
        v = self.out_projection(self.depthwise(self.activation(u)))
        return u + v


class PatchGuidedDPADecoder(TPAMultiScaleMLPDecoder):
    """Patch-guided DPA-Cal followed by the unchanged B1 TPA/MS-MLP path.

    ``adapter_sharing='shared'`` creates one patch adapter whose output is
    sent to four independent depth-alignment heads.  ``'independent'`` creates
    four structurally identical but separately parameterized patch adapters.
    In both cases the only signal used to calibrate a DINO depth feature is
    the raw pre-transformer patch projection supplied by ``DPT``.
    """

    DEPTH_NAMES = ("3", "6", "9", "12")

    def __init__(
        self,
        in_dims,
        decoder_channels=256,
        num_classes=4,
        adapter_sharing="shared",
    ):
        super().__init__(
            in_dims,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )
        if len(in_dims) != 4:
            raise ValueError("PatchGuidedDPADecoder requires four depth features")
        if adapter_sharing not in {"shared", "independent"}:
            raise ValueError(
                "adapter_sharing must be 'shared' or 'independent', got "
                f"{adapter_sharing!r}"
            )
        self.adapter_sharing = adapter_sharing
        if adapter_sharing == "shared":
            self.patch_adapter = PatchAdapter(
                in_channels=in_dims[0], out_channels=decoder_channels
            )
        else:
            self.patch_adapters = nn.ModuleList(
                [
                    PatchAdapter(
                        in_channels=in_dims[index],
                        out_channels=decoder_channels,
                    )
                    for index in range(4)
                ]
            )
        self.depth_alignments = nn.ModuleList(
            [
                nn.Conv2d(decoder_channels, decoder_channels, 1, bias=False)
                for _ in range(4)
            ]
        )
        self.alpha_3 = nn.Parameter(torch.zeros(()))
        self.alpha_6 = nn.Parameter(torch.zeros(()))
        self.alpha_9 = nn.Parameter(torch.zeros(()))
        self.alpha_12 = nn.Parameter(torch.zeros(()))

    @property
    def alphas(self):
        return (self.alpha_3, self.alpha_6, self.alpha_9, self.alpha_12)

    def _patch_aligned_features(self, patch_embedding, target_size):
        if patch_embedding.ndim != 4:
            raise ValueError(
                "Patch-guided DPA requires raw patch embedding [B,384,H,W], got "
                f"{tuple(patch_embedding.shape)}"
            )
        if self.adapter_sharing == "shared":
            shared_prior = self.patch_adapter(patch_embedding)
            priors = [shared_prior] * 4
        else:
            priors = [adapter(patch_embedding) for adapter in self.patch_adapters]
        # B2PS-S4/B2PI-S4 are deliberately a control: the S4 prior is first
        # reduced to the DINO token grid before the original 32x32 DPA is
        # evaluated.  The native prior is retained in the trace so that the
        # information bottleneck is explicit in the experiment record.
        aligned_priors = [
            prior
            if prior.shape[-2:] == target_size
            else F.interpolate(prior, size=target_size, mode="bilinear", align_corners=False)
            for prior in priors
        ]
        aligned = [
            projection(prior)
            for projection, prior in zip(self.depth_alignments, aligned_priors)
        ]
        return priors, aligned_priors, aligned

    def _calibrate_projected(self, projected, patch_embedding):
        if len(projected) != 4:
            raise ValueError(
                f"Patch-guided DPA requires four projected maps, got {len(projected)}"
            )
        target_size = projected[0].shape[-2:]
        priors, aligned_priors, aligned = self._patch_aligned_features(
            patch_embedding, target_size
        )
        calibrated = []
        scores = []
        cosines = []
        residuals = []
        for x_i, q_i, alpha_i in zip(projected, aligned, self.alphas):
            cosine = F.cosine_similarity(x_i, q_i, dim=1, eps=1e-6)
            score = ((cosine + 1.0) * 0.5).unsqueeze(1)
            residual = alpha_i * score * q_i
            calibrated.append(x_i + residual)
            scores.append(score)
            cosines.append(cosine)
            residuals.append(residual)
        return calibrated, priors, aligned_priors, aligned, scores, cosines, residuals

    def forward(self, features, patch_h, patch_w, patch_embedding=None):
        if patch_embedding is None:
            raise ValueError("Patch-guided DPA requires raw patch embedding input")
        projected = self._project_tokens(features, patch_h, patch_w)
        calibrated, _, _, _, _, _, _ = self._calibrate_projected(
            projected, patch_embedding
        )
        pyramid = self._build_tpa_pyramid(calibrated)
        return self.ms_mlp(pyramid)

    def forward_with_diagnostics(self, features, patch_h, patch_w, patch_embedding=None):
        if patch_embedding is None:
            raise ValueError("Patch-guided DPA requires raw patch embedding input")
        projected = self._project_tokens(features, patch_h, patch_w)
        calibrated, priors, aligned_priors, aligned, scores, cosines, residuals = self._calibrate_projected(
            projected, patch_embedding
        )
        pyramid = self._build_tpa_pyramid(calibrated)
        logits = self.ms_mlp(pyramid)
        return logits, {
            "projected": tuple(projected),
            "calibrated": tuple(calibrated),
            "patch_priors": tuple(priors),
            "resized_patch_priors": tuple(aligned_priors),
            "aligned_patch_priors": tuple(aligned),
            "scores": tuple(scores),
            "cosines": tuple(cosines),
            "residuals": tuple(residuals),
            "pyramid": tuple(pyramid),
        }


class PatchGuidedResidualBlock(nn.Module):
    """Replace one SAD residual block with patch-guided signed calibration.

    The block deliberately has no normalization, activation, depthwise
    refinement, or attention.  It uses the same cosine-gated residual idea as
    the validated patch-DPA path, but consumes a patch prior already aligned
    to the current SAD scale.  With ``alpha=0`` it is an exact identity.
    """

    def __init__(self, channels, alpha_init=0.0):
        super().__init__()
        self.phi = nn.Conv2d(channels, channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x, patch, return_trace=False):
        if x.ndim != 4 or patch.ndim != 4:
            raise ValueError(
                "PatchGuidedResidualBlock expects feature and patch maps [B,C,H,W], "
                f"got {tuple(x.shape)} and {tuple(patch.shape)}"
            )
        if x.shape != patch.shape:
            raise ValueError(
                "PatchGuidedResidualBlock requires exact feature/patch alignment, "
                f"got {tuple(x.shape)} and {tuple(patch.shape)}"
            )
        q = self.phi(patch)
        cosine = F.cosine_similarity(x, q, dim=1, eps=1e-6)
        score = ((cosine + 1.0) * 0.5).unsqueeze(1)
        residual = self.alpha * score * q
        output = x + residual
        if return_trace:
            return output, {
                "aligned_patch": patch,
                "q": q,
                "score": score,
                "cosine": cosine,
                "residual": residual,
                "input": x,
            }
        return output


class PatchGuidedSADDecoder(nn.Module):
    """TPA PR followed by SAD with every R replaced by PatchGuidedR.

    The four TPA branches are copied from the current OSD PR exactly.  A raw
    frozen K16 patch projection is adapted once at its native grid, then
    aligned independently to P2/P4/P8/P16.  Both the four intra-scale and
    four top-down SAD R locations consume the patch prior at their own scale;
    no S4 prior is first collapsed to 32x32 in this decoder.
    """

    SCALE_NAMES = ("P16", "P8", "P4", "P2")

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__()
        if len(in_dims) != 4 or len(set(in_dims)) != 1:
            raise ValueError(
                f"PatchGuidedSADDecoder requires four equal input dims, got {in_dims}"
            )
        self.in_dims = list(in_dims)
        self.decoder_channels = int(decoder_channels)
        self.token_projections = nn.ModuleList(
            [nn.Conv2d(channels, decoder_channels, 1, bias=False) for channels in in_dims]
        )
        self.tpa_branch_1 = TPAResampleProject(decoder_channels, scale_factor=8)
        self.tpa_branch_2 = TPAResampleProject(decoder_channels, scale_factor=4)
        self.tpa_branch_3 = TPAResampleProject(decoder_channels, scale_factor=2)
        self.tpa_branch_4 = TPAResampleProject(decoder_channels, scale_factor=1)

        # One shared local adapter keeps the patch source definition fixed;
        # scale-specific 1x1 alignments provide independent latent spaces.
        self.patch_adapter = PatchAdapter(
            in_channels=in_dims[0], out_channels=decoder_channels
        )
        self.patch_scale_align = nn.ModuleDict(
            {
                name: nn.Conv2d(decoder_channels, decoder_channels, 1, bias=False)
                for name in ("P2", "P4", "P8", "P16")
            }
        )

        # These eight modules occupy exactly the former SAD intra/inter R
        # locations.  They are identities at initialization and contain no
        # ordinary SAD refinement path.
        self.patch_intra_p2 = PatchGuidedResidualBlock(decoder_channels)
        self.patch_intra_p4 = PatchGuidedResidualBlock(decoder_channels)
        self.patch_intra_p8 = PatchGuidedResidualBlock(decoder_channels)
        self.patch_intra_p16 = PatchGuidedResidualBlock(decoder_channels)
        self.patch_inter_p16 = PatchGuidedResidualBlock(decoder_channels)
        self.patch_inter_p8 = PatchGuidedResidualBlock(decoder_channels)
        self.patch_inter_p4 = PatchGuidedResidualBlock(decoder_channels)
        self.patch_inter_p2 = PatchGuidedResidualBlock(decoder_channels)

        self.out_conv = nn.Conv2d(decoder_channels, num_classes, 1)

    @staticmethod
    def _tokens_to_feature_map(x, patch_h, patch_w):
        if isinstance(x, (list, tuple)):
            x = x[0]
        num_patches = patch_h * patch_w
        if x.ndim != 3:
            raise ValueError(f"Expected token tensor with 3 dims, got shape {tuple(x.shape)}")
        if x.shape[1] < num_patches:
            raise ValueError(
                f"Token count {x.shape[1]} is smaller than expected patch grid {num_patches}"
            )
        if x.shape[1] != num_patches:
            x = x[:, -num_patches:, :]
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)

    def _project_tokens(self, features, patch_h, patch_w):
        if len(features) != 4:
            raise ValueError(f"PatchGuidedSADDecoder requires four features, got {len(features)}")
        return [
            projection(self._tokens_to_feature_map(tokens, patch_h, patch_w))
            for projection, tokens in zip(self.token_projections, features)
        ]

    def _build_tpa_pyramid(self, projected):
        if len(projected) != 4:
            raise ValueError(f"TPA pyramid requires four projected maps, got {len(projected)}")
        return (
            self.tpa_branch_1(projected[0]),
            self.tpa_branch_2(projected[1]),
            self.tpa_branch_3(projected[2]),
            self.tpa_branch_4(projected[3]),
        )

    def _build_patch_pyramid(self, raw_patch, patch_h, patch_w):
        if raw_patch.ndim != 4:
            raise ValueError(
                "PatchGuidedSADDecoder requires raw patch map [B,C,H,W], got "
                f"{tuple(raw_patch.shape)}"
            )
        native = self.patch_adapter(raw_patch)
        sizes = {
            "P2": (patch_h * 8, patch_w * 8),
            "P4": (patch_h * 4, patch_w * 4),
            "P8": (patch_h * 2, patch_w * 2),
            "P16": (patch_h, patch_w),
        }
        pyramid = {}
        for name, size in sizes.items():
            aligned = native
            if aligned.shape[-2:] != size:
                aligned = F.interpolate(
                    aligned, size=size, mode="bilinear", align_corners=False
                )
            pyramid[name] = self.patch_scale_align[name](aligned)
        return pyramid, native

    @property
    def alphas(self):
        return tuple(module.alpha for _, module in self.named_patch_blocks())

    def named_patch_blocks(self):
        return (
            ("alpha_intra_P2", self.patch_intra_p2),
            ("alpha_intra_P4", self.patch_intra_p4),
            ("alpha_intra_P8", self.patch_intra_p8),
            ("alpha_intra_P16", self.patch_intra_p16),
            ("alpha_inter_P16", self.patch_inter_p16),
            ("alpha_inter_P8", self.patch_inter_p8),
            ("alpha_inter_P4", self.patch_inter_p4),
            ("alpha_inter_P2", self.patch_inter_p2),
        )

    def _forward_impl(self, features, raw_patch, patch_h, patch_w, diagnostics=False):
        projected = self._project_tokens(features, patch_h, patch_w)
        pyramid = self._build_tpa_pyramid(projected)
        patch_pyramid, native_patch = self._build_patch_pyramid(
            raw_patch, patch_h, patch_w
        )
        p2, p4, p8, p16 = pyramid
        q2, q4, q8, q16 = (
            patch_pyramid["P2"],
            patch_pyramid["P4"],
            patch_pyramid["P8"],
            patch_pyramid["P16"],
        )

        level2, intra2 = self.patch_intra_p2(p2, q2, return_trace=True)
        level4, intra4 = self.patch_intra_p4(p4, q4, return_trace=True)
        level8, intra8 = self.patch_intra_p8(p8, q8, return_trace=True)
        level16, intra16 = self.patch_intra_p16(p16, q16, return_trace=True)

        x16, inter16 = self.patch_inter_p16(level16, q16, return_trace=True)
        x8_input = F.interpolate(
            x16, size=level8.shape[-2:], mode="bilinear", align_corners=False
        ) + level8
        x8, inter8 = self.patch_inter_p8(x8_input, q8, return_trace=True)
        x4_input = F.interpolate(
            x8, size=level4.shape[-2:], mode="bilinear", align_corners=False
        ) + level4
        x4, inter4 = self.patch_inter_p4(x4_input, q4, return_trace=True)
        x2_input = F.interpolate(
            x4, size=level2.shape[-2:], mode="bilinear", align_corners=False
        ) + level2
        x2, inter2 = self.patch_inter_p2(x2_input, q2, return_trace=True)
        logits = self.out_conv(x2)
        if not diagnostics:
            return logits

        # Coarse-to-fine order makes the diagnostic table match the decoder
        # equations: P16 -> P8 -> P4 -> P2.
        inter_trace = (inter16, inter8, inter4, inter2)
        intra_trace = (intra16, intra8, intra4, intra2)
        return logits, {
            "projected": tuple(trace["input"] for trace in inter_trace),
            "patch_priors": tuple(patch_pyramid[name] for name in self.SCALE_NAMES),
            "native_patch": native_patch,
            "scores": tuple(trace["score"] for trace in inter_trace),
            "cosines": tuple(trace["cosine"] for trace in inter_trace),
            "residuals": tuple(trace["residual"] for trace in inter_trace),
            "intra_scores": tuple(trace["score"] for trace in intra_trace),
            "intra_residuals": tuple(trace["residual"] for trace in intra_trace),
            "pyramid": tuple(pyramid),
        }

    def forward(self, features, patch_h, patch_w, patch_embedding=None):
        if patch_embedding is None:
            raise ValueError("PatchGuidedSADDecoder requires raw patch embedding input")
        return self._forward_impl(features, patch_embedding, patch_h, patch_w, diagnostics=False)

    def forward_with_diagnostics(self, features, patch_h, patch_w, patch_embedding=None):
        if patch_embedding is None:
            raise ValueError("PatchGuidedSADDecoder requires raw patch embedding input")
        return self._forward_impl(features, patch_embedding, patch_h, patch_w, diagnostics=True)


class TPASADBaseDecoder(nn.Module):
    """TPA pyramid followed by the minimal progressive top-down consumer.

    This is the neutral decoder node for the decoder-side factorial study.
    ``TPAResampleProject`` is kept exactly as the existing PR implementation;
    the consumer has only progressive bilinear upsample-and-add operations
    followed by a classifier.  It deliberately has no SAD residual blocks,
    normalization, attention, gates, or intra-scale refinement.
    """

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__()
        if len(in_dims) != 4:
            raise ValueError(f"TPASADBaseDecoder requires four input dims, got {len(in_dims)}")
        self.token_projections = nn.ModuleList(
            [nn.Conv2d(channels, decoder_channels, 1, bias=False) for channels in in_dims]
        )
        self.tpa_branch_1 = TPAResampleProject(decoder_channels, scale_factor=8)
        self.tpa_branch_2 = TPAResampleProject(decoder_channels, scale_factor=4)
        self.tpa_branch_3 = TPAResampleProject(decoder_channels, scale_factor=2)
        self.tpa_branch_4 = TPAResampleProject(decoder_channels, scale_factor=1)
        self.out_conv = nn.Conv2d(decoder_channels, num_classes, 1)

    @staticmethod
    def _tokens_to_feature_map(x, patch_h, patch_w):
        if isinstance(x, (list, tuple)):
            x = x[0]
        num_patches = patch_h * patch_w
        if x.ndim != 3:
            raise ValueError(f"Expected token tensor with 3 dims, got shape {tuple(x.shape)}")
        if x.shape[1] < num_patches:
            raise ValueError(
                f"Token count {x.shape[1]} is smaller than expected patch grid {num_patches}"
            )
        if x.shape[1] != num_patches:
            x = x[:, -num_patches:, :]
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)

    def _project_tokens(self, features, patch_h, patch_w):
        if len(features) != 4:
            raise ValueError(f"TPASADBaseDecoder requires four features, got {len(features)}")
        return [
            projection(self._tokens_to_feature_map(tokens, patch_h, patch_w))
            for projection, tokens in zip(self.token_projections, features)
        ]

    def _build_tpa_pyramid(self, projected):
        if len(projected) != 4:
            raise ValueError(f"TPA pyramid requires four projected maps, got {len(projected)}")
        return (
            self.tpa_branch_1(projected[0]),
            self.tpa_branch_2(projected[1]),
            self.tpa_branch_3(projected[2]),
            self.tpa_branch_4(projected[3]),
        )

    @staticmethod
    def _progressive_top_down(pyramid):
        if len(pyramid) != 4:
            raise ValueError(f"SAD-Base requires four pyramid levels, got {len(pyramid)}")
        p2, p4, p8, p16 = pyramid
        x = p16
        x = F.interpolate(x, size=p8.shape[-2:], mode="bilinear", align_corners=False) + p8
        x = F.interpolate(x, size=p4.shape[-2:], mode="bilinear", align_corners=False) + p4
        x = F.interpolate(x, size=p2.shape[-2:], mode="bilinear", align_corners=False) + p2
        return x

    def forward(self, features, patch_h, patch_w):
        projected = self._project_tokens(features, patch_h, patch_w)
        pyramid = self._build_tpa_pyramid(projected)
        return self.out_conv(self._progressive_top_down(pyramid))


class DPASADBaseDecoder(TPASADBaseDecoder):
    """SAD-Base consumer with the already validated DPA calibration path."""

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__(
            in_dims,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )
        self.phi_3 = nn.Conv2d(decoder_channels, decoder_channels, 1, bias=False)
        self.phi_6 = nn.Conv2d(decoder_channels, decoder_channels, 1, bias=False)
        self.phi_9 = nn.Conv2d(decoder_channels, decoder_channels, 1, bias=False)
        self.alpha_3 = nn.Parameter(torch.zeros(()))
        self.alpha_6 = nn.Parameter(torch.zeros(()))
        self.alpha_9 = nn.Parameter(torch.zeros(()))

    def _calibrate_projected(self, projected):
        if len(projected) != 4:
            raise ValueError(f"DPA requires four projected maps, got {len(projected)}")
        x3, x6, x9, g = projected
        calibrated = []
        scores = []
        aligned = []
        for x_i, phi_i, alpha_i in (
            (x3, self.phi_3, self.alpha_3),
            (x6, self.phi_6, self.alpha_6),
            (x9, self.phi_9, self.alpha_9),
        ):
            g_i = phi_i(g)
            s_i = F.cosine_similarity(x_i, g_i, dim=1, eps=1e-6).unsqueeze(1)
            s_i = (s_i + 1.0) * 0.5
            calibrated.append(x_i + alpha_i * s_i * g_i)
            scores.append(s_i)
            aligned.append(g_i)
        calibrated.append(g)
        return calibrated, scores, aligned

    def forward(self, features, patch_h, patch_w):
        projected = self._project_tokens(features, patch_h, patch_w)
        calibrated, _, _ = self._calibrate_projected(projected)
        pyramid = self._build_tpa_pyramid(calibrated)
        return self.out_conv(self._progressive_top_down(pyramid))

    def forward_with_diagnostics(self, features, patch_h, patch_w):
        projected = self._project_tokens(features, patch_h, patch_w)
        calibrated, scores, aligned = self._calibrate_projected(projected)
        pyramid = self._build_tpa_pyramid(calibrated)
        logits = self.out_conv(self._progressive_top_down(pyramid))
        return logits, {
            "projected": tuple(projected),
            "calibrated": tuple(calibrated),
            "aligned_l12": tuple(aligned),
            "scores": tuple(scores),
            "pyramid": tuple(pyramid),
        }


class DPT(nn.Module):
    def __init__(
        self,
        encoder_size='base',
        nclass=2,
        decoder_channels=128,
        patch_size=16,
        use_bn=False,
        backbone=None,
        layer_mapping=None,
        adaptive_readout=False,
        readout_mode="matrix",
        readout_init="uniform",
        readout_temperature=1.0,
        wcf_enabled=False,
        wcf_reduction=4,
        wcf_alpha_init=1e-2,
        decoder_variant="tpa_sad",
        spatial_stride=4,
    ):
        super(DPT, self).__init__()

        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
        }

        self.encoder_size = encoder_size
        self.patch_size = patch_size
        self.backbone = backbone
        self.layer_mapping = list(layer_mapping) if layer_mapping is not None else None
        self.wcf_enabled = bool(wcf_enabled)
        if self.wcf_enabled and self.layer_mapping not in (None, [0, 1, 2, 3]):
            raise ValueError(
                "WCF requires the four native intermediate features [L3,L6,L9,L12]; "
                "use layer_mapping=null or [0,1,2,3]"
            )
        self._backbone_locked = False
        self.nclass = nclass
        self.decoder_variant = str(decoder_variant)
        self.spatial_stride = int(spatial_stride)
        if self.decoder_variant not in {
            "tpa_sad",
            "tpa_ms_mlp",
            "dpa_ms_mlp",
            "tpa_sad_base",
            "dpa_sad_base",
            "patch_dpa_shared",
            "patch_dpa_independent",
            "patch_guided_sad",
            "ltp_cli",
            "semantic_spatial",
            "mlp_same_scale",
        }:
            raise ValueError(
                f"Unknown decoder_variant '{self.decoder_variant}'. "
                "Expected 'tpa_sad', 'tpa_ms_mlp', 'dpa_ms_mlp', "
                "'tpa_sad_base', 'dpa_sad_base', 'patch_dpa_shared', "
                "'patch_dpa_independent', 'patch_guided_sad', 'ltp_cli', 'semantic_spatial', or "
                "'mlp_same_scale'."
            )
        if self.decoder_variant in {"semantic_spatial", "mlp_same_scale"}:
            if self.layer_mapping is not None:
                if self.decoder_variant == "semantic_spatial":
                    raise ValueError(
                        "semantic_spatial requires layer_mapping=null: its semantic source is L12 only"
                    )
                if (
                    len(self.layer_mapping) != 4
                    or any(index < 0 or index >= 4 for index in self.layer_mapping)
                ):
                    raise ValueError(
                        "mlp_same_scale layer_mapping must contain four indices in [0, 3]"
                    )
            if self.wcf_enabled or adaptive_readout:
                raise ValueError(
                    f"{self.decoder_variant} cannot combine WCF or ALSR"
                )
        if self.decoder_variant == "dpa_ms_mlp":
            if self.layer_mapping is not None:
                raise ValueError(
                    "dpa_ms_mlp requires layer_mapping=null: use native [L3,L6,L9,L12]"
                )
            if self.wcf_enabled or adaptive_readout:
                raise ValueError("dpa_ms_mlp cannot combine WCF or ALSR")
        if self.decoder_variant == "dpa_sad_base":
            if self.layer_mapping is not None:
                raise ValueError(
                    "dpa_sad_base requires layer_mapping=null: use native [L3,L6,L9,L12]"
                )
            if self.wcf_enabled or adaptive_readout:
                raise ValueError("dpa_sad_base cannot combine WCF or ALSR")
        if self.decoder_variant in LTP_CLI_VARIANTS:
            if self.layer_mapping is not None:
                raise ValueError(
                    "ltp_cli requires layer_mapping=null: use native [L3,L6,L9,L12]"
                )
            if self.wcf_enabled or adaptive_readout:
                raise ValueError("ltp_cli cannot combine WCF or ALSR")
        if self.decoder_variant in PATCH_PRIOR_VARIANTS:
            if self.layer_mapping is not None:
                raise ValueError(
                    f"{self.decoder_variant} requires layer_mapping=null: use native [L3,L6,L9,L12]"
                )
            if self.wcf_enabled or adaptive_readout:
                raise ValueError(f"{self.decoder_variant} cannot combine WCF or ALSR")
            if self.spatial_stride <= 0 or self.spatial_stride > self.patch_size:
                raise ValueError(
                    f"{self.decoder_variant} requires 0 < spatial_stride <= patch_size; "
                    f"got spatial_stride={self.spatial_stride}, patch_size={self.patch_size}"
                )
        if self.decoder_variant == "semantic_spatial" and (
            self.spatial_stride <= 0 or self.spatial_stride >= self.patch_size
        ):
            raise ValueError(
                "semantic_spatial expects 0 < spatial_stride < patch_size; "
                f"got spatial_stride={self.spatial_stride}, patch_size={self.patch_size}"
            )
        self.in_dims = [self.backbone.embed_dim] * 4
        self.wcf_blocks = nn.ModuleList(
            [
                L12AnchoredWCF(
                    self.backbone.embed_dim,
                    reduction=wcf_reduction,
                    alpha_init=wcf_alpha_init,
                )
                for _ in range(3)
            ]
            if self.wcf_enabled
            else []
        )
        if self.decoder_variant == "tpa_sad":
            self.decoder = TPASADDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
                use_group_norm=not use_bn,
                adaptive_readout=adaptive_readout,
                readout_mode=readout_mode,
                readout_init=readout_init,
                readout_temperature=readout_temperature,
            )
        elif self.decoder_variant == "tpa_ms_mlp":
            self.decoder = TPAMultiScaleMLPDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "dpa_ms_mlp":
            self.decoder = DPAMultiScaleMLPDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "tpa_sad_base":
            self.decoder = TPASADBaseDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "dpa_sad_base":
            self.decoder = DPASADBaseDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "patch_dpa_shared":
            self.decoder = PatchGuidedDPADecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
                adapter_sharing="shared",
            )
        elif self.decoder_variant == "patch_dpa_independent":
            self.decoder = PatchGuidedDPADecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
                adapter_sharing="independent",
            )
        elif self.decoder_variant == "patch_guided_sad":
            self.decoder = PatchGuidedSADDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "ltp_cli":
            self.decoder = LTPCLIMultiScaleMLPDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "semantic_spatial":
            self.decoder = SemanticSpatialSADDecoder(
                backbone_channels=self.backbone.embed_dim,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
                use_group_norm=not use_bn,
            )
        else:
            self.decoder = SameScaleMLPDecoder(
                backbone_channels=self.backbone.embed_dim,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )

    def get_routing_weights(self):
        if hasattr(self.decoder, "alsr") and self.decoder.alsr is not None:
            return self.decoder.alsr.get_routing_weights()
        return None

    def get_wcf_blocks(self):
        return self.wcf_blocks

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._backbone_locked = True
        self.backbone.eval()

    def _shared_highres_patch_projection(self, x):
        """Reuse the frozen K16 patch projection with functional stride S4."""
        patch_embed = getattr(self.backbone, "patch_embed", None)
        proj = getattr(patch_embed, "proj", None)
        if proj is None or not isinstance(proj, nn.Conv2d):
            raise TypeError("Expected backbone.patch_embed.proj to be nn.Conv2d")
        if tuple(proj.kernel_size) != (self.patch_size, self.patch_size):
            raise ValueError(
                "semantic_spatial requires the backbone patch projection kernel to match "
                f"patch_size={self.patch_size}, got {proj.kernel_size}"
            )

        # Do not mutate proj.stride and do not create/register another Parameter.
        # The same frozen weight and bias are passed directly to F.conv2d.
        return F.conv2d(
            x,
            proj.weight,
            proj.bias,
            stride=(self.spatial_stride, self.spatial_stride),
            padding=proj.padding,
            dilation=proj.dilation,
            groups=proj.groups,
        )

    def _raw_patch_embedding(self, x):
        """Return the frozen K16 patch projection before the ViT.

        DINOv3's ``PatchEmbed`` normally continues with flattening and its
        patch norm.  Patch-guided variants intentionally stop at the raw
        ``patch_embed.proj`` output.  ``spatial_stride`` is functional: it
        changes the sampling stride while reusing the exact frozen K16
        weight/bias and never registering a second projection parameter.
        """
        patch_embed = getattr(self.backbone, "patch_embed", None)
        proj = getattr(patch_embed, "proj", None)
        if proj is None or not isinstance(proj, nn.Conv2d):
            raise TypeError("Expected backbone.patch_embed.proj to be nn.Conv2d")
        expected_kernel = (self.patch_size, self.patch_size)
        if tuple(proj.kernel_size) != expected_kernel:
            raise ValueError(
                "Patch-guided variants require the original K16 patch kernel; "
                f"got kernel={proj.kernel_size}, expected={expected_kernel}"
            )
        stride = (self.spatial_stride, self.spatial_stride)
        return F.conv2d(
            x,
            proj.weight,
            proj.bias,
            stride=stride,
            padding=proj.padding,
            dilation=proj.dilation,
            groups=proj.groups,
        )

    def train(self, mode=True):
        """Keep a deliberately frozen DINOv3 backbone in eval mode.

        ``nn.Module.train()`` recursively switches every child to training
        mode.  That is easy to miss when the encoder is frozen and can still
        update running-statistics in a backbone variant that contains them.
        Keeping the state explicit makes the paper-style frozen-backbone
        profile reproducible and avoids unnecessary activation bookkeeping.
        """
        super().train(mode)
        if self._backbone_locked:
            self.backbone.eval()
        return self

    def forward(self, x, return_feats=False, return_wcf_gates=False):
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        patch_embedding = None
        if self.decoder_variant == "semantic_spatial":
            final_layer_idx = self.intermediate_layer_idx[self.encoder_size][-1]
            if self._backbone_locked:
                with torch.no_grad():
                    feats = self.backbone.get_intermediate_layers(x, n=[final_layer_idx])
                    spatial_prior = self._shared_highres_patch_projection(x)
            else:
                feats = self.backbone.get_intermediate_layers(x, n=[final_layer_idx])
                spatial_prior = self._shared_highres_patch_projection(x)
            semantic_tokens = feats[-1]
            out = self.decoder(semantic_tokens, spatial_prior, patch_h, patch_w)
            returned_feature = semantic_tokens
        elif self._backbone_locked:
            with torch.no_grad():
                feats = self.backbone.get_intermediate_layers(
                    x, n=self.intermediate_layer_idx[self.encoder_size]
                )
                if self.decoder_variant in PATCH_PRIOR_VARIANTS:
                    patch_embedding = self._raw_patch_embedding(x)
        else:
            feats = self.backbone.get_intermediate_layers(
                x, n=self.intermediate_layer_idx[self.encoder_size]
            )
            if self.decoder_variant in PATCH_PRIOR_VARIANTS:
                patch_embedding = self._raw_patch_embedding(x)

        if self.decoder_variant != "semantic_spatial":
            wcf_gates = []
            if self.wcf_enabled:
                anchor = feats[-1]
                supplemented = []
                for block, auxiliary in zip(self.wcf_blocks, feats[:3]):
                    result, gate = block(auxiliary, anchor, return_gate=True)
                    supplemented.append(result)
                    wcf_gates.append(gate)
                feats = supplemented + [anchor]
            elif self.layer_mapping is not None:
                feats = [feats[i] for i in self.layer_mapping]

            if self.decoder_variant in PATCH_PRIOR_VARIANTS:
                out = self.decoder(feats, patch_h, patch_w, patch_embedding)
            elif self.decoder_variant in LTP_CLI_VARIANTS:
                out = self.decoder(feats, patch_h, patch_w, x)
            else:
                out = self.decoder(feats, patch_h, patch_w)
            returned_feature = feats[-1]
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if return_wcf_gates:
            return out, returned_feature, tuple(wcf_gates) if self.decoder_variant != "semantic_spatial" else tuple()
        if return_feats:
            return out, returned_feature
        return out

    def dpa_diagnostics(self, x):
        """Run the DPA path and return logits plus detached-path diagnostics.

        This is an observation-only API for the OSD runner.  The ordinary
        ``forward`` path and the trained module topology remain unchanged.
        Callers are expected to put the call under ``no_grad``/inference mode
        when collecting per-epoch statistics.
        """
        if self.decoder_variant not in {
            "dpa_ms_mlp",
            "dpa_sad_base",
            "patch_dpa_shared",
            "patch_dpa_independent",
            "patch_guided_sad",
        }:
            raise RuntimeError(
                "dpa_diagnostics is only available for a DPA decoder variant"
            )
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        if self._backbone_locked:
            with torch.no_grad():
                features = self.backbone.get_intermediate_layers(
                    x, n=self.intermediate_layer_idx[self.encoder_size]
                )
                patch_embedding = (
                    self._raw_patch_embedding(x)
                    if self.decoder_variant in PATCH_PRIOR_VARIANTS
                    else None
                )
        else:
            features = self.backbone.get_intermediate_layers(
                x, n=self.intermediate_layer_idx[self.encoder_size]
            )
            patch_embedding = (
                self._raw_patch_embedding(x)
                if self.decoder_variant in PATCH_PRIOR_VARIANTS
                else None
            )
        if self.decoder_variant in PATCH_PRIOR_VARIANTS:
            lowres_logits, trace = self.decoder.forward_with_diagnostics(
                features, patch_h, patch_w, patch_embedding
            )
        else:
            lowres_logits, trace = self.decoder.forward_with_diagnostics(
                features, patch_h, patch_w
            )
        logits = F.interpolate(
            lowres_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits, trace
