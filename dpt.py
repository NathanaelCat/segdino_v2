import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TPASADDecoder(nn.Module):
    def __init__(self, in_dims, decoder_channels=128, num_classes=2, use_group_norm=True):
        super().__init__()
        assert len(in_dims) == 4

        # TPA: project backbone tokens into decoder channels and align them to
        # the four spatial branches used by the decoder.
        self.token_projections = nn.ModuleList(
            [nn.Conv2d(channels, decoder_channels, 1, bias=False) for channels in in_dims]
        )
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
    """
    Final-layer semantic anchoring + pre-transformer spatial recovery.

    L12 is the only semantic anchor (P16). A shared frozen patch projection
    produces an overlapping high-resolution spatial prior that is first reduced
    to decoder width, then resized into decoder P8/P4/P2 scales. The original
    SAD refinement and top-down fusion are preserved.
    """

    def __init__(self, backbone_channels, decoder_channels=128, num_classes=2, use_group_norm=True):
        super().__init__()
        self.semantic_projection = nn.Conv2d(backbone_channels, decoder_channels, 1, bias=False)
        self.spatial_projection = nn.Conv2d(backbone_channels, decoder_channels, 1, bias=False)

        self.p2_project = SpatialScaleProject(decoder_channels)
        self.p4_project = SpatialScaleProject(decoder_channels)
        self.p8_project = SpatialScaleProject(decoder_channels)
        self.p16_project = SpatialScaleProject(decoder_channels)

        # Keep the original SAD organization unchanged.
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
        semantic_map = self._tokens_to_feature_map(semantic_tokens, patch_h, patch_w)
        semantic_map = self.semantic_projection(semantic_map)
        spatial_map = self.spatial_projection(spatial_prior)

        # Define decoder scales from the native P16 semantic grid so the pyramid
        # stays exactly 2x hierarchical even when the raw S4 valid-convolution
        # prior has a non-standard size (e.g. 125x125 for a 512x512 input).
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


class DPT(nn.Module):
    def __init__(
        self,
        encoder_size='base',
        nclass=2,
        decoder_channels=128,
        patch_size=16,
        use_bn=False,
        backbone=None,
        decoder_variant='tpa_sad',
        spatial_stride=4,
        freeze_backbone=True,
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
        self.nclass = nclass
        self.decoder_variant = decoder_variant
        self.spatial_stride = spatial_stride
        self.freeze_backbone = freeze_backbone
        self.in_dims = [self.backbone.embed_dim] * 4

        if self.decoder_variant == 'tpa_sad':
            self.decoder = TPASADDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
                use_group_norm=not use_bn,
            )
        elif self.decoder_variant == 'semantic_spatial':
            if self.spatial_stride <= 0 or self.spatial_stride >= self.patch_size:
                raise ValueError(
                    "semantic_spatial expects 0 < spatial_stride < patch_size; "
                    f"got spatial_stride={self.spatial_stride}, patch_size={self.patch_size}"
                )
            self.decoder = SemanticSpatialSADDecoder(
                backbone_channels=self.backbone.embed_dim,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
                use_group_norm=not use_bn,
            )
        else:
            raise ValueError(
                f"Unknown decoder_variant '{self.decoder_variant}'. "
                "Expected 'tpa_sad' or 'semantic_spatial'."
            )

        if self.freeze_backbone:
            self.lock_backbone()

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _shared_highres_patch_projection(self, x):
        """Reuse the backbone patch projection weights without mutating its stride."""
        patch_embed = getattr(self.backbone, 'patch_embed', None)
        proj = getattr(patch_embed, 'proj', None)
        if proj is None or not isinstance(proj, nn.Conv2d):
            raise TypeError("Expected backbone.patch_embed.proj to be nn.Conv2d")

        # The DINOv3 S16 backbone keeps its original K16/S16 forward untouched.
        # This functional call shares the exact frozen projection parameters but
        # samples them at stride S4 (or the configured spatial_stride).
        return F.conv2d(
            x,
            proj.weight,
            proj.bias,
            stride=(self.spatial_stride, self.spatial_stride),
            padding=proj.padding,
            dilation=proj.dilation,
            groups=proj.groups,
        )

    def forward(self, x, return_feats=False):
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size

        if self.decoder_variant == 'semantic_spatial':
            final_layer_idx = self.intermediate_layer_idx[self.encoder_size][-1]
            feats = self.backbone.get_intermediate_layers(x, n=[final_layer_idx])
            semantic_tokens = feats[-1]
            spatial_prior = self._shared_highres_patch_projection(x)
            out = self.decoder(semantic_tokens, spatial_prior, patch_h, patch_w)
            returned_feature = semantic_tokens
        else:
            feats = self.backbone.get_intermediate_layers(
                x, n=self.intermediate_layer_idx[self.encoder_size]
            )
            out = self.decoder(feats, patch_h, patch_w)
            returned_feature = feats[-1]

        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if return_feats:
            return out, returned_feature
        return out
