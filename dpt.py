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
        self._backbone_locked = False
        self.nclass = nclass
        self.in_dims = [self.backbone.embed_dim] * 4
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

    def get_routing_weights(self):
        if hasattr(self.decoder, "alsr") and self.decoder.alsr is not None:
            return self.decoder.alsr.get_routing_weights()
        return None

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._backbone_locked = True
        self.backbone.eval()

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

    def forward(self, x, return_feats=False):
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        if self._backbone_locked:
            with torch.no_grad():
                feats = self.backbone.get_intermediate_layers(
                    x, n=self.intermediate_layer_idx[self.encoder_size]
                )
        else:
            feats = self.backbone.get_intermediate_layers(
                x, n=self.intermediate_layer_idx[self.encoder_size]
            )

        if self.layer_mapping is not None:
            feats = [feats[i] for i in self.layer_mapping]

        out = self.decoder(feats, patch_h, patch_w)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if return_feats:
            return out, feats[-1]
        return out
