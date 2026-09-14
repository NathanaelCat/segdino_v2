import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltp_cli import CrossLinearInteraction, LightweightTokenPyramid, fixed_2d_sincos


PATCH_DPA_VARIANTS = {"patch_dpa_shared", "patch_dpa_independent"}
PATCH_GUIDED_SAD_VARIANTS = {"patch_guided_sad"}
PATCH_PRIOR_VARIANTS = PATCH_DPA_VARIANTS | PATCH_GUIDED_SAD_VARIANTS
LTP_CLI_VARIANTS = {"ltp_cli"}
SPM_CCFM_VARIANTS = {"spm_ccfm_sad", "spm_ccfm_ms_mlp"}
SPSR_VARIANTS = {"l12_spm_ms_mlp", "l12_spsr_ms_mlp"}
SPM_VARIANTS = SPM_CCFM_VARIANTS | {"spm_sad"} | SPSR_VARIANTS
L12_MSEF_VARIANTS = {
    "l12_a_msef",
    "l12_a_cross_msef",
    "l12_a_cross_r",
    "l12_a_cross_dweca",
    "l12_a_cross_bottleneck64",
    "l12_a_cross_msef_cdr",
    "l12_a_cross_msef_ds",
    "l12_a_cross_ee",
    "l12_a_cross_see",
    "l12_a_cross_r_weighted",
    "l12_a_cross_r_dysample_splus",
}
CROSS_MSEF_VARIANTS = {
    "l12_a_cross_msef",
    "l12_a_cross_r",
    "l12_a_cross_dweca",
    "l12_a_cross_bottleneck64",
    "l12_a_cross_msef_cdr",
    "l12_a_cross_msef_ds",
    "l12_a_cross_ee",
    "l12_a_cross_see",
    "l12_a_cross_r_weighted",
    "l12_a_cross_r_dysample_splus",
}
DEEP_SUPERVISION_VARIANTS = {"l12_a_cross_msef_ds"}


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


class MSEFResidualBlock(nn.Module):
    """OSD adaptation of Multinex's MSEF block.

    The original MSEF core is ``DWConv(LN(x)) * SE(LN(x)) + x``.  For this
    experiment the core is wrapped by a zero-initialized outer residual
    coefficient.  The wrapper is intentional: it makes every MSEF location
    an exact identity at initialization, so replacing the existing zero-gamma
    SAD-R blocks is a one-variable, parent-equivalent comparison.

    This keeps the paper operation intact: channel-last LayerNorm, depthwise
    3x3 convolution, and squeeze-excitation with ReLU/tanh.  It does not add
    pointwise convolution, spatial attention, or a second projection.
    """

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        channels = int(channels)
        reduction_ratio = int(reduction_ratio)
        if channels <= 0 or reduction_ratio <= 0:
            raise ValueError("channels and reduction_ratio must be positive")
        hidden = max(1, channels // reduction_ratio)

        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.layer_norm = nn.LayerNorm(channels)
        # The public MSEF implementation uses the default bias=True here.
        self.depthwise_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )
        self.se_fc1 = nn.Linear(channels, hidden)
        self.se_fc2 = nn.Linear(hidden, channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _normalized(self, x):
        # LayerNorm is applied over channels at each spatial position, as in
        # the reference implementation, without changing the BCHW contract.
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        return x.permute(0, 3, 1, 2)

    def _msef_core(self, x):
        x_norm = self._normalized(x)
        depthwise = self.depthwise_conv(x_norm)
        pooled = F.adaptive_avg_pool2d(x_norm, output_size=1).flatten(1)
        excitation = F.relu(self.se_fc1(pooled))
        excitation = torch.tanh(self.se_fc2(excitation)).view(
            x.shape[0], self.channels, 1, 1
        )
        # SEBlock in the reference MSEF returns x_norm * excitation; MSEF
        # then multiplies that feature tensor with the depthwise branch.
        se_feature = x_norm * excitation
        return depthwise * se_feature

    def forward(self, x):
        return x + self.gamma * self._msef_core(x)


class CDRBlock(nn.Module):
    """Context-guided Detail Refinement block used at every SAD position.

    The block deliberately uses one identical definition at all four intra
    and four top-down SAD locations:

        Z = channel-wise LayerNorm(X)
        B = AvgPool3x3(Z)
        H = Z - B
        C = Group1x1(GELU(DWConv3x3(Z)))
        D = GELU(DWConv3x3(H))
        A = sigmoid(Group1x1(B))
        Y = X + C + beta * (A * D)

    All convolutions are bias-free to keep the block lightweight.  ``beta``
    is initialized to one as prescribed for the fresh CDR replacement
    experiment; there is no SE, feature multiplication, or outer zero-gamma.
    """

    def __init__(self, channels=256, groups=32):
        super().__init__()
        channels = int(channels)
        groups = int(groups)
        if channels <= 0 or groups <= 0 or channels % groups != 0:
            raise ValueError("CDR channels must be positive and divisible by groups")
        self.channels = channels
        self.groups = groups
        self.norm = nn.LayerNorm(channels)
        self.local_depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.local_projection = nn.Conv2d(
            channels, channels, kernel_size=1, groups=groups, bias=False
        )
        self.detail_depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.context_gate = nn.Conv2d(
            channels, channels, kernel_size=1, groups=groups, bias=False
        )
        self.act = nn.GELU()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def _normalize(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def _forward_impl(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"CDR expected [B,{self.channels},H,W], got {tuple(x.shape)}"
            )
        z = self._normalize(x)
        low = F.avg_pool2d(
            z, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        high = z - low

        local = self.local_depthwise(z)
        local = self.act(local)
        local = self.local_projection(local)

        detail = self.detail_depthwise(high)
        detail = self.act(detail)
        attention = torch.sigmoid(self.context_gate(low))
        selected_detail = attention * detail
        output = x + local + self.beta * selected_detail
        trace = {
            "X": x,
            "C": local,
            "D": detail,
            "A": attention,
            "A_times_D": selected_detail,
            "Y": output,
        }
        return output, trace

    def forward(self, x):
        return self._forward_impl(x)[0]

    def forward_with_trace(self, x):
        return self._forward_impl(x)


class EdgeEnhancerBlock(nn.Module):
    """The official EdgeEnhancer operator used as an OSD SAD replacement.

    The reference operator is a residual high-pass path::

        edge = x - AvgPool3x3(x)
        edge = Conv1x1(edge) -> Norm(edge) -> activation(edge)
        y = x + edge

    ``activation`` is the sole difference between the two experiments:
    ``sigmoid`` is the original EE and ``tanh`` is Signed-EE.  The OSD
    protocol uses GroupNorm when ``use_bn=false``; the same normalization is
    therefore used for both variants so their comparison changes only the
    response range.  ``count_include_pad=False`` is fixed by the OSD
    experiment specification to avoid a border-dependent average.
    """

    def __init__(
        self,
        channels: int,
        activation: str = "sigmoid",
        groups: int = 32,
        count_include_pad: bool = False,
    ):
        super().__init__()
        channels = int(channels)
        groups = int(groups)
        if channels <= 0 or groups <= 0 or channels % groups != 0:
            raise ValueError("EdgeEnhancer channels must be positive and divisible by groups")
        activation = str(activation).lower()
        if activation not in {"sigmoid", "tanh"}:
            raise ValueError(f"Unsupported EdgeEnhancer activation: {activation}")
        self.channels = channels
        self.activation = activation
        self.pool = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=bool(count_include_pad),
        )
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(groups, channels)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"EdgeEnhancer expected [B,{self.channels},H,W], got {tuple(x.shape)}"
            )
        edge = x - self.pool(x)
        edge = self.norm(self.out_conv(edge))
        if self.activation == "sigmoid":
            edge = torch.sigmoid(edge)
        else:
            edge = torch.tanh(edge)
        return x + edge


class OfficialMSEFBlock(nn.Module):
    """The unwrapped MultiNex MSEF operation used by the new A parent.

    This intentionally does not contain the zero-initialized outer residual
    coefficient used by the historical OSD MSEF experiment.  The operation is
    exactly::

        z = LayerNorm(x)
        l = DWConv3x3(z)
        c = z * tanh(FC2(ReLU(FC1(GAP(z)))))
        out = x + l * c

    The channel-last LayerNorm is only a layout change; the module's public
    contract remains BCHW.  ``c`` is the complete ``z * channel_weight``
    feature, not just the channel gate.
    """

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        channels = int(channels)
        reduction_ratio = int(reduction_ratio)
        if channels <= 0 or reduction_ratio <= 0:
            raise ValueError("channels and reduction_ratio must be positive")
        hidden = max(1, channels // reduction_ratio)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.layer_norm = nn.LayerNorm(channels)
        self.depthwise_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )
        self.se_fc1 = nn.Linear(channels, hidden)
        self.se_fc2 = nn.Linear(hidden, channels)

    def _normalized(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        return x.permute(0, 3, 1, 2)

    def forward(self, x):
        z = self._normalized(x)
        local = self.depthwise_conv(z)
        pooled = F.adaptive_avg_pool2d(z, output_size=1).flatten(1)
        channel_weight = torch.tanh(self.se_fc2(F.relu(self.se_fc1(pooled))))
        channel_weight = channel_weight.view(x.shape[0], self.channels, 1, 1)
        channel_feature = z * channel_weight
        return x + local * channel_feature


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


def _construct_with_fixed_seed(factory, seed):
    """Construct a CPU module with a seed independent of constructor order.

    The factorial runs share the initialization of their common parameters so
    that a single-seed comparison is not also a comparison of unrelated SAD
    or projection initializations.  The caller's RNG state is restored.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(seed))
        return factory()
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _safe_grad_norm(parameter):
    """Return a JSON-friendly gradient norm for optional diagnostics."""
    gradient = getattr(parameter, "grad", None)
    if gradient is None:
        return 0.0
    return float(gradient.detach().float().norm().item())


class TPAL12FactorialDecoder(nn.Module):
    """CTRL-002 TPA/SAD with two independently switchable TPA factors.

    This is a controlled decoder-side ablation for the already fixed
    ``layer_mapping=[3,3,3,3]`` protocol.  The four inputs are therefore all
    L12, while the decoder can independently vary:

    * projection sharing: one 1x1 projection reused four times vs. four
      independent 1x1 projections;
    * spatial reconstruction: bilinear-only scale expansion vs. the four
      independent 3x3 convolutions used by CTRL-002.

    SAD is copied exactly from ``TPASADDecoder`` in every cell.  The D cell is
    the existing CTRL-002 result and is not retrained.
    """

    INIT_SEEDS = {
        "shared_projection": 41001,
        "projection_0": 41001,
        "projection_1": 41002,
        "projection_2": 41003,
        "projection_3": 41004,
        "branch_1": 42001,
        "branch_2": 42002,
        "branch_3": 42003,
        "branch_4": 42004,
        "sad_intra_1": 43001,
        "sad_intra_2": 43002,
        "sad_intra_3": 43003,
        "sad_intra_4": 43004,
        "sad_inter_4": 43005,
        "sad_inter_3": 43006,
        "sad_inter_2": 43007,
        "sad_inter_1": 43008,
        "out_conv": 44001,
    }

    def __init__(
        self,
        in_dims,
        decoder_channels=256,
        num_classes=4,
        use_group_norm=True,
        projection_mode="shared",
        use_spatial_conv=False,
    ):
        super().__init__()
        if len(in_dims) != 4:
            raise ValueError(
                f"TPAL12FactorialDecoder requires four input dims, got {len(in_dims)}"
            )
        if projection_mode not in {"shared", "independent"}:
            raise ValueError(f"Unknown projection_mode={projection_mode!r}")

        self.projection_mode = projection_mode
        self.use_spatial_conv = bool(use_spatial_conv)
        self.decoder_channels = int(decoder_channels)

        if projection_mode == "shared":
            self.shared_token_projection = _construct_with_fixed_seed(
                lambda: nn.Conv2d(in_dims[0], decoder_channels, 1, bias=False),
                self.INIT_SEEDS["shared_projection"],
            )
            self.token_projections = None
        else:
            self.shared_token_projection = None
            self.token_projections = nn.ModuleList(
                [
                    _construct_with_fixed_seed(
                        lambda channels=channels, index=index: nn.Conv2d(
                            channels, decoder_channels, 1, bias=False
                        ),
                        self.INIT_SEEDS[f"projection_{index}"],
                    )
                    for index, channels in enumerate(in_dims)
                ]
            )

        if self.use_spatial_conv:
            self.tpa_branches = nn.ModuleList(
                [
                    _construct_with_fixed_seed(
                        lambda scale=scale: TPAResampleProject(
                            decoder_channels, scale_factor=scale
                        ),
                        self.INIT_SEEDS[f"branch_{index}"],
                    )
                    for index, scale in enumerate((8, 4, 2, 1), start=1)
                ]
            )
        else:
            self.tpa_branches = None

        # Keep the SAD consumer identical in all four cells.
        for name, seed in (
            ("sad_intra_1", self.INIT_SEEDS["sad_intra_1"]),
            ("sad_intra_2", self.INIT_SEEDS["sad_intra_2"]),
            ("sad_intra_3", self.INIT_SEEDS["sad_intra_3"]),
            ("sad_intra_4", self.INIT_SEEDS["sad_intra_4"]),
            ("sad_inter_4", self.INIT_SEEDS["sad_inter_4"]),
            ("sad_inter_3", self.INIT_SEEDS["sad_inter_3"]),
            ("sad_inter_2", self.INIT_SEEDS["sad_inter_2"]),
            ("sad_inter_1", self.INIT_SEEDS["sad_inter_1"]),
        ):
            setattr(
                self,
                name,
                _construct_with_fixed_seed(
                    lambda: ResidualDepthwiseBlock(
                        decoder_channels, use_group_norm=use_group_norm
                    ),
                    seed,
                ),
            )
        self.out_conv = _construct_with_fixed_seed(
            lambda: nn.Conv2d(decoder_channels, num_classes, 1),
            self.INIT_SEEDS["out_conv"],
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

    def _project_tokens(self, features, patch_h, patch_w):
        if len(features) != 4:
            raise ValueError(f"Factorial decoder requires four features, got {len(features)}")
        maps = [self._tokens_to_feature_map(tokens, patch_h, patch_w) for tokens in features]
        if self.projection_mode == "shared":
            return [self.shared_token_projection(feature) for feature in maps]
        return [projection(feature) for projection, feature in zip(self.token_projections, maps)]

    def _build_pyramid(self, projected):
        if len(projected) != 4:
            raise ValueError(f"Factorial pyramid requires four projected maps, got {len(projected)}")
        if self.tpa_branches is not None:
            return tuple(branch(feature) for branch, feature in zip(self.tpa_branches, projected))
        return (
            F.interpolate(projected[0], scale_factor=8, mode="bilinear", align_corners=False),
            F.interpolate(projected[1], scale_factor=4, mode="bilinear", align_corners=False),
            F.interpolate(projected[2], scale_factor=2, mode="bilinear", align_corners=False),
            projected[3],
        )

    def forward(self, features, patch_h, patch_w):
        projected = self._project_tokens(features, patch_h, patch_w)
        branch_1, branch_2, branch_3, branch_4 = self._build_pyramid(projected)

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


class L12AMSEFDecoder(nn.Module):
    """Factorial-A's light L12 pyramid with the exact official MSEF consumer.

    The semantic path is the A cell from the L12 factorial: one shared 1x1
    projection followed by bilinear-only expansion to P2/P4/P8/P16.  The
    eight ordinary SAD-R blocks are replaced by the unwrapped official MSEF
    operation.  Construction uses the same fixed per-module seeds as the A
    cell for the shared projection and output path, but no checkpoint is read.
    """

    INIT_SEEDS = TPAL12FactorialDecoder.INIT_SEEDS

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__()
        if len(in_dims) != 4:
            raise ValueError(f"L12AMSEFDecoder requires four input dims, got {len(in_dims)}")
        self.decoder_channels = int(decoder_channels)
        self.projection_mode = "shared"
        self.use_spatial_conv = False
        self.shared_token_projection = _construct_with_fixed_seed(
            lambda: nn.Conv2d(in_dims[0], decoder_channels, 1, bias=False),
            self.INIT_SEEDS["shared_projection"],
        )
        self.token_projections = None

        for name, seed in (
            ("sad_intra_1", self.INIT_SEEDS["sad_intra_1"]),
            ("sad_intra_2", self.INIT_SEEDS["sad_intra_2"]),
            ("sad_intra_3", self.INIT_SEEDS["sad_intra_3"]),
            ("sad_intra_4", self.INIT_SEEDS["sad_intra_4"]),
            ("sad_inter_4", self.INIT_SEEDS["sad_inter_4"]),
            ("sad_inter_3", self.INIT_SEEDS["sad_inter_3"]),
            ("sad_inter_2", self.INIT_SEEDS["sad_inter_2"]),
            ("sad_inter_1", self.INIT_SEEDS["sad_inter_1"]),
        ):
            setattr(
                self,
                name,
                _construct_with_fixed_seed(
                    lambda: OfficialMSEFBlock(decoder_channels, reduction_ratio=16),
                    seed,
                ),
            )
        self.out_conv = _construct_with_fixed_seed(
            lambda: nn.Conv2d(decoder_channels, num_classes, 1),
            self.INIT_SEEDS["out_conv"],
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

    def _project_l12(self, semantic_tokens, patch_h, patch_w):
        semantic_map = self._tokens_to_feature_map(semantic_tokens, patch_h, patch_w)
        return self.shared_token_projection(semantic_map)

    @staticmethod
    def _build_bilinear_pyramid(projected):
        return (
            F.interpolate(projected, scale_factor=8, mode="bilinear", align_corners=False),
            F.interpolate(projected, scale_factor=4, mode="bilinear", align_corners=False),
            F.interpolate(projected, scale_factor=2, mode="bilinear", align_corners=False),
            projected,
        )

    def _forward_msef_sad(self, pyramid):
        p2, p4, p8, p16 = pyramid
        level_1 = self.sad_intra_1(p2)
        level_2 = self.sad_intra_2(p4)
        level_3 = self.sad_intra_3(p8)
        level_4 = self.sad_intra_4(p16)

        x4 = self.sad_inter_4(level_4)
        x3 = self.sad_inter_3(
            F.interpolate(x4, size=level_3.shape[-2:], mode="bilinear", align_corners=False)
            + level_3
        )
        x2 = self.sad_inter_2(
            F.interpolate(x3, size=level_2.shape[-2:], mode="bilinear", align_corners=False)
            + level_2
        )
        x1 = self.sad_inter_1(
            F.interpolate(x2, size=level_1.shape[-2:], mode="bilinear", align_corners=False)
            + level_1
        )
        return self.out_conv(x1)

    def forward(self, features, patch_h, patch_w):
        if len(features) != 4:
            raise ValueError(f"L12AMSEFDecoder requires four features, got {len(features)}")
        projected = self._project_l12(features[-1], patch_h, patch_w)
        return self._forward_msef_sad(self._build_bilinear_pyramid(projected))


class CrossMSEF(nn.Module):
    """Patch-prescribed Cross-MSEF interaction at one decoder scale.

    It keeps the full spatial feature after the EAG-style enhancement, applies
    the official MSEF local/channel branches in the 64-dimensional latent
    space, and injects the resulting 256-channel residual with a zero scalar
    gamma.  There is no attention matrix, extra normalization, or post-EAG
    ECA/SA path.
    """

    def __init__(self, semantic_channels=256, spatial_channels=32, interaction_dim=64):
        super().__init__()
        semantic_channels = int(semantic_channels)
        spatial_channels = int(spatial_channels)
        interaction_dim = int(interaction_dim)
        if interaction_dim % 32 != 0:
            raise ValueError("Cross-MSEF interaction_dim must be divisible by groups=32")
        self.semantic_channels = semantic_channels
        self.spatial_channels = spatial_channels
        self.interaction_dim = interaction_dim

        self.semantic_projection = nn.Conv2d(
            semantic_channels, interaction_dim, kernel_size=1, bias=False
        )
        self.spatial_projection = nn.Conv2d(
            spatial_channels, interaction_dim, kernel_size=1, bias=False
        )

        self.semantic_guide = nn.Sequential(
            nn.Conv2d(
                interaction_dim,
                interaction_dim,
                kernel_size=1,
                groups=32,
                bias=False,
            ),
            nn.BatchNorm2d(interaction_dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_guide = nn.Sequential(
            nn.Conv2d(
                interaction_dim,
                interaction_dim,
                kernel_size=1,
                groups=32,
                bias=False,
            ),
            nn.BatchNorm2d(interaction_dim),
            nn.ReLU(inplace=True),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(interaction_dim, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.spatial_norm = nn.LayerNorm(interaction_dim)
        self.spatial_depthwise = nn.Conv2d(
            interaction_dim,
            interaction_dim,
            kernel_size=3,
            padding=1,
            groups=interaction_dim,
            bias=True,
        )
        self.semantic_norm = nn.LayerNorm(interaction_dim)
        hidden = max(1, interaction_dim // 16)
        self.se_fc1 = nn.Linear(interaction_dim, hidden)
        self.se_fc2 = nn.Linear(hidden, interaction_dim)
        self.delta_projection = nn.Conv2d(
            interaction_dim, semantic_channels, kernel_size=1, bias=False
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _channel_layer_norm(x, norm):
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, semantic_feature, spatial_feature):
        if semantic_feature.ndim != 4 or spatial_feature.ndim != 4:
            raise ValueError("Cross-MSEF expects BCHW semantic and spatial features")
        if semantic_feature.shape[0] != spatial_feature.shape[0]:
            raise ValueError("Cross-MSEF batch sizes do not match")
        if semantic_feature.shape[-2:] != spatial_feature.shape[-2:]:
            raise ValueError(
                "Cross-MSEF semantic/spatial resolutions do not match: "
                f"{tuple(semantic_feature.shape[-2:])} vs {tuple(spatial_feature.shape[-2:])}"
            )

        p = self.semantic_projection(semantic_feature)
        s = self.spatial_projection(spatial_feature)

        pg = self.semantic_guide(p)
        sg = self.spatial_guide(s)
        a = F.relu(pg + sg, inplace=False)
        psi = self.psi(a)
        s_hat = s * psi + s

        zs = self._channel_layer_norm(s_hat, self.spatial_norm)
        local = self.spatial_depthwise(zs)

        zp = self._channel_layer_norm(p, self.semantic_norm)
        pooled = F.adaptive_avg_pool2d(zp, output_size=1).flatten(1)
        channel_weight = torch.tanh(self.se_fc2(F.relu(self.se_fc1(pooled))))
        channel_weight = channel_weight.view(p.shape[0], self.interaction_dim, 1, 1)
        channel_feature = zp * channel_weight

        delta64 = local * channel_feature
        delta256 = self.delta_projection(delta64)
        return semantic_feature + self.gamma * delta256


class L12ACrossMSEFDecoder(L12AMSEFDecoder):
    """EXP-2: A-MSEF plus RGB Lite-SPM Cross-MSEF at P2/P4/P8."""

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__(in_dims, decoder_channels=decoder_channels, num_classes=num_classes)
        if int(decoder_channels) != 256:
            raise ValueError("The prescribed Cross-MSEF experiment requires decoder_channels=256")
        self.cross_msef_2 = _construct_with_fixed_seed(
            lambda: CrossMSEF(256, 32, 64), 45002
        )
        self.cross_msef_4 = _construct_with_fixed_seed(
            lambda: CrossMSEF(256, 64, 64), 45004
        )
        self.cross_msef_8 = _construct_with_fixed_seed(
            lambda: CrossMSEF(256, 128, 64), 45008
        )

    def forward(self, semantic_tokens, spatial_features, patch_h, patch_w):
        if len(spatial_features) != 3:
            raise ValueError(f"Cross-MSEF requires D2/D4/D8, got {len(spatial_features)} maps")
        projected = self._project_l12(semantic_tokens, patch_h, patch_w)
        p2, p4, p8, p16 = self._build_bilinear_pyramid(projected)
        p2 = self.cross_msef_2(p2, spatial_features[0])
        p4 = self.cross_msef_4(p4, spatial_features[1])
        p8 = self.cross_msef_8(p8, spatial_features[2])
        return self._forward_msef_sad((p2, p4, p8, p16))


class L12ACrossEEBaseDecoder(L12ACrossMSEFDecoder):
    """Cross-MSEF front-end with a uniform eight-position EE-family SAD.

    The inherited L12 semantic pyramid, RGB Lite-SPM, and three Cross-MSEF
    modules are unchanged.  Only the eight actual SAD consumer blocks are
    replaced.  Both variants use the same fixed constructor seeds, so the
    EE-versus-Signed-EE comparison changes only the pointwise activation.
    """

    EE_SEEDS = {
        "sad_intra_1": 49001,
        "sad_intra_2": 49002,
        "sad_intra_3": 49003,
        "sad_intra_4": 49004,
        "sad_inter_4": 49005,
        "sad_inter_3": 49006,
        "sad_inter_2": 49007,
        "sad_inter_1": 49008,
    }

    def __init__(
        self,
        in_dims,
        decoder_channels=256,
        num_classes=4,
        edge_activation="sigmoid",
    ):
        super().__init__(
            in_dims,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )
        if int(decoder_channels) != 256:
            raise ValueError("The prescribed EE experiment requires decoder_channels=256")
        edge_activation = str(edge_activation).lower()
        if edge_activation not in {"sigmoid", "tanh"}:
            raise ValueError(f"Unsupported EE activation: {edge_activation}")
        self.edge_activation = edge_activation
        for name, seed in self.EE_SEEDS.items():
            setattr(
                self,
                name,
                _construct_with_fixed_seed(
                    lambda: EdgeEnhancerBlock(
                        decoder_channels,
                        activation=edge_activation,
                        groups=32,
                        count_include_pad=False,
                    ),
                    seed,
                ),
            )


class L12ACrossRDecoder(L12ACrossMSEFDecoder):
    """A + RGB Lite-SPM Cross-MSEF with the original SAD-R blocks.

    This is the missing causal control for the Cross-MSEF experiment. The
    L12 shared projection, bilinear semantic pyramid, RGB Lite-SPM, and the
    three Cross-MSEF modules are inherited unchanged. Only the eight SAD
    consumer blocks are replaced with the original ResidualDepthwiseBlock.
    """

    R_SEEDS = {
        "sad_intra_1": 43001,
        "sad_intra_2": 43002,
        "sad_intra_3": 43003,
        "sad_intra_4": 43004,
        "sad_inter_4": 43005,
        "sad_inter_3": 43006,
        "sad_inter_2": 43007,
        "sad_inter_1": 43008,
    }

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__(in_dims, decoder_channels=decoder_channels, num_classes=num_classes)
        if int(decoder_channels) != 256:
            raise ValueError("The prescribed Cross-R control requires decoder_channels=256")
        for name, seed in self.R_SEEDS.items():
            setattr(
                self,
                name,
                _construct_with_fixed_seed(
                    lambda: ResidualDepthwiseBlock(
                        decoder_channels, use_group_norm=True
                    ),
                    seed,
                ),
            )


class DWECABlock(nn.Module):
    """Lightweight depthwise spatial refinement with ECA recalibration.

    This is the SAD-DW-ECA candidate.  The normalization follows the original
    SAD-R block's GroupNorm convention, while the dense 1x1 channel mixing and
    GELU are removed.  A standard ECA 1-D channel interaction is applied to
    the globally pooled depthwise response.  The outer scalar gamma is
    zero-initialized so each block is an exact identity at initialization.
    """

    def __init__(self, channels=256, groups=32, eca_kernel_size=5):
        super().__init__()
        channels = int(channels)
        groups = int(groups)
        eca_kernel_size = int(eca_kernel_size)
        if channels <= 0 or groups <= 0 or channels % groups != 0:
            raise ValueError("channels must be positive and divisible by groups")
        if eca_kernel_size <= 0 or eca_kernel_size % 2 == 0:
            raise ValueError("ECA kernel size must be a positive odd integer")
        self.channels = channels
        self.eca_kernel_size = eca_kernel_size
        self.norm = nn.GroupNorm(groups, channels)
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        self.eca = nn.Conv1d(
            1, 1, kernel_size=eca_kernel_size,
            padding=eca_kernel_size // 2, bias=False
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"DWECABlock expected [B,{self.channels},H,W], got {tuple(x.shape)}"
            )
        z = self.depthwise(self.norm(x))
        pooled = F.adaptive_avg_pool2d(z, output_size=1).flatten(1).unsqueeze(1)
        attention = torch.sigmoid(self.eca(pooled)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return x + self.gamma * (attention * z)


class L12ACrossDWECADecoder(L12ACrossMSEFDecoder):
    """Cross-MSEF front-end with uniform DW-ECA blocks in SAD."""

    DWECA_SEEDS = {
        "sad_intra_1": 51001,
        "sad_intra_2": 51002,
        "sad_intra_3": 51003,
        "sad_intra_4": 51004,
        "sad_inter_4": 51005,
        "sad_inter_3": 51006,
        "sad_inter_2": 51007,
        "sad_inter_1": 51008,
    }

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__(in_dims, decoder_channels=decoder_channels, num_classes=num_classes)
        if int(decoder_channels) != 256:
            raise ValueError("The prescribed DW-ECA experiment requires decoder_channels=256")
        for name, seed in self.DWECA_SEEDS.items():
            setattr(
                self,
                name,
                _construct_with_fixed_seed(
                    lambda: DWECABlock(decoder_channels, groups=32, eca_kernel_size=5),
                    seed,
                ),
            )


class BottleneckResidualBlock(nn.Module):
    """SAD refinement with a nonlinear low-width channel bottleneck.

    The block keeps the R block's local depthwise operation, post-mixer
    GroupNorm/GELU, and zero-initialized residual scale, while replacing the
    dense 256->256 pointwise mixer with 256->rank->256 and an intermediate
    GELU.  This is intentionally a candidate SAD* block rather than a strict
    linear low-rank factorization.
    """

    def __init__(self, channels=256, bottleneck=64, groups=32):
        super().__init__()
        channels = int(channels)
        bottleneck = int(bottleneck)
        groups = int(groups)
        if channels <= 0 or bottleneck <= 0 or groups <= 0:
            raise ValueError("channels, bottleneck, and groups must be positive")
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        self.channels = channels
        self.bottleneck = bottleneck
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        self.down = nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False)
        self.inner_act = nn.GELU()
        self.up = nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(groups, channels)
        self.out_act = nn.GELU()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"BottleneckResidualBlock expected [B,{self.channels},H,W], "
                f"got {tuple(x.shape)}"
            )
        residual = self.depthwise(x)
        residual = self.inner_act(self.down(residual))
        residual = self.up(residual)
        residual = self.out_act(self.norm(residual))
        return x + self.gamma * residual


class L12ACrossBottleneck64Decoder(L12ACrossMSEFDecoder):
    """Cross-MSEF front-end with uniform rank-64 bottleneck SAD blocks."""

    BOTTLENECK_SEEDS = {
        "sad_intra_1": 52001,
        "sad_intra_2": 52002,
        "sad_intra_3": 52003,
        "sad_intra_4": 52004,
        "sad_inter_4": 52005,
        "sad_inter_3": 52006,
        "sad_inter_2": 52007,
        "sad_inter_1": 52008,
    }

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__(in_dims, decoder_channels=decoder_channels, num_classes=num_classes)
        if int(decoder_channels) != 256:
            raise ValueError("The prescribed bottleneck experiment requires decoder_channels=256")
        for name, seed in self.BOTTLENECK_SEEDS.items():
            setattr(
                self,
                name,
                _construct_with_fixed_seed(
                    lambda: BottleneckResidualBlock(
                        decoder_channels, bottleneck=64, groups=32
                    ),
                    seed,
                ),
            )


class OfficialDySample(nn.Module):
    """Official DySample operator kept local to the OSD runner.

    This follows the authors' released implementation: feature-conditioned
    offsets are converted to sampling grids and applied with ``grid_sample``.
    The OSD diagnostic uses the released DySample-S+ setting
    (``style='pl'``, ``groups=8``, ``dyscope=True``, ``scale=2``).
    """

    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        self.scale = int(scale)
        self.style = str(style)
        self.groups = int(groups)
        if self.style not in {"lp", "pl"}:
            raise ValueError(f"Unsupported DySample style={self.style!r}")
        if self.style == "pl":
            if in_channels < self.scale ** 2 or in_channels % (self.scale ** 2) != 0:
                raise ValueError("DySample PL requires channels divisible by scale^2")
        if in_channels < self.groups or in_channels % self.groups != 0:
            raise ValueError("DySample requires channels divisible by groups")

        offset_in_channels = int(in_channels)
        if self.style == "pl":
            offset_in_channels //= self.scale ** 2
            out_channels = 2 * self.groups
        else:
            out_channels = 2 * self.groups * self.scale ** 2

        self.offset = nn.Conv2d(offset_in_channels, out_channels, 1)
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.offset.bias, 0.0)
        if dyscope:
            self.scope = nn.Conv2d(offset_in_channels, out_channels, 1, bias=False)
            nn.init.constant_(self.scope.weight, 0.0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange(
            (-self.scale + 1) / 2,
            (self.scale - 1) / 2 + 1,
        ) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h]))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def sample(self, x, offset):
        batch, _, height, width = offset.shape
        offset = offset.view(batch, 2, -1, height, width)
        coords_h = torch.arange(height, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(width, device=x.device, dtype=x.dtype) + 0.5
        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h]))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
        )
        normalizer = torch.tensor(
            [width, height], dtype=x.dtype, device=x.device
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(batch, -1, height, width), self.scale)
            .view(batch, 2, -1, self.scale * height, self.scale * width)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(batch * self.groups, -1, height, width),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(batch, -1, self.scale * height, self.scale * width)

    def forward_lp(self, x):
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = (
                F.pixel_unshuffle(
                    self.offset(x_) * self.scope(x_).sigmoid(), self.scale
                )
                * 0.5
                + self.init_pos
            )
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == "pl":
            return self.forward_pl(x)
        return self.forward_lp(x)


class L12ACrossWeightedRDecoder(L12ACrossRDecoder):
    """Cross-R with zero-initialized scalar weights for P+Up fusion."""

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__(in_dims, decoder_channels=decoder_channels, num_classes=num_classes)
        if int(decoder_channels) != 256:
            raise ValueError("Weighted Cross-R requires decoder_channels=256")
        # 2*softmax(theta) gives [1, 1] at theta=[0, 0], exactly preserving
        # the historical unweighted P+Up operation at initialization.
        self.fusion_logits = nn.Parameter(torch.zeros(3, 2))

    def _fusion_weights(self, index):
        return 2.0 * F.softmax(self.fusion_logits[index], dim=0)

    def forward(self, semantic_tokens, spatial_features, patch_h, patch_w):
        if len(spatial_features) != 3:
            raise ValueError(
                f"Weighted Cross-R requires D2/D4/D8, got {len(spatial_features)} maps"
            )
        projected = self._project_l12(semantic_tokens, patch_h, patch_w)
        p2, p4, p8, p16 = self._build_bilinear_pyramid(projected)
        p2 = self.cross_msef_2(p2, spatial_features[0])
        p4 = self.cross_msef_4(p4, spatial_features[1])
        p8 = self.cross_msef_8(p8, spatial_features[2])

        level_1 = self.sad_intra_1(p2)
        level_2 = self.sad_intra_2(p4)
        level_3 = self.sad_intra_3(p8)
        level_4 = self.sad_intra_4(p16)
        x4 = self.sad_inter_4(level_4)

        w8 = self._fusion_weights(0)
        x3_up = F.interpolate(x4, size=level_3.shape[-2:], mode="bilinear", align_corners=False)
        x3 = self.sad_inter_3(w8[0] * level_3 + w8[1] * x3_up)

        w4 = self._fusion_weights(1)
        x2_up = F.interpolate(x3, size=level_2.shape[-2:], mode="bilinear", align_corners=False)
        x2 = self.sad_inter_2(w4[0] * level_2 + w4[1] * x2_up)

        w2 = self._fusion_weights(2)
        x1_up = F.interpolate(x2, size=level_1.shape[-2:], mode="bilinear", align_corners=False)
        x1 = self.sad_inter_1(w2[0] * level_1 + w2[1] * x1_up)
        return self.out_conv(x1)


class L12ACrossDySampleRDecoder(L12ACrossRDecoder):
    """Cross-R with official DySample-S+ at the three inter upsampling sites."""

    DYSAMPLE_SEEDS = {
        "dysample_p8": 52008,
        "dysample_p4": 52004,
        "dysample_p2": 52002,
    }

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__(in_dims, decoder_channels=decoder_channels, num_classes=num_classes)
        if int(decoder_channels) != 256:
            raise ValueError("DySample Cross-R requires decoder_channels=256")
        for name, seed in self.DYSAMPLE_SEEDS.items():
            setattr(
                self,
                name,
                _construct_with_fixed_seed(
                    lambda: OfficialDySample(
                        decoder_channels,
                        scale=2,
                        style="pl",
                        groups=8,
                        dyscope=True,
                    ),
                    seed,
                ),
            )

    def forward(self, semantic_tokens, spatial_features, patch_h, patch_w):
        if len(spatial_features) != 3:
            raise ValueError(
                f"DySample Cross-R requires D2/D4/D8, got {len(spatial_features)} maps"
            )
        projected = self._project_l12(semantic_tokens, patch_h, patch_w)
        p2, p4, p8, p16 = self._build_bilinear_pyramid(projected)
        p2 = self.cross_msef_2(p2, spatial_features[0])
        p4 = self.cross_msef_4(p4, spatial_features[1])
        p8 = self.cross_msef_8(p8, spatial_features[2])

        level_1 = self.sad_intra_1(p2)
        level_2 = self.sad_intra_2(p4)
        level_3 = self.sad_intra_3(p8)
        level_4 = self.sad_intra_4(p16)
        x4 = self.sad_inter_4(level_4)
        x3 = self.sad_inter_3(self.dysample_p8(x4) + level_3)
        x2 = self.sad_inter_2(self.dysample_p4(x3) + level_2)
        x1 = self.sad_inter_1(self.dysample_p2(x2) + level_1)
        return self.out_conv(x1)


class L12ACDRDecoder(L12ACrossMSEFDecoder):
    """Cross-MSEF front-end with one uniform CDR block at all SAD positions."""

    CDR_SEEDS = {
        "sad_intra_1": 47001,
        "sad_intra_2": 47002,
        "sad_intra_3": 47003,
        "sad_intra_4": 47004,
        "sad_inter_4": 47005,
        "sad_inter_3": 47006,
        "sad_inter_2": 47007,
        "sad_inter_1": 47008,
    }

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__(in_dims, decoder_channels=decoder_channels, num_classes=num_classes)
        if int(decoder_channels) != 256:
            raise ValueError("The prescribed CDR experiment requires decoder_channels=256")
        for name, seed in self.CDR_SEEDS.items():
            setattr(
                self,
                name,
                _construct_with_fixed_seed(
                    lambda: CDRBlock(decoder_channels, groups=32), seed
                ),
            )

    def _cross_pyramid(self, semantic_tokens, spatial_features, patch_h, patch_w):
        if len(spatial_features) != 3:
            raise ValueError(f"CDR requires D2/D4/D8, got {len(spatial_features)} maps")
        projected = self._project_l12(semantic_tokens, patch_h, patch_w)
        p2, p4, p8, p16 = self._build_bilinear_pyramid(projected)
        p2 = self.cross_msef_2(p2, spatial_features[0])
        p4 = self.cross_msef_4(p4, spatial_features[1])
        p8 = self.cross_msef_8(p8, spatial_features[2])
        return p2, p4, p8, p16

    def _forward_cdr_sad(self, pyramid, return_trace=False):
        p2, p4, p8, p16 = pyramid
        if not return_trace:
            # The diagnostic trace contains several full-resolution tensors at
            # P2/P4.  Do not materialize or retain it during ordinary training
            # and inference; otherwise all eight block traces stay alive until
            # this whole SAD forward returns.  The numerical path below is
            # identical to the traced path.
            level_1 = self.sad_intra_1(p2)
            level_2 = self.sad_intra_2(p4)
            level_3 = self.sad_intra_3(p8)
            level_4 = self.sad_intra_4(p16)
            x4 = self.sad_inter_4(level_4)
            x3 = self.sad_inter_3(
                F.interpolate(x4, size=level_3.shape[-2:], mode="bilinear", align_corners=False)
                + level_3
            )
            x2 = self.sad_inter_2(
                F.interpolate(x3, size=level_2.shape[-2:], mode="bilinear", align_corners=False)
                + level_2
            )
            x1 = self.sad_inter_1(
                F.interpolate(x2, size=level_1.shape[-2:], mode="bilinear", align_corners=False)
                + level_1
            )
            return self.out_conv(x1)

        level_1, trace_1 = self.sad_intra_1.forward_with_trace(p2)
        level_2, trace_2 = self.sad_intra_2.forward_with_trace(p4)
        level_3, trace_3 = self.sad_intra_3.forward_with_trace(p8)
        level_4, trace_4 = self.sad_intra_4.forward_with_trace(p16)

        x4, trace_5 = self.sad_inter_4.forward_with_trace(level_4)
        x3_input = F.interpolate(
            x4, size=level_3.shape[-2:], mode="bilinear", align_corners=False
        ) + level_3
        x3, trace_6 = self.sad_inter_3.forward_with_trace(x3_input)
        x2_input = F.interpolate(
            x3, size=level_2.shape[-2:], mode="bilinear", align_corners=False
        ) + level_2
        x2, trace_7 = self.sad_inter_2.forward_with_trace(x2_input)
        x1_input = F.interpolate(
            x2, size=level_1.shape[-2:], mode="bilinear", align_corners=False
        ) + level_1
        x1, trace_8 = self.sad_inter_1.forward_with_trace(x1_input)
        logits = self.out_conv(x1)
        if not return_trace:
            return logits
        traces = {
            "sad_intra_1": trace_1,
            "sad_intra_2": trace_2,
            "sad_intra_3": trace_3,
            "sad_intra_4": trace_4,
            "sad_inter_4": trace_5,
            "sad_inter_3": trace_6,
            "sad_inter_2": trace_7,
            "sad_inter_1": trace_8,
        }
        return logits, traces

    def forward(self, semantic_tokens, spatial_features, patch_h, patch_w):
        pyramid = self._cross_pyramid(semantic_tokens, spatial_features, patch_h, patch_w)
        return self._forward_cdr_sad(pyramid)

    def forward_with_diagnostics(self, semantic_tokens, spatial_features, patch_h, patch_w):
        pyramid = self._cross_pyramid(semantic_tokens, spatial_features, patch_h, patch_w)
        return self._forward_cdr_sad(pyramid, return_trace=True)


class L12ACrossMSEFDeepSupervisionDecoder(L12ACrossMSEFDecoder):
    """Cross-MSEF baseline with auxiliary CE heads on inter3/inter2/inter1."""

    AUX_HEAD_SEEDS = {
        "aux_inter3": 48001,
        "aux_inter2": 48002,
        "aux_inter1": 48003,
    }

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__(in_dims, decoder_channels=decoder_channels, num_classes=num_classes)
        if int(decoder_channels) != 256:
            raise ValueError("The prescribed deep-supervision experiment requires decoder_channels=256")
        for name, seed in self.AUX_HEAD_SEEDS.items():
            setattr(
                self,
                name,
                _construct_with_fixed_seed(
                    lambda: nn.Conv2d(decoder_channels, num_classes, 1), seed
                ),
            )

    def _forward_msef_sad_with_aux(self, pyramid):
        p2, p4, p8, p16 = pyramid
        level_1 = self.sad_intra_1(p2)
        level_2 = self.sad_intra_2(p4)
        level_3 = self.sad_intra_3(p8)
        level_4 = self.sad_intra_4(p16)

        x4 = self.sad_inter_4(level_4)
        x3 = self.sad_inter_3(
            F.interpolate(x4, size=level_3.shape[-2:], mode="bilinear", align_corners=False)
            + level_3
        )
        x2 = self.sad_inter_2(
            F.interpolate(x3, size=level_2.shape[-2:], mode="bilinear", align_corners=False)
            + level_2
        )
        x1 = self.sad_inter_1(
            F.interpolate(x2, size=level_1.shape[-2:], mode="bilinear", align_corners=False)
            + level_1
        )
        return self.out_conv(x1), (
            self.aux_inter3(x3),
            self.aux_inter2(x2),
            self.aux_inter1(x1),
        )

    def forward_with_aux(self, semantic_tokens, spatial_features, patch_h, patch_w):
        if len(spatial_features) != 3:
            raise ValueError(f"Deep supervision requires D2/D4/D8, got {len(spatial_features)} maps")
        projected = self._project_l12(semantic_tokens, patch_h, patch_w)
        p2, p4, p8, p16 = self._build_bilinear_pyramid(projected)
        p2 = self.cross_msef_2(p2, spatial_features[0])
        p4 = self.cross_msef_4(p4, spatial_features[1])
        p8 = self.cross_msef_8(p8, spatial_features[2])
        return self._forward_msef_sad_with_aux((p2, p4, p8, p16))


class TPASADMSEFDecoder(TPASADDecoder):
    """TPA + SAD with all eight SAD-R locations replaced by MSEF.

    TPA branches, layer routing, feature resolutions, top-down order, and the
    classifier are inherited unchanged from ``TPASADDecoder``.  Only the four
    intra-scale and four inter-scale refinement blocks are replaced.
    """

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
        msef_reduction=16,
    ):
        # ``use_group_norm`` is accepted to keep the parent constructor
        # interface/configuration identical; MSEF uses LayerNorm by design.
        del use_group_norm
        super().__init__(
            in_dims,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            use_group_norm=True,
            adaptive_readout=adaptive_readout,
            readout_mode=readout_mode,
            readout_init=readout_init,
            readout_temperature=readout_temperature,
        )

        msef_blocks = {
            "sad_intra_1": MSEFResidualBlock(decoder_channels, msef_reduction),
            "sad_intra_2": MSEFResidualBlock(decoder_channels, msef_reduction),
            "sad_intra_3": MSEFResidualBlock(decoder_channels, msef_reduction),
            "sad_intra_4": MSEFResidualBlock(decoder_channels, msef_reduction),
            "sad_inter_4": MSEFResidualBlock(decoder_channels, msef_reduction),
            "sad_inter_3": MSEFResidualBlock(decoder_channels, msef_reduction),
            "sad_inter_2": MSEFResidualBlock(decoder_channels, msef_reduction),
            "sad_inter_1": MSEFResidualBlock(decoder_channels, msef_reduction),
        }
        for name, block in msef_blocks.items():
            setattr(self, name, block)


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


class LiteSpatialPriorStem(nn.Module):
    """Trainable RGB-only spatial-prior stem for the SPM/CCFM experiments.

    The stem is deliberately separate from the frozen DINOv3 path.  It uses
    three successive 3x3, stride-2 convolutions, yielding genuine RGB maps at
    1/2, 1/4, and 1/8 of the input resolution.  The fourth 1/16 input to CCFM
    is supplied by the DINOv3 L12 feature map.
    """

    def __init__(self, in_channels=3, channels=(32, 64, 128)):
        super().__init__()
        layers = []
        current = int(in_channels)
        for output_channels in channels:
            output_channels = int(output_channels)
            layers.append(
                nn.Sequential(
                    nn.Conv2d(current, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                )
            )
            current = output_channels
        self.layers = nn.ModuleList(layers)
        self.out_channels = tuple(int(value) for value in channels)

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return tuple(outputs)


class SpatialGuidedSemanticResampler(nn.Module):
    """Shared sparse resampler for the L12 spatial-prior prototype.

    The source is the projected frozen-DINO L12 map at 1/16 resolution.  A
    high-resolution RGB spatial-prior map supplies per-target offsets, while
    the sampled semantic values are ranked by an explicit spatial-query /
    semantic-value compatibility score.  The module deliberately has no
    dense attention matrix: it samples ``num_samples`` source points per
    target location and aggregates only along that small sample dimension.

    Coordinates are expressed in source-token units.  ``grid_sample`` uses
    the equivalent ``align_corners=False`` normalized coordinates, so the
    same source coordinate convention is used for P8, P4, and P2.
    """

    def __init__(
        self,
        semantic_channels=256,
        spatial_channels=64,
        interaction_dim=64,
        num_samples=4,
        offset_limit=1.0,
    ):
        super().__init__()
        semantic_channels = int(semantic_channels)
        spatial_channels = int(spatial_channels)
        interaction_dim = int(interaction_dim)
        num_samples = int(num_samples)
        if semantic_channels <= 0 or spatial_channels <= 0 or interaction_dim <= 0:
            raise ValueError("SPSR channel dimensions must be positive")
        if num_samples != 4:
            raise ValueError("The SPSR preflight is fixed to K=4 samples")
        if offset_limit <= 0:
            raise ValueError("SPSR offset_limit must be positive")

        self.semantic_channels = semantic_channels
        self.spatial_channels = spatial_channels
        self.interaction_dim = interaction_dim
        self.num_samples = num_samples
        self.offset_limit = float(offset_limit)

        self.offset_predictor = nn.Sequential(
            nn.Conv2d(spatial_channels, interaction_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(interaction_dim, 2 * num_samples, kernel_size=1, bias=True),
        )
        # Learned offsets start at zero.  The fixed anchors still provide four
        # distinct source locations, so R-P is non-zero at initialization.
        nn.init.zeros_(self.offset_predictor[-1].weight)
        nn.init.zeros_(self.offset_predictor[-1].bias)

        self.query_projection = nn.Conv2d(
            spatial_channels, interaction_dim, kernel_size=1, bias=False
        )
        self.key_projection = nn.Conv2d(
            semantic_channels, interaction_dim, kernel_size=1, bias=False
        )
        # (x, y) offsets in source-token units.  They are deliberately fixed
        # and small; the learned predictor supplies only the residual offset.
        self.register_buffer(
            "anchor_offsets",
            torch.tensor(
                [
                    [-0.5, -0.5],
                    [-0.5, 0.5],
                    [0.5, -0.5],
                    [0.5, 0.5],
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    @staticmethod
    def _target_grid(height, width, device, dtype):
        # Pixel-centre coordinates for align_corners=False.  The resulting
        # grid maps a target location to the same normalized source position
        # before an anchor or learned source-token offset is added.
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        y = y * 2.0 - 1.0
        x = x * 2.0 - 1.0
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1)

    def forward(
        self,
        semantic_map,
        spatial_map,
        gamma,
        return_trace=False,
        return_monitor=False,
    ):
        if semantic_map.ndim != 4 or spatial_map.ndim != 4:
            raise ValueError("SPSR expects BCHW semantic and spatial maps")
        if semantic_map.shape[0] != spatial_map.shape[0]:
            raise ValueError("SPSR semantic/spatial batch sizes do not match")
        batch_size, channels, source_h, source_w = semantic_map.shape
        if channels != self.semantic_channels:
            raise ValueError(
                f"SPSR semantic channels mismatch: got {channels}, expected {self.semantic_channels}"
            )

        target_h, target_w = spatial_map.shape[-2:]
        offset_tokens = self.offset_predictor(spatial_map)
        offset_tokens = offset_tokens.reshape(
            batch_size, self.num_samples, 2, target_h, target_w
        ).permute(0, 1, 3, 4, 2)
        offset_tokens = torch.tanh(offset_tokens) * self.offset_limit

        base_grid = self._target_grid(
            target_h, target_w, semantic_map.device, semantic_map.dtype
        ).view(1, 1, target_h, target_w, 2)
        source_scale = semantic_map.new_tensor(
            [2.0 / source_w, 2.0 / source_h]
        ).view(1, 1, 1, 1, 2)
        anchor_grid = self.anchor_offsets.to(
            device=semantic_map.device, dtype=semantic_map.dtype
        ).view(1, self.num_samples, 1, 1, 2) * source_scale
        grids = base_grid + anchor_grid + offset_tokens * source_scale

        # Grid sampling is batched over the K fixed candidate points.  No
        # target-by-source attention matrix is materialized.
        sampled = F.grid_sample(
            semantic_map.unsqueeze(1)
            .expand(-1, self.num_samples, -1, -1, -1)
            .reshape(batch_size * self.num_samples, channels, source_h, source_w),
            grids.reshape(batch_size * self.num_samples, target_h, target_w, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled = sampled.reshape(
            batch_size, self.num_samples, channels, target_h, target_w
        )

        query = self.query_projection(spatial_map)
        key = self.key_projection(
            sampled.reshape(batch_size * self.num_samples, channels, target_h, target_w)
        ).reshape(
            batch_size, self.num_samples, self.interaction_dim, target_h, target_w
        )
        compatibility = (
            query.unsqueeze(1) * key
        ).sum(dim=2) / (self.interaction_dim ** 0.5)
        weights = torch.softmax(compatibility, dim=1)
        resampled = (weights.unsqueeze(2) * sampled).sum(dim=1)

        bilinear = F.interpolate(
            semantic_map,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        residual = resampled - bilinear
        output = bilinear + gamma * residual

        monitor = None
        if return_monitor:
            uniform_weight = 1.0 / float(self.num_samples)
            monitor = {
                "offset_signed_mean_source_tokens": float(
                    offset_tokens.detach().float().mean().item()
                ),
                "offset_abs_mean_source_tokens": float(
                    offset_tokens.detach().float().abs().mean().item()
                ),
                "offset_abs_max_source_tokens": float(
                    offset_tokens.detach().float().abs().max().item()
                ),
                "weight_abs_deviation_from_uniform_mean": float(
                    (weights.detach().float() - uniform_weight).abs().mean().item()
                ),
                "weight_abs_deviation_from_uniform_max": float(
                    (weights.detach().float() - uniform_weight).abs().max().item()
                ),
                "weight_std": float(weights.detach().float().std(unbiased=False).item()),
            }

        if not return_trace and not return_monitor:
            return output
        if not return_trace:
            return output, monitor
        if return_monitor:
            return output, {
                "trace": {
                    "spatial_map": spatial_map,
                    "offset_tokens": offset_tokens,
                    "grids": grids,
                    "sampled_values": sampled,
                    "compatibility": compatibility,
                    "weights": weights,
                    "resampled": resampled,
                    "bilinear": bilinear,
                    "residual": residual,
                },
                "monitor": monitor,
            }
        return output, {
            "spatial_map": spatial_map,
            "offset_tokens": offset_tokens,
            "grids": grids,
            "sampled_values": sampled,
            "compatibility": compatibility,
            "weights": weights,
            "resampled": resampled,
            "bilinear": bilinear,
            "residual": residual,
        }


SPSR_COMMON_SEEDS = {
    "semantic_projection": 61001,
    "spatial_projection_2": 61002,
    "spatial_projection_4": 61004,
    "spatial_projection_8": 61008,
    "ms_mlp": 61010,
    "resampler": 61020,
}


class L12SPMMSMLPDecoder(nn.Module):
    """Direct SPSR parent: L12 bilinear pyramid + Lite-SPM + MS-MLP.

    This is the required direct control for the SPSR experiment.  It has the
    same semantic projection, RGB stem, three spatial projections, and
    MS-MLP initialization as ``L12SPSRMSMLPDecoder``; it simply has no
    resampler branch or gamma parameters.
    """

    def __init__(self, backbone_channels, decoder_channels=256, num_classes=4):
        super().__init__()
        if int(decoder_channels) != 256:
            raise ValueError("The L12 SPM MS-MLP parent requires decoder_channels=256")
        self.backbone_channels = int(backbone_channels)
        self.decoder_channels = int(decoder_channels)
        self.spatial_interaction_dim = 64
        self.semantic_projection = _construct_with_fixed_seed(
            lambda: nn.Conv2d(
                self.backbone_channels, self.decoder_channels, kernel_size=1, bias=False
            ),
            SPSR_COMMON_SEEDS["semantic_projection"],
        )
        self.spatial_projections = nn.ModuleList(
            [
                _construct_with_fixed_seed(
                    lambda channels=channels: nn.Conv2d(
                        channels, self.spatial_interaction_dim, kernel_size=1, bias=False
                    ),
                    SPSR_COMMON_SEEDS[f"spatial_projection_{scale}"],
                )
                for channels, scale in ((32, 2), (64, 4), (128, 8))
            ]
        )
        self.ms_mlp = _construct_with_fixed_seed(
            lambda: MultiScaleMLPFusion(
                input_channels=self.decoder_channels,
                decoder_channels=self.decoder_channels,
                num_classes=num_classes,
            ),
            SPSR_COMMON_SEEDS["ms_mlp"],
        )

    @staticmethod
    def _tokens_to_feature_map(tokens, patch_h, patch_w):
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        if tokens.ndim != 3:
            raise ValueError(
                f"L12 SPM MS-MLP expects patch tokens [B,N,C], got {tuple(tokens.shape)}"
            )
        expected = int(patch_h * patch_w)
        if tokens.shape[1] != expected:
            raise ValueError(
                "L12 SPM MS-MLP requires patch-only L12 tokens with no CLS/register tokens: "
                f"got N={tokens.shape[1]}, expected {expected}"
            )
        return tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[-1], patch_h, patch_w
        )

    def _project_l12(self, semantic_tokens, patch_h, patch_w):
        return self.semantic_projection(
            self._tokens_to_feature_map(semantic_tokens, patch_h, patch_w)
        )

    @staticmethod
    def _build_pyramid(semantic_map):
        return (
            F.interpolate(semantic_map, scale_factor=8, mode="bilinear", align_corners=False),
            F.interpolate(semantic_map, scale_factor=4, mode="bilinear", align_corners=False),
            F.interpolate(semantic_map, scale_factor=2, mode="bilinear", align_corners=False),
            semantic_map,
        )

    def forward(self, semantic_tokens, spatial_features, patch_h, patch_w):
        if len(spatial_features) != 3:
            raise ValueError(f"L12 SPM MS-MLP requires D2/D4/D8, got {len(spatial_features)} maps")
        semantic_map = self._project_l12(semantic_tokens, patch_h, patch_w)
        p2, p4, p8, p16 = self._build_pyramid(semantic_map)
        expected = (
            (patch_h * 8, patch_w * 8),
            (patch_h * 4, patch_w * 4),
            (patch_h * 2, patch_w * 2),
            (patch_h, patch_w),
        )
        actual = tuple(feature.shape[-2:] for feature in (*spatial_features, semantic_map))
        if actual != expected:
            raise RuntimeError(f"L12 SPM/MS-MLP scale mismatch: got {actual}, expected {expected}")
        # The direct parent keeps the identical Lite-SPM modules in the model
        # graph, but has no semantic-spatial interaction.  Its active path is
        # the bilinear L12 pyramid -> MS-MLP fallback; the candidate adds only
        # the SPSR branch between those two endpoints.
        return self.ms_mlp((p2, p4, p8, p16))


class L12SPSRMSMLPDecoder(L12SPMMSMLPDecoder):
    """L12 semantic anchor + RGB SPM + shared SPSR + neutral MS-MLP.

    This is a preflightable prototype, not a trained claim.  The bilinear
    L12 pyramid remains the identity-safe fallback.  At gamma=0, each SPSR
    output is exactly its corresponding bilinear map and the final logits
    equal the fallback MS-MLP logits.
    """

    def __init__(self, backbone_channels, decoder_channels=256, num_classes=4):
        super().__init__(backbone_channels, decoder_channels=decoder_channels, num_classes=num_classes)
        if int(decoder_channels) != 256:
            raise ValueError("The SPSR prototype requires decoder_channels=256")
        self.num_samples = 4
        self.resampler = _construct_with_fixed_seed(
            lambda: SpatialGuidedSemanticResampler(
                semantic_channels=self.decoder_channels,
                spatial_channels=self.spatial_interaction_dim,
                interaction_dim=self.spatial_interaction_dim,
                num_samples=self.num_samples,
                offset_limit=1.0,
            ),
            SPSR_COMMON_SEEDS["resampler"],
        )
        self.gammas = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(3)]
        )
        self.spsr_monitor_enabled = False
        self._last_spsr_monitor = None

    @staticmethod
    def _tokens_to_feature_map(tokens, patch_h, patch_w):
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        if tokens.ndim != 3:
            raise ValueError(
                f"SPSR expects patch tokens [B,N,C], got {tuple(tokens.shape)}"
            )
        expected = int(patch_h * patch_w)
        if tokens.shape[1] != expected:
            raise ValueError(
                "SPSR requires patch-only L12 tokens with no CLS/register tokens: "
                f"got N={tokens.shape[1]}, expected {expected}"
            )
        return tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[-1], patch_h, patch_w
        )

    def _project_l12(self, semantic_tokens, patch_h, patch_w):
        return self.semantic_projection(
            self._tokens_to_feature_map(semantic_tokens, patch_h, patch_w)
        )

    def _build_pyramid(self, semantic_map):
        return (
            F.interpolate(
                semantic_map, scale_factor=8, mode="bilinear", align_corners=False
            ),
            F.interpolate(
                semantic_map, scale_factor=4, mode="bilinear", align_corners=False
            ),
            F.interpolate(
                semantic_map, scale_factor=2, mode="bilinear", align_corners=False
            ),
            semantic_map,
        )

    def _forward_impl(
        self, semantic_tokens, spatial_features, patch_h, patch_w, return_diagnostics=False
    ):
        if len(spatial_features) != 3:
            raise ValueError(f"SPSR requires D2/D4/D8, got {len(spatial_features)} maps")
        semantic_map = self._project_l12(semantic_tokens, patch_h, patch_w)
        p2, p4, p8, p16 = self._build_pyramid(semantic_map)
        expected = (
            (patch_h * 8, patch_w * 8),
            (patch_h * 4, patch_w * 4),
            (patch_h * 2, patch_w * 2),
            (patch_h, patch_w),
        )
        actual = tuple(feature.shape[-2:] for feature in (*spatial_features, semantic_map))
        if actual != expected:
            raise RuntimeError(f"SPSR/SPM scale mismatch: got {actual}, expected {expected}")

        outputs = []
        traces = []
        monitors = []
        monitor_enabled = self.spsr_monitor_enabled and not return_diagnostics
        for spatial_projection, pyramid, spatial_feature, gamma in zip(
            self.spatial_projections,
            (p2, p4, p8),
            spatial_features,
            self.gammas,
        ):
            projected_spatial = spatial_projection(spatial_feature)
            result = self.resampler(
                pyramid,
                projected_spatial,
                gamma,
                return_trace=return_diagnostics,
                return_monitor=monitor_enabled,
            )
            if return_diagnostics:
                result, trace = result
                trace["projected_spatial"] = projected_spatial
                trace["gamma"] = gamma
                traces.append(trace)
            elif monitor_enabled:
                result, monitor = result
                monitors.append(monitor)
            outputs.append(result)

        candidate_logits = self.ms_mlp((*outputs, p16))
        if not return_diagnostics:
            if monitor_enabled:
                self._last_spsr_monitor = {
                    "scales": monitors,
                    "gamma_before_step": [
                        float(gamma.detach().item()) for gamma in self.gammas
                    ],
                }
            else:
                self._last_spsr_monitor = None
            return candidate_logits

        fallback_logits = self.ms_mlp((p2, p4, p8, p16))
        return candidate_logits, {
            "semantic_map": semantic_map,
            "pyramid": (p2, p4, p8, p16),
            "outputs": tuple(outputs),
            "projected_spatial": tuple(trace["projected_spatial"] for trace in traces),
            "resamplers": tuple(traces),
            "fallback_logits": fallback_logits,
        }

    def forward(self, semantic_tokens, spatial_features, patch_h, patch_w):
        return self._forward_impl(
            semantic_tokens,
            spatial_features,
            patch_h,
            patch_w,
            return_diagnostics=False,
        )

    def forward_with_diagnostics(
        self, semantic_tokens, spatial_features, patch_h, patch_w
    ):
        return self._forward_impl(
            semantic_tokens,
            spatial_features,
            patch_h,
            patch_w,
            return_diagnostics=True,
        )

    def set_monitor_enabled(self, enabled):
        self.spsr_monitor_enabled = bool(enabled)

    def get_learning_snapshot(self):
        if self._last_spsr_monitor is None:
            return None
        gradient_map = {
            "offset_final_weight_grad_norm": _safe_grad_norm(
                self.resampler.offset_predictor[-1].weight
            ),
            "offset_final_bias_grad_norm": _safe_grad_norm(
                self.resampler.offset_predictor[-1].bias
            ),
            "offset_hidden_grad_norm": _safe_grad_norm(
                self.resampler.offset_predictor[0].weight
            ),
            "query_grad_norm": _safe_grad_norm(self.resampler.query_projection.weight),
            "key_grad_norm": _safe_grad_norm(self.resampler.key_projection.weight),
        }
        return {
            "scales": [dict(value) for value in self._last_spsr_monitor["scales"]],
            "gamma_before_step": list(self._last_spsr_monitor["gamma_before_step"]),
            "gradients_before_step": gradient_map,
        }


class RTDETRConvNormLayer(nn.Module):
    """The ConvNormLayer used by the official RT-DETR HybridEncoder."""

    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        if act is None or act == "identity":
            self.act = nn.Identity()
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported RT-DETR activation: {act}")

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RTDETRRepVggBlock(nn.Module):
    """Training-time RepVGG block copied from RT-DETR's CCFM path."""

    def __init__(self, ch_in, ch_out, act="silu"):
        super().__init__()
        self.conv1 = RTDETRConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = RTDETRConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act in (None, "identity"):
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported RT-DETR activation: {act}")

    def forward(self, x):
        return self.act(self.conv1(x) + self.conv2(x))


class RTDETRCSPRepLayer(nn.Module):
    """The official RT-DETR CSPRepLayer, used without detection components."""

    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=None, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = RTDETRConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = RTDETRConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[
                RTDETRRepVggBlock(hidden_channels, hidden_channels, act=act)
                for _ in range(round(3 * 1.0) if num_blocks is None else int(num_blocks))
            ]
        )
        if hidden_channels != out_channels:
            self.conv3 = RTDETRConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.bottlenecks(self.conv1(x))
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class FourScaleRTDETRCCFM(nn.Module):
    """Four-scale FPN+PAN CCFM adapted from RT-DETR's HybridEncoder.

    RT-DETR's released encoder is normally configured for three input levels.
    This OSD adapter keeps its input projection, nearest-neighbor top-down
    FPN, CSPRep blocks, stride-2 bottom-up PAN, and CSPRep blocks, while
    extending the same loops to four ordered levels:
    ``[1/2, 1/4, 1/8, 1/16]``.

    The transformer encoder and all RT-DETR detection/query components are
    intentionally absent.  This module is only the requested cross-scale
    feature fusion path.
    """

    def __init__(
        self,
        in_channels=(32, 64, 128, 384),
        hidden_dim=256,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
    ):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"FourScaleRTDETRCCFM requires four inputs, got {len(in_channels)}")
        self.in_channels = tuple(int(value) for value in in_channels)
        self.hidden_dim = int(hidden_dim)
        self.out_channels = (self.hidden_dim,) * 4
        self.out_strides = (2, 4, 8, 16)

        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, self.hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.hidden_dim),
                )
                for channels in self.in_channels
            ]
        )

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.lateral_convs.append(
                RTDETRConvNormLayer(self.hidden_dim, self.hidden_dim, 1, 1, act=act)
            )
            self.fpn_blocks.append(
                RTDETRCSPRepLayer(
                    self.hidden_dim * 2,
                    self.hidden_dim,
                    num_blocks=round(3 * depth_mult),
                    expansion=expansion,
                    act=act,
                )
            )

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.downsample_convs.append(
                RTDETRConvNormLayer(self.hidden_dim, self.hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                RTDETRCSPRepLayer(
                    self.hidden_dim * 2,
                    self.hidden_dim,
                    num_blocks=round(3 * depth_mult),
                    expansion=expansion,
                    act=act,
                )
            )

    @staticmethod
    def _require_pair(actual, expected, context):
        if tuple(actual) != tuple(expected):
            raise RuntimeError(f"{context} shape mismatch: got {tuple(actual)}, expected {tuple(expected)}")

    def forward(self, features):
        if len(features) != 4:
            raise ValueError(f"FourScaleRTDETRCCFM requires four feature maps, got {len(features)}")
        proj_feats = [projection(feature) for projection, feature in zip(self.input_proj, features)]

        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feature_high = self.lateral_convs[len(self.in_channels) - 1 - idx](inner_outs[0])
            inner_outs[0] = feature_high
            upsampled = F.interpolate(feature_high, scale_factor=2.0, mode="nearest")
            feature_low = proj_feats[idx - 1]
            self._require_pair(upsampled.shape[-2:], feature_low.shape[-2:], "CCFM top-down")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsampled, feature_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        outputs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feature_low = outputs[-1]
            feature_high = inner_outs[idx + 1]
            downsampled = self.downsample_convs[idx](feature_low)
            self._require_pair(downsampled.shape[-2:], feature_high.shape[-2:], "CCFM bottom-up")
            outputs.append(
                self.pan_blocks[idx](torch.cat([downsampled, feature_high], dim=1))
            )
        return tuple(outputs)


class SPMCCFMDecoderBase(nn.Module):
    """Common RGB-SPM + four-scale CCFM front-end for both backends."""

    def __init__(self, backbone_channels, decoder_channels=256):
        super().__init__()
        self.backbone_channels = int(backbone_channels)
        self.decoder_channels = int(decoder_channels)
        self.ccfm = FourScaleRTDETRCCFM(
            in_channels=(32, 64, 128, self.backbone_channels),
            hidden_dim=self.decoder_channels,
        )

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

    def build_ccfm_features(self, semantic_tokens, spatial_features, patch_h, patch_w):
        if len(spatial_features) != 3:
            raise ValueError(f"SPM requires D2/D4/D8, got {len(spatial_features)} maps")
        semantic_map = self._tokens_to_feature_map(semantic_tokens, patch_h, patch_w)
        expected = (
            (patch_h * 8, patch_w * 8),
            (patch_h * 4, patch_w * 4),
            (patch_h * 2, patch_w * 2),
            (patch_h, patch_w),
        )
        actual = tuple(feature.shape[-2:] for feature in (*spatial_features, semantic_map))
        if actual != expected:
            raise RuntimeError(f"SPM/CCFM input scales mismatch: got {actual}, expected {expected}")
        return self.ccfm((*spatial_features, semantic_map))


class SPMCCFMSADDecoder(SPMCCFMDecoderBase):
    """SPM+CCFM followed by the original OSD SAD consumer."""

    def __init__(self, backbone_channels, decoder_channels=256, num_classes=4, use_group_norm=True):
        super().__init__(backbone_channels, decoder_channels=decoder_channels)
        self.sad_intra_1 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_intra_2 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_intra_3 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_intra_4 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_inter_4 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_inter_3 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_inter_2 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.sad_inter_1 = ResidualDepthwiseBlock(decoder_channels, use_group_norm=use_group_norm)
        self.out_conv = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward(self, semantic_tokens, spatial_features, patch_h, patch_w):
        p2, p4, p8, p16 = self.build_ccfm_features(
            semantic_tokens, spatial_features, patch_h, patch_w
        )
        level_1 = self.sad_intra_1(p2)
        level_2 = self.sad_intra_2(p4)
        level_3 = self.sad_intra_3(p8)
        level_4 = self.sad_intra_4(p16)
        x4 = self.sad_inter_4(level_4)
        x3 = self.sad_inter_3(
            F.interpolate(x4, size=level_3.shape[-2:], mode="bilinear", align_corners=False)
            + level_3
        )
        x2 = self.sad_inter_2(
            F.interpolate(x3, size=level_2.shape[-2:], mode="bilinear", align_corners=False)
            + level_2
        )
        x1 = self.sad_inter_1(
            F.interpolate(x2, size=level_1.shape[-2:], mode="bilinear", align_corners=False)
            + level_1
        )
        return self.out_conv(x1)


class SPMSADDecoder(nn.Module):
    """Direct Lite-SPM pyramid followed by the original OSD SAD consumer.

    This is the clean CTRL-002 replacement experiment: RGB Lite-SPM supplies
    P2/P4/P8, the frozen DINOv3 L12 map supplies P16, and those four maps enter
    the unchanged SAD path directly.  There is deliberately no TPA resample
    branch and no CCFM between the spatial prior and SAD.
    """

    def __init__(self, backbone_channels, decoder_channels=128, num_classes=2, use_group_norm=True):
        super().__init__()
        self.spatial_projections = nn.ModuleList(
            [
                nn.Conv2d(32, decoder_channels, 1, bias=False),
                nn.Conv2d(64, decoder_channels, 1, bias=False),
                nn.Conv2d(128, decoder_channels, 1, bias=False),
            ]
        )
        self.semantic_projection = nn.Conv2d(
            backbone_channels, decoder_channels, 1, bias=False
        )

        # Keep the SAD blocks identical to the current OSD implementation.
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
        return tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[-1], patch_h, patch_w
        )

    def forward(self, semantic_tokens, spatial_features, patch_h, patch_w):
        if len(spatial_features) != 3:
            raise ValueError(f"SPM/SAD requires D2/D4/D8, got {len(spatial_features)} maps")
        p2, p4, p8 = [
            projection(feature)
            for projection, feature in zip(self.spatial_projections, spatial_features)
        ]
        p16 = self.semantic_projection(
            self._tokens_to_feature_map(semantic_tokens, patch_h, patch_w)
        )

        expected = (
            (patch_h * 8, patch_w * 8),
            (patch_h * 4, patch_w * 4),
            (patch_h * 2, patch_w * 2),
            (patch_h, patch_w),
        )
        actual = tuple(feature.shape[-2:] for feature in (p2, p4, p8, p16))
        if actual != expected:
            raise RuntimeError(f"SPM/SAD input shape mismatch: got {actual}, expected {expected}")

        level_1 = self.sad_intra_1(p2)
        level_2 = self.sad_intra_2(p4)
        level_3 = self.sad_intra_3(p8)
        level_4 = self.sad_intra_4(p16)

        x4 = self.sad_inter_4(level_4)
        x3 = self.sad_inter_3(
            F.interpolate(x4, size=level_3.shape[-2:], mode="bilinear", align_corners=False)
            + level_3
        )
        x2 = self.sad_inter_2(
            F.interpolate(x3, size=level_2.shape[-2:], mode="bilinear", align_corners=False)
            + level_2
        )
        x1 = self.sad_inter_1(
            F.interpolate(x2, size=level_1.shape[-2:], mode="bilinear", align_corners=False)
            + level_1
        )
        return self.out_conv(x1)


class SPMCCFMMSMLPDecoder(SPMCCFMDecoderBase):
    """SPM+CCFM followed by the existing neutral MS-MLP consumer."""

    def __init__(self, backbone_channels, decoder_channels=256, num_classes=4):
        super().__init__(backbone_channels, decoder_channels=decoder_channels)
        self.ms_mlp = MultiScaleMLPFusion(
            input_channels=decoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )

    def forward(self, semantic_tokens, spatial_features, patch_h, patch_w):
        ccfm_features = self.build_ccfm_features(
            semantic_tokens, spatial_features, patch_h, patch_w
        )
        return self.ms_mlp(ccfm_features)


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


class TPAChangeCascadeDecoder(nn.Module):
    """TPA pyramid with a ChangeViT-style cascade consumer.

    The official ChangeViT decoder is designed for bi-temporal change
    detection and therefore also contains pairwise difference modeling and a
    feature injector.  OSD is single-image semantic segmentation, so this
    adapter intentionally keeps only the reusable cascade topology:
    deepest-to-shallowest 1x1 channel alignment, transposed-convolution
    upsampling, and additive fusion.  The existing OSD TPA/PR path is kept
    unchanged.
    """

    def __init__(self, in_dims, decoder_channels=256, num_classes=4):
        super().__init__()
        if len(in_dims) != 4:
            raise ValueError(f"TPAChangeCascadeDecoder requires four input dims, got {len(in_dims)}")
        self.token_projections = nn.ModuleList(
            [nn.Conv2d(channels, decoder_channels, 1, bias=False) for channels in in_dims]
        )
        self.tpa_branch_1 = TPAResampleProject(decoder_channels, scale_factor=8)
        self.tpa_branch_2 = TPAResampleProject(decoder_channels, scale_factor=4)
        self.tpa_branch_3 = TPAResampleProject(decoder_channels, scale_factor=2)
        self.tpa_branch_4 = TPAResampleProject(decoder_channels, scale_factor=1)

        # ChangeViT's cascade uses Conv1x1 followed by a 4x4, stride-2
        # transposed convolution at each coarse-to-fine transition.  All OSD
        # pyramid branches have the same decoder width, so channel alignment
        # remains decoder_channels -> decoder_channels.
        self.up_p16 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 1, bias=False),
            nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=4, stride=2, padding=1),
        )
        self.up_p8 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 1, bias=False),
            nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=4, stride=2, padding=1),
        )
        self.up_p4 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 1, bias=False),
            nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=4, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
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

    def _project_tokens(self, features, patch_h, patch_w):
        if len(features) != 4:
            raise ValueError(f"TPAChangeCascadeDecoder requires four features, got {len(features)}")
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

    def _cascade(self, pyramid):
        if len(pyramid) != 4:
            raise ValueError(f"ChangeViT cascade requires four pyramid levels, got {len(pyramid)}")
        p2, p4, p8, p16 = pyramid
        y16 = p16
        y8 = p8 + self.up_p16(y16)
        y4 = p4 + self.up_p8(y8)
        y2 = p2 + self.up_p4(y4)
        return y16, y8, y4, y2

    def forward(self, features, patch_h, patch_w):
        projected = self._project_tokens(features, patch_h, patch_w)
        pyramid = self._build_tpa_pyramid(projected)
        _, _, _, y2 = self._cascade(pyramid)
        return self.classifier(y2)


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
        query_position = fixed_2d_sincos(
            patch_h,
            patch_w,
            self.cli.interaction_channels,
            device=memory.device,
            dtype=memory.dtype,
        ).unsqueeze(0)
        calibrated, cli_trace = self.cli(
            projected,
            memory,
            ltp_trace["memory_position"],
            query_position,
            return_trace=True,
        )
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
            "memory_position": ltp_trace["memory_position"],
            "query_position": query_position,
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


class OfficialDINOv3AdapterDecoder(nn.Module):
    """OSD wrapper around Meta's official DINOv3 segmentation adapter.

    The adapter and its linear head are imported from the official DINOv3
    checkout selected by ``runtime.load_backbone``.  This positive control does
    not reuse the OSD TPA/PR/SAD path: it uses the official Spatial Prior
    Module plus multi-scale deformable interaction, followed by the official
    linear dense-prediction head adapted only to four OSD classes.

    ViT-S has twelve blocks, so the official four-even-interval rule resolves
    to zero-based interaction indices ``[2, 5, 8, 11]``.  All other adapter
    arguments retain the official DINOv3 segmentation defaults.
    """

    def __init__(self, backbone, num_classes=4):
        super().__init__()
        try:
            import dinov3

            # DINOv3's released segmentation op uses the newer torch.amp
            # decorator spelling.  PyTorch 2.2 exposes the equivalent CUDA
            # decorators under torch.cuda.amp; bridge only the decorator API
            # in-process so the official operator implementation is unchanged.
            if not hasattr(torch.amp, "custom_fwd"):
                from torch.cuda.amp import custom_bwd as _cuda_custom_bwd
                from torch.cuda.amp import custom_fwd as _cuda_custom_fwd

                def _compat_custom_fwd(fn=None, *, device_type=None, cast_inputs=None):
                    decorator = lambda target: _cuda_custom_fwd(
                        target, cast_inputs=cast_inputs
                    )
                    return decorator if fn is None else decorator(fn)

                def _compat_custom_bwd(fn=None, *, device_type=None):
                    decorator = lambda target: _cuda_custom_bwd(target)
                    return decorator if fn is None else decorator(fn)

                torch.amp.custom_fwd = _compat_custom_fwd
                torch.amp.custom_bwd = _compat_custom_bwd

            # The official setup.py places the compiled extension beside the
            # official segmentation ops.  Make that directory importable for
            # this process without copying or modifying the official source.
            official_ops = (
                Path(dinov3.__file__).resolve().parent
                / "eval"
                / "segmentation"
                / "models"
                / "utils"
                / "ops"
            )
            if official_ops.is_dir() and str(official_ops) not in sys.path:
                sys.path.insert(0, str(official_ops))
            import MultiScaleDeformableAttention  # noqa: F401

            # Importing the official adapter submodule through the normal
            # package path executes the official segmentation registry first,
            # which pulls the optional torchmetrics dependency.  The adapter
            # and LinearHead themselves do not need that registry, so expose
            # only the official package path for this import.
            official_models = official_ops.parent.parent
            package_name = "dinov3.eval.segmentation.models"
            if package_name not in sys.modules:
                package = types.ModuleType(package_name)
                package.__path__ = [str(official_models)]
                package.__package__ = package_name
                sys.modules[package_name] = package

            from dinov3.eval.segmentation.models.backbone.dinov3_adapter import (
                DINOv3_Adapter,
            )
            from dinov3.eval.segmentation.models.heads.linear_head import LinearHead
        except ImportError as exc:
            raise ImportError(
                "The official DINOv3 segmentation adapter is unavailable. "
                "Ensure model.dino_repo points to the official DINOv3 checkout "
                "and its MSDeformableAttention extension is installed."
            ) from exc

        if int(getattr(backbone, "n_blocks", 0)) != 12:
            raise ValueError(
                "The OSD positive control is configured for DINOv3-S/16 with "
                f"12 blocks, got n_blocks={getattr(backbone, 'n_blocks', None)}"
            )
        embed_dim = int(backbone.embed_dim)
        self.adapter = DINOv3_Adapter(
            backbone,
            interaction_indexes=[2, 5, 8, 11],
            pretrain_size=512,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=16,
            drop_path_rate=0.3,
            init_values=0.0,
            with_cffn=True,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            add_vit_feature=True,
            use_extra_extractor=True,
            with_cp=True,
        )
        # Official DINOv3 linear dense-prediction head; only the class count is
        # adapted from the upstream segmentation task to OSD's four classes.
        self.head = LinearHead(
            in_channels=[embed_dim] * 4,
            n_output_channels=int(num_classes),
            use_batchnorm=True,
            use_cls_token=False,
            dropout=0.1,
        )
        self.output_feature_strides = (4, 8, 16, 32)
        self.official_interaction_indexes = (2, 5, 8, 11)

    def forward(self, x):
        features = self.adapter(x)
        ordered = [features[str(index)] for index in range(1, 5)]
        return self.head(ordered)


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
            "tpa_sad_msef",
            "tpa_ms_mlp",
            "tpa_change_cascade",
            "dpa_ms_mlp",
            "tpa_sad_base",
            "dpa_sad_base",
            "patch_dpa_shared",
            "patch_dpa_independent",
            "patch_guided_sad",
            "ltp_cli",
            "semantic_spatial",
            "mlp_same_scale",
            "spm_ccfm_sad",
            "spm_ccfm_ms_mlp",
            "spm_sad",
            "l12_spm_ms_mlp",
            "l12_spsr_ms_mlp",
            "tpa_l12_shared_bilinear",
            "tpa_l12_shared_conv",
            "tpa_l12_independent_bilinear",
            "tpa_l12_independent_conv",
            "l12_a_msef",
            "l12_a_cross_msef",
            "l12_a_cross_r",
            "l12_a_cross_dweca",
            "l12_a_cross_bottleneck64",
            "l12_a_cross_msef_cdr",
            "l12_a_cross_msef_ds",
            "l12_a_cross_ee",
            "l12_a_cross_see",
            "l12_a_cross_r_weighted",
            "l12_a_cross_r_dysample_splus",
            "dinov3_adapter",
        }:
            raise ValueError(
                f"Unknown decoder_variant '{self.decoder_variant}'. "
                "Expected 'tpa_sad', 'tpa_sad_msef', 'tpa_ms_mlp', 'dpa_ms_mlp', "
                "'tpa_change_cascade', "
                "'tpa_sad_base', 'dpa_sad_base', 'patch_dpa_shared', "
                "'patch_dpa_independent', 'patch_guided_sad', 'ltp_cli', 'semantic_spatial', "
                "'mlp_same_scale', 'spm_ccfm_sad', 'spm_ccfm_ms_mlp', 'spm_sad', "
                "'l12_spm_ms_mlp', 'l12_spsr_ms_mlp', or "
                "an L12 factorial/MSEF variant."
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
        if self.decoder_variant in SPM_VARIANTS:
            if self.layer_mapping is not None:
                raise ValueError(
                    f"{self.decoder_variant} uses L12 only and requires layer_mapping=null"
                )
            if self.wcf_enabled or adaptive_readout:
                raise ValueError(f"{self.decoder_variant} cannot combine WCF or ALSR")
        if self.decoder_variant == "l12_a_msef":
            if self.layer_mapping != [3, 3, 3, 3]:
                raise ValueError(
                    "l12_a_msef requires factorial-A routing layer_mapping=[3,3,3,3]"
                )
            if self.wcf_enabled or adaptive_readout:
                raise ValueError("l12_a_msef cannot combine WCF or ALSR")
        if self.decoder_variant in CROSS_MSEF_VARIANTS:
            if self.layer_mapping is not None:
                raise ValueError(f"{self.decoder_variant} uses the single final semantic source and requires layer_mapping=null")
            if self.wcf_enabled or adaptive_readout:
                raise ValueError(f"{self.decoder_variant} cannot combine WCF or ALSR")
        if self.decoder_variant == "dinov3_adapter":
            if self.layer_mapping is not None:
                raise ValueError("dinov3_adapter uses official [L3,L6,L9,L12] interactions and cannot remap layers")
            if self.wcf_enabled or adaptive_readout:
                raise ValueError("dinov3_adapter cannot combine WCF or ALSR")
        self.in_dims = [self.backbone.embed_dim] * 4
        self.spm_stem = (
            LiteSpatialPriorStem(in_channels=3, channels=(32, 64, 128))
            if self.decoder_variant in (SPM_VARIANTS | CROSS_MSEF_VARIANTS)
            else None
        )
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
        elif self.decoder_variant == "tpa_sad_msef":
            self.decoder = TPASADMSEFDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
                use_group_norm=not use_bn,
                adaptive_readout=adaptive_readout,
                readout_mode=readout_mode,
                readout_init=readout_init,
                readout_temperature=readout_temperature,
                msef_reduction=16,
            )
        elif self.decoder_variant == "tpa_ms_mlp":
            self.decoder = TPAMultiScaleMLPDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "tpa_change_cascade":
            self.decoder = TPAChangeCascadeDecoder(
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
        elif self.decoder_variant == "spm_ccfm_sad":
            self.decoder = SPMCCFMSADDecoder(
                backbone_channels=self.backbone.embed_dim,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
                use_group_norm=not use_bn,
            )
        elif self.decoder_variant == "spm_ccfm_ms_mlp":
            self.decoder = SPMCCFMMSMLPDecoder(
                backbone_channels=self.backbone.embed_dim,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "spm_sad":
            self.decoder = SPMSADDecoder(
                backbone_channels=self.backbone.embed_dim,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
                use_group_norm=not use_bn,
            )
        elif self.decoder_variant == "l12_spsr_ms_mlp":
            self.decoder = L12SPSRMSMLPDecoder(
                backbone_channels=self.backbone.embed_dim,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "l12_spm_ms_mlp":
            self.decoder = L12SPMMSMLPDecoder(
                backbone_channels=self.backbone.embed_dim,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant in {
            "tpa_l12_shared_bilinear",
            "tpa_l12_shared_conv",
            "tpa_l12_independent_bilinear",
            "tpa_l12_independent_conv",
        }:
            self.decoder = TPAL12FactorialDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
                use_group_norm=not use_bn,
                projection_mode=(
                    "shared"
                    if self.decoder_variant in {
                        "tpa_l12_shared_bilinear",
                        "tpa_l12_shared_conv",
                    }
                    else "independent"
                ),
                use_spatial_conv=self.decoder_variant in {
                    "tpa_l12_shared_conv",
                    "tpa_l12_independent_conv",
                },
            )
        elif self.decoder_variant == "l12_a_msef":
            self.decoder = L12AMSEFDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "l12_a_cross_msef":
            self.decoder = L12ACrossMSEFDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "l12_a_cross_r":
            self.decoder = L12ACrossRDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "l12_a_cross_dweca":
            self.decoder = L12ACrossDWECADecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "l12_a_cross_bottleneck64":
            self.decoder = L12ACrossBottleneck64Decoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "l12_a_cross_r_weighted":
            self.decoder = L12ACrossWeightedRDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "l12_a_cross_r_dysample_splus":
            self.decoder = L12ACrossDySampleRDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "l12_a_cross_msef_cdr":
            self.decoder = L12ACDRDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "l12_a_cross_msef_ds":
            self.decoder = L12ACrossMSEFDeepSupervisionDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
            )
        elif self.decoder_variant == "l12_a_cross_ee":
            self.decoder = L12ACrossEEBaseDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
                edge_activation="sigmoid",
            )
        elif self.decoder_variant == "l12_a_cross_see":
            self.decoder = L12ACrossEEBaseDecoder(
                self.in_dims,
                decoder_channels=decoder_channels,
                num_classes=self.nclass,
                edge_activation="tanh",
            )
        elif self.decoder_variant == "dinov3_adapter":
            self.decoder = OfficialDINOv3AdapterDecoder(
                self.backbone,
                num_classes=self.nclass,
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

    def set_spsr_monitor(self, enabled):
        """Enable lightweight early-training SPSR learning diagnostics."""
        setter = getattr(self.decoder, "set_monitor_enabled", None)
        if setter is not None:
            setter(enabled)

    def get_spsr_learning_snapshot(self):
        getter = getattr(self.decoder, "get_learning_snapshot", None)
        if getter is None:
            return None
        return getter()

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
        wcf_gates = []
        if self.decoder_variant == "dinov3_adapter":
            out = self.decoder(x)
            returned_feature = None
        elif self.decoder_variant in CROSS_MSEF_VARIANTS:
            final_layer_idx = self.intermediate_layer_idx[self.encoder_size][-1]
            if self._backbone_locked:
                with torch.no_grad():
                    semantic_features = self.backbone.get_intermediate_layers(
                        x, n=[final_layer_idx]
                    )
            else:
                semantic_features = self.backbone.get_intermediate_layers(
                    x, n=[final_layer_idx]
                )
            semantic_tokens = semantic_features[-1]
            spatial_features = self.spm_stem(x)
            out = self.decoder(
                semantic_tokens,
                spatial_features,
                patch_h,
                patch_w,
            )
            returned_feature = semantic_tokens
        elif self.decoder_variant in SPM_VARIANTS:
            final_layer_idx = self.intermediate_layer_idx[self.encoder_size][-1]
            if self._backbone_locked:
                with torch.no_grad():
                    semantic_features = self.backbone.get_intermediate_layers(
                        x, n=[final_layer_idx]
                    )
            else:
                semantic_features = self.backbone.get_intermediate_layers(
                    x, n=[final_layer_idx]
                )
            semantic_tokens = semantic_features[-1]
            spatial_features = self.spm_stem(x)
            out = self.decoder(
                semantic_tokens,
                spatial_features,
                patch_h,
                patch_w,
            )
            returned_feature = semantic_tokens
        elif self.decoder_variant == "semantic_spatial":
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

        if self.decoder_variant not in (
            SPM_VARIANTS | {"semantic_spatial"} | CROSS_MSEF_VARIANTS
        ) and self.decoder_variant != "dinov3_adapter":
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

    def forward_with_aux(self, x):
        """Return final and auxiliary logits for the DS-only experiment.

        The ordinary ``forward`` remains inference-compatible and returns only
        the final logits.  Auxiliary heads are attached after inter3, inter2,
        and inter1 inside the Cross-MSEF SAD decoder and are used only by the
        training runner.
        """
        if self.decoder_variant not in DEEP_SUPERVISION_VARIANTS:
            raise RuntimeError(
                "forward_with_aux is only available for the deep-supervision variant"
            )
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        final_layer_idx = self.intermediate_layer_idx[self.encoder_size][-1]
        if self._backbone_locked:
            with torch.no_grad():
                semantic_features = self.backbone.get_intermediate_layers(
                    x, n=[final_layer_idx]
                )
        else:
            semantic_features = self.backbone.get_intermediate_layers(
                x, n=[final_layer_idx]
            )
        semantic_tokens = semantic_features[-1]
        spatial_features = self.spm_stem(x)
        lowres_logits, lowres_aux = self.decoder.forward_with_aux(
            semantic_tokens,
            spatial_features,
            patch_h,
            patch_w,
        )
        output_size = x.shape[-2:]
        final_logits = F.interpolate(
            lowres_logits, size=output_size, mode="bilinear", align_corners=False
        )
        aux_logits = tuple(
            F.interpolate(aux, size=output_size, mode="bilinear", align_corners=False)
            for aux in lowres_aux
        )
        return final_logits, aux_logits

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
