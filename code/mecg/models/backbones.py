"""
1D backbones for ECG representation learning.

Changes vs. the original submission (driven by reviewer comments):

* `Inception1D` now has an OPTIONAL squeeze-and-excitation path. The submitted
  manuscript described the Inception1D baseline as using SE attention, but the
  released code did not implement it. `use_se` makes the claim explicit and
  testable; set it in the config and report what you actually ran.
* Residual connections in `Inception1D` follow InceptionTime (Ismail Fawaz et
  al.) and are applied every `residual_every` blocks rather than every block.
* Channel width is configurable so that `inception1d` can be instantiated as a
  faithful InceptionTime (constant width) or as the wider variant used in the
  original submission.
* All backbones expose `embedding_dim` and `get_num_params()`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
class SEBlock1d(nn.Module):
    """Squeeze-and-excitation for 1D feature maps."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


class ConvBlock1d(nn.Module):
    """Conv1d -> BatchNorm1d -> (ReLU)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None,
                 groups=1, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_ch),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock1d(nn.Module):
    """Bottleneck residual block with optional SE."""

    def __init__(self, in_ch, mid_ch, out_ch, stride=1, se=True):
        super().__init__()
        self.conv1 = ConvBlock1d(in_ch, mid_ch, kernel_size=1, stride=1)
        self.conv2 = ConvBlock1d(mid_ch, mid_ch, kernel_size=3, stride=stride)
        self.conv3 = ConvBlock1d(mid_ch, out_ch, kernel_size=1, stride=1, act=False)
        self.se = SEBlock1d(out_ch) if se else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.AvgPool1d(stride, stride=stride, ceil_mode=True)
                if stride > 1 else nn.Identity(),
                ConvBlock1d(in_ch, out_ch, kernel_size=1, stride=1, act=False),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        out = self.se(out)
        return self.act(out + identity)


# ----------------------------------------------------------------------------
# XResNet1D
# ----------------------------------------------------------------------------
class XResNet1D(nn.Module):
    def __init__(self, input_channels=12, base_filters=64,
                 layers=(3, 4, 23, 3), embedding_dim=512, dropout=0.3, se=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        bf = base_filters

        self.stem = nn.Sequential(
            ConvBlock1d(input_channels, bf // 2, kernel_size=7, stride=2),
            ConvBlock1d(bf // 2, bf // 2, kernel_size=3, stride=1),
            ConvBlock1d(bf // 2, bf, kernel_size=3, stride=1),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.stage1 = self._make_stage(bf, bf, bf * 4, layers[0], 1, se)
        self.stage2 = self._make_stage(bf * 4, bf * 2, bf * 8, layers[1], 2, se)
        self.stage3 = self._make_stage(bf * 8, bf * 4, bf * 16, layers[2], 2, se)
        self.stage4 = self._make_stage(bf * 16, bf * 8, bf * 32, layers[3], 2, se)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(bf * 32, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self._init_weights()

    @staticmethod
    def _make_stage(in_ch, mid_ch, out_ch, num_blocks, stride, se):
        blocks = [ResBlock1d(in_ch, mid_ch, out_ch, stride=stride, se=se)]
        for _ in range(1, num_blocks):
            blocks.append(ResBlock1d(out_ch, mid_ch, out_ch, stride=1, se=se))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.bn(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# Inception1D
# ----------------------------------------------------------------------------
class InceptionBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck_ch=32, kernels=(9, 19, 39),
                 use_se=False):
        super().__init__()
        self.bottleneck = ConvBlock1d(in_ch, bottleneck_ch, kernel_size=1, act=False)
        self.branches = nn.ModuleList([
            ConvBlock1d(bottleneck_ch, out_ch, kernel_size=k, act=False)
            for k in kernels
        ])
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            ConvBlock1d(in_ch, out_ch, kernel_size=1, act=False),
        )
        total_ch = out_ch * (len(kernels) + 1)
        self.bn = nn.BatchNorm1d(total_ch)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock1d(total_ch) if use_se else nn.Identity()
        self.out_channels = total_ch

    def forward(self, x):
        b = self.bottleneck(x)
        out = torch.cat([br(b) for br in self.branches] + [self.branch_pool(x)], dim=1)
        out = self.se(self.bn(out))
        return self.act(out)


class Inception1D(nn.Module):
    """
    InceptionTime-style 1D backbone.

    Args:
        width_mode: 'constant' reproduces InceptionTime (fixed filter count);
                    'growing' reproduces the width schedule used in the original
                    submission (base_filters * 2**min(i, 3)).
        use_se:     enable squeeze-and-excitation inside each Inception block.
        residual_every: add a residual connection every N blocks (InceptionTime
                    uses 3). Set to 1 to reproduce the original submission.
    """

    def __init__(self, input_channels=12, embedding_dim=512, num_blocks=6,
                 base_filters=32, dropout=0.3, use_se=False,
                 width_mode="growing", residual_every=1, kernels=(9, 19, 39)):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.residual_every = max(1, int(residual_every))

        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        in_ch = input_channels
        residual_in = input_channels
        for i in range(num_blocks):
            if width_mode == "constant":
                out_ch = base_filters
            else:
                out_ch = base_filters * (2 ** min(i, 3))
            blk = InceptionBlock1d(in_ch, out_ch, kernels=kernels, use_se=use_se)
            self.blocks.append(blk)
            in_ch = blk.out_channels

            self.pools.append(nn.MaxPool1d(2, stride=2) if (i + 1) % 2 == 0
                              else nn.Identity())

            if (i + 1) % self.residual_every == 0:
                self.shortcuts.append(
                    ConvBlock1d(residual_in, in_ch, kernel_size=1, act=False)
                    if residual_in != in_ch else nn.Identity()
                )
                residual_in = in_ch
            else:
                self.shortcuts.append(None)

        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        res = x
        for i, blk in enumerate(self.blocks):
            out = blk(x)
            sc = self.shortcuts[i]
            if sc is not None:
                out = self.act(out + sc(res))
                res = out
            x = self.pools[i](out)
            # `res` must be pooled unconditionally: with residual_every > 1 the
            # shortcut spans several blocks, and if res kept its original
            # temporal length while `out` was downsampled the addition above
            # would fail on shape. Pooling both keeps them aligned.
            res = self.pools[i](res)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.bn(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
_XRESNET_CFG = {
    "xresnet1d50": {"layers": (3, 4, 6, 3), "base_filters": 64},
    "xresnet1d101": {"layers": (3, 4, 23, 3), "base_filters": 64},
}


def create_backbone(name="inception1d", input_channels=12, embedding_dim=512,
                    dropout=0.3, verbose=True, **kwargs):
    if name in _XRESNET_CFG:
        cfg = _XRESNET_CFG[name]
        model = XResNet1D(
            input_channels=input_channels,
            base_filters=cfg["base_filters"],
            layers=cfg["layers"],
            embedding_dim=embedding_dim,
            dropout=dropout,
            se=kwargs.get("use_se", True),
        )
    elif name == "inception1d":
        model = Inception1D(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
            num_blocks=kwargs.get("num_blocks", 6),
            base_filters=kwargs.get("base_filters", 32),
            use_se=kwargs.get("use_se", False),
            width_mode=kwargs.get("width_mode", "growing"),
            residual_every=kwargs.get("residual_every", 1),
        )
    else:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Choose from {list(_XRESNET_CFG) + ['inception1d']}."
        )
    if verbose:
        print(f"  backbone={name}  params={model.get_num_params():,}")
    return model
