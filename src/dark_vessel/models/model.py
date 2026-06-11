"""Stage 2 + 3 — the center-heatmap detector with attribute heads.

Why a heatmap and not bounding boxes: an xView3 vessel is a handful of
bright pixels, essentially a point. Anchor-based detectors (Faster R-CNN,
YOLO) waste capacity learning box sizes that barely exist and need careful
anchor tuning for tiny objects. A CenterNet-style formulation instead
predicts a dense per-pixel "is there a vessel centred here?" score; each
vessel becomes a small Gaussian peak. Inference = find local maxima.

Architecture: U-Net. A ResNet encoder (optionally ImageNet-pretrained)
downsamples 5x; a lightweight decoder upsamples back to full resolution
with skip connections so the final layer still sees pixel-sharp detail —
essential when the object *is* 3 pixels wide.

Heads (1x1 convs on the shared full-resolution feature map):
    heatmap  — 1ch, vessel-center confidence (sigmoid at inference)
    vessel   — 1ch, vessel vs fixed infrastructure (turbines, platforms...)
    fishing  — 1ch, fishing vs non-fishing vessel
    length   — 1ch, log(vessel length in metres)

Attribute heads are supervised only AT ground-truth center pixels
(gathered with `gather_at_points`), the standard CenterNet trick.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResNetEncoder(nn.Module):
    """ResNet trunk exposing 5 feature scales (stride 1..32... well, 2..32).

    The first conv is rebuilt to accept `in_channels` inputs (SAR VH/VV +
    ancillary layers). When pretrained, the RGB weights are averaged and
    replicated across the new channels — a standard trick that keeps most
    of the pretrained low-level filters useful.
    """

    CHANNELS = {
        "resnet18": [64, 64, 128, 256, 512],
        "resnet34": [64, 64, 128, 256, 512],
        "resnet50": [64, 256, 512, 1024, 2048],
    }

    def __init__(self, name: str, in_channels: int, pretrained: bool):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        net = getattr(torchvision.models, name)(weights=weights)

        old = net.conv1
        net.conv1 = nn.Conv2d(in_channels, old.out_channels,
                              kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                mean_w = old.weight.mean(dim=1, keepdim=True)
                net.conv1.weight.copy_(mean_w.repeat(1, in_channels, 1, 1))

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)  # stride 2
        self.pool = net.maxpool                                   # stride 4
        self.layer1 = net.layer1                                  # stride 4
        self.layer2 = net.layer2                                  # stride 8
        self.layer3 = net.layer3                                  # stride 16
        self.layer4 = net.layer4                                  # stride 32
        self.out_channels = self.CHANNELS[name]

    def forward(self, x):
        f0 = self.stem(x)                 # 1/2
        f1 = self.layer1(self.pool(f0))   # 1/4
        f2 = self.layer2(f1)              # 1/8
        f3 = self.layer3(f2)              # 1/16
        f4 = self.layer4(f3)              # 1/32
        return [f0, f1, f2, f3, f4]


class UNetDecoder(nn.Module):
    """Upsample + skip-connect back to full resolution."""

    def __init__(self, enc_channels: list[int], out_ch: int):
        super().__init__()
        c0, c1, c2, c3, c4 = enc_channels
        self.up3 = _conv_block(c4 + c3, 256)
        self.up2 = _conv_block(256 + c2, 128)
        self.up1 = _conv_block(128 + c1, 96)
        self.up0 = _conv_block(96 + c0, out_ch)
        self.final = _conv_block(out_ch, out_ch)

    @staticmethod
    def _up_cat(x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, feats, full_size):
        f0, f1, f2, f3, f4 = feats
        x = self.up3(self._up_cat(f4, f3))
        x = self.up2(self._up_cat(x, f2))
        x = self.up1(self._up_cat(x, f1))
        x = self.up0(self._up_cat(x, f0))
        x = F.interpolate(x, size=full_size, mode="bilinear", align_corners=False)
        return self.final(x)


class DarkVesselNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.encoder = ResNetEncoder(m.encoder, m.in_channels, m.pretrained)
        self.decoder = UNetDecoder(self.encoder.out_channels, m.head_channels)

        def head():
            return nn.Sequential(
                nn.Conv2d(m.head_channels, m.head_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(m.head_channels, 1, 1),
            )

        self.heatmap_head = head()
        self.vessel_head = head()
        self.fishing_head = head()
        self.length_head = head()

        # Bias init so the heatmap starts near-empty (prior prob ~0.01);
        # without this the focal loss spends epochs un-learning noise.
        nn.init.constant_(self.heatmap_head[-1].bias, -4.6)

    def forward(self, x):
        feats = self.encoder(x)
        shared = self.decoder(feats, x.shape[-2:])
        return {
            "heatmap": self.heatmap_head(shared),   # logits
            "vessel": self.vessel_head(shared),     # logits
            "fishing": self.fishing_head(shared),   # logits
            "length": self.length_head(shared),     # log-metres
        }


def gather_at_points(feature_map: torch.Tensor, rows: torch.Tensor,
                     cols: torch.Tensor) -> torch.Tensor:
    """Pick the (B, 1, H, W) map's values at per-sample point lists.

    rows/cols: (B, N) integer tensors. Returns (B, N).
    """
    b, _, h, w = feature_map.shape
    flat = feature_map.view(b, h * w)
    idx = rows * w + cols
    return torch.gather(flat, 1, idx)
