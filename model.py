import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleLayer(nn.Module):
    def __init__(self, in_channel, out_channel, use_bias=False):
        super().__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=(2, 2))
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding='same', bias=use_bias),
            nn.BatchNorm2d(out_channel, eps=0.001),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channel, out_channel, 3, padding='same', bias=use_bias),
            nn.BatchNorm2d(out_channel, eps=0.001),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channel, out_channel, 3, padding='same', bias=use_bias),
            nn.BatchNorm2d(out_channel, eps=0.001),
            nn.LeakyReLU(0.1),
        )

    def forward(self, down: torch.Tensor, up: torch.Tensor):
        # shape: [batch, channel, height, width]
        # down: from downsample layer
        # up: from upsample layer or center
        up = self.upsample(up)
        x = torch.cat([down, up], dim=1)
        x = self.layer(x)
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, in_channel, out_channel=None, use_bias=False, pooling=True):
        super().__init__()
        if out_channel is None:
            out_channel = in_channel * 2
        if pooling:
            self.pooling = nn.MaxPool2d((2, 2), stride=(2, 2))
        else:
            self.pooling = None
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding='same', bias=use_bias),
            nn.BatchNorm2d(out_channel, eps=0.001),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channel, out_channel, 3, padding='same', bias=use_bias),
            nn.BatchNorm2d(out_channel, eps=0.001),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        # shape: [batch, channel, height, width]
        x = self.layer(x)
        if self.pooling:
            xp = self.pooling(x)
        else:
            xp = None
        return x, xp


class LineDetector(nn.Module):

    def __init__(self, num_classes, use_bias=False):
        super().__init__()
        self.downsample256 = DownsampleLayer(3, 16, use_bias=use_bias)
        self.downsample128 = DownsampleLayer(16, use_bias=use_bias)
        self.downsample64 = DownsampleLayer(32, use_bias=use_bias)
        self.downsample32 = DownsampleLayer(64, use_bias=use_bias)
        self.downsample16 = DownsampleLayer(128, use_bias=use_bias)
        self.downsample8 = DownsampleLayer(256, use_bias=use_bias)
        self.downsample_center = DownsampleLayer(512, use_bias=use_bias, pooling=False)
        self.upsample16 = UpsampleLayer(1024 + 1024 // 2, 512, use_bias=use_bias)
        self.upsample32 = UpsampleLayer(512 + 512 // 2, 256, use_bias=use_bias)
        self.upsample64 = UpsampleLayer(256 + 256 // 2, 128, use_bias=use_bias)
        self.upsample128 = UpsampleLayer(128 + 128 // 2, 64, use_bias=use_bias)
        self.upsample256 = UpsampleLayer(64 + 64 // 2, 32, use_bias=use_bias)
        self.upsample512 = UpsampleLayer(32 + 32 // 2, 16, use_bias=use_bias)
        self.classifier = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        # x: [b, 3, 512, 512]
        d0a, d0a_pool = self.downsample256(x)  # [b, 16, 256, 256]
        d0, d0_pool = self.downsample128(d0a_pool)  # [b, 32, 128, 128]
        d1, d1_pool = self.downsample64(d0_pool)  # [b, 64, 64, 64]
        d2, d2_pool = self.downsample32(d1_pool)  # [b, 128, 32, 32]
        d3, d3_pool = self.downsample16(d2_pool)  # [b, 256, 16, 16]
        d4, d4_pool = self.downsample8(d3_pool)  # [b, 512, 8, 8]
        center, _ = self.downsample_center(d4_pool)  # [b, 512, 8, 8]
        u4 = self.upsample16(d4, center)  # [b, 512, 16, 16]
        u3 = self.upsample32(d3, u4)  # [b, 256, 32, 32]
        u2 = self.upsample64(d2, u3)  # [b, 128, 64, 64]
        u1 = self.upsample128(d1, u2)  # [b, 64, 128, 128]
        u0 = self.upsample256(d0, u1)  # [b, 32, 256, 256]
        u0a = self.upsample512(d0a, u0)  # [b, 16, 512, 512]
        out = self.classifier(u0a)  # [b, num_classes, 512, 512]
        return F.sigmoid(out)
