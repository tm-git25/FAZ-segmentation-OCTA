import torch
import torch.nn as nn

from .unet_blocks import DoubleConv, Down, Up, OutConv


class SimpleResNeStEncoder(nn.Module):
    """Very lightweight ResNeSt-like encoder stub.
    """

    def __init__(self, in_channels: int = 1, base_c: int = 64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x1, x2, x3, x4


class ResNeStUNet(nn.Module):
    """ResNeSt-like U-Net for FAZ segmentation."""

    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_c: int = 64):
        super().__init__()
        self.encoder = SimpleResNeStEncoder(in_channels, base_c)
        self.up1 = Up(base_c * 8 + base_c * 4, base_c * 4)
        self.up2 = Up(base_c * 4 + base_c * 2, base_c * 2)
        self.up3 = Up(base_c * 2 + base_c, base_c)
        self.outc = OutConv(base_c, num_classes)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


class ResNeStUNetConditional(nn.Module):
    """Same architecture as ResNeStUNet, used with ConditionalFAZLoss."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = ResNeStUNet(in_channels=in_channels, num_classes=1)

    def forward(self, x):
        return self.net(x)
