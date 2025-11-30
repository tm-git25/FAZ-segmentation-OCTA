from typing import Dict

import torch
import torch.nn as nn

from .cosnet_modified import COSNetModified
from .unet_blocks import DoubleConv, Down, Up, OutConv


class SharedEncoder(nn.Module):
    """Shared encoder used for multitask learning (FAZ + vessels)."""

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


class FAZDecoder(nn.Module):
    """Decoder head for FAZ segmentation."""

    def __init__(self, base_c: int = 64):
        super().__init__()
        self.up1 = Up(base_c * 8 + base_c * 4, base_c * 4)
        self.up2 = Up(base_c * 4 + base_c * 2, base_c * 2)
        self.up3 = Up(base_c * 2 + base_c, base_c)
        self.outc = OutConv(base_c, 1)

    def forward(self, x1, x2, x3, x4):
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


class MultitaskFAZModel(nn.Module):
    """Multitask learning framework: joint FAZ + vessel segmentation.

    - Shared encoder.
    - FAZ decoder head.
    - Vessel head implemented by the coarse stage of a modified COSNet.
    """

    def __init__(self, in_channels: int = 1, base_c: int = 64):
        super().__init__()
        self.encoder = SharedEncoder(in_channels=in_channels, base_c=base_c)
        self.faz_decoder = FAZDecoder(base_c=base_c)

        # Vessel branch: a small COSNetModified instance operating directly on the image.
        # In a full implementation, you might want to feed encoder features instead.
        self.vessel_branch = COSNetModified(in_channels=in_channels, base_c=32)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x1, x2, x3, x4 = self.encoder(x)
        faz_logits = self.faz_decoder(x1, x2, x3, x4)

        thick_logit, thin_logit, fused_bin = self.vessel_branch(x)

        return {
            "faz_logits": faz_logits,
            "vessel_thick_logits": thick_logit,
            "vessel_thin_logits": thin_logit,
            "vessel_fused_binary": fused_bin,
        }
