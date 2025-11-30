from typing import Tuple

import torch
import torch.nn as nn

from .unet_blocks import DoubleConv, Down, Up, OutConv


class ProjectionHead(nn.Module):
    """Simple projection head to obtain pixel-wise embeddings."""

    def __init__(self, in_channels: int, emb_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class COSNet(nn.Module):
    """Simplified COSNet-style network with contrastive projection heads.

    Coarse stage only (as used in the paper's description).
    """

    def __init__(self, in_channels: int = 1, base_c: int = 32, emb_dim: int = 256):
        super().__init__()
        # shared encoder
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)

        # thick branch decoder
        self.up_thick = Up(base_c * 4 + base_c * 2, base_c * 2)
        self.out_thick = OutConv(base_c * 2, 1)
        self.proj_thick = ProjectionHead(base_c * 2, emb_dim=emb_dim)

        # thin branch decoder
        self.up_thin = Up(base_c * 4 + base_c * 2, base_c * 2)
        self.out_thin = OutConv(base_c * 2, 1)
        self.proj_thin = ProjectionHead(base_c * 2, emb_dim=emb_dim)

    def encode(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        return x1, x2, x3

    def forward(
        self, x
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits and embeddings for thick and thin branches."""
        x1, x2, x3 = self.encode(x)

        # thick
        t = self.up_thick(x3, x2)
        thick_logit = self.out_thick(t)
        thick_emb = self.proj_thick(t)

        # thin
        s = self.up_thin(x3, x2)
        thin_logit = self.out_thin(s)
        thin_emb = self.proj_thin(s)

        return thick_logit, thin_logit, thick_emb, thin_emb
