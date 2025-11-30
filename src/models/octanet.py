from typing import Tuple, Dict

import torch
import torch.nn as nn

from .unet_blocks import DoubleConv, Down, Up, OutConv


class OCTANetCoarse(nn.Module):
    """
    Coarse stage of OCTA-Net with thick and thin vessel branches.

    - Shared encoder.
    - Two decoder branches:
        * thick branch: focuses on large / thick vessels
        * thin branch: focuses on small / capillary vessels

    This module outputs:
        - thick_logits: coarse prediction for thick vessels
        - thin_logits:  coarse prediction for thin vessels
    """

    def __init__(self, in_channels: int = 1, base_c: int = 32):
        super().__init__()

        # Shared encoder (U-Net style)
        self.inc = DoubleConv(in_channels, base_c)        # e.g. 1 -> 32
        self.down1 = Down(base_c, base_c * 2)            # 32 -> 64
        self.down2 = Down(base_c * 2, base_c * 4)        # 64 -> 128
        self.down3 = Down(base_c * 4, base_c * 8)        # 128 -> 256

        # Thick branch decoder
        self.up1_thick = Up(base_c * 8 + base_c * 4, base_c * 4)
        self.up2_thick = Up(base_c * 4 + base_c * 2, base_c * 2)
        self.up3_thick = Up(base_c * 2 + base_c, base_c)
        self.out_thick = OutConv(base_c, 1)

        # Thin branch decoder
        self.up1_thin = Up(base_c * 8 + base_c * 4, base_c * 4)
        self.up2_thin = Up(base_c * 4 + base_c * 2, base_c * 2)
        self.up3_thin = Up(base_c * 2 + base_c, base_c)
        self.out_thin = OutConv(base_c, 1)

    def encode(self, x: torch.Tensor):
        x1 = self.inc(x)     # (N, base_c, H,   W)
        x2 = self.down1(x1)  # (N, 2*base_c, H/2, W/2)
        x3 = self.down2(x2)  # (N, 4*base_c, H/4, W/4)
        x4 = self.down3(x3)  # (N, 8*base_c, H/8, W/8)
        return x1, x2, x3, x4

    def decode_thick(self, x1, x2, x3, x4):
        t = self.up1_thick(x4, x3)
        t = self.up2_thick(t, x2)
        t = self.up3_thick(t, x1)
        thick_logits = self.out_thick(t)
        return thick_logits

    def decode_thin(self, x1, x2, x3, x4):
        s = self.up1_thin(x4, x3)
        s = self.up2_thin(s, x2)
        s = self.up3_thin(s, x1)
        thin_logits = self.out_thin(s)
        return thin_logits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2, x3, x4 = self.encode(x)
        thick_logits = self.decode_thick(x1, x2, x3, x4)
        thin_logits = self.decode_thin(x1, x2, x3, x4)
        return thick_logits, thin_logits


class OCTANetFine(nn.Module):
    """
    Fine stage of OCTA-Net.

    Takes as input:
        - original image
        - coarse thick prediction
        - coarse thin prediction

    and refines the vessel map via another U-Net style network.
    """

    def __init__(self, in_channels: int = 3, base_c: int = 32):
        """
        Args:
            in_channels: defaults to 3 = [image, coarse_thick, coarse_thin]
        """
        super().__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)

        # Decoder
        self.up1 = Up(base_c * 8 + base_c * 4, base_c * 4)
        self.up2 = Up(base_c * 4 + base_c * 2, base_c * 2)
        self.up3 = Up(base_c * 2 + base_c, base_c)
        self.outc = OutConv(base_c, 1)

    def forward(self, x_concat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_concat: (N, 3, H, W) = [image, coarse_thick, coarse_thin]

        Returns:
            fine_logits: refined vessel logits, (N, 1, H, W)
        """
        x1 = self.inc(x_concat)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


class OCTANet(nn.Module):
    """
    Full OCTA-Net style model: coarse + fine stages.

    Coarse stage:
        - Shared encoder
        - Two decoders: thick and thin branches

    Fine stage:
        - U-Net that refines using:
            [input_image, coarse_thick_prob, coarse_thin_prob]

    Forward outputs:
        - coarse_thick_logits
        - coarse_thin_logits
        - fine_logits
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_c: int = 32,
    ):
        super().__init__()
        self.coarse = OCTANetCoarse(in_channels=in_channels, base_c=base_c)

        # Fine stage expects concatenation of:
        #   image (1 ch) + thick_prob (1 ch) + thin_prob (1 ch) = 3 ch
        self.fine = OCTANetFine(in_channels=3, base_c=base_c)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: input OCTA image, shape (N, 1, H, W)

        Returns:
            dict with:
                'coarse_thick_logits': (N, 1, H, W)
                'coarse_thin_logits':  (N, 1, H, W)
                'fine_logits':         (N, 1, H, W)
        """
        # Coarse stage
        thick_logits, thin_logits = self.coarse(x)

        # Convert coarse logits to probabilities for fine stage input
        thick_prob = torch.sigmoid(thick_logits)
        thin_prob = torch.sigmoid(thin_logits)

        # Concatenate original image + coarse maps
        x_concat = torch.cat([x, thick_prob, thin_prob], dim=1)

        # Fine stage refinement
        fine_logits = self.fine(x_concat)

        return {
            "coarse_thick_logits": thick_logits,
            "coarse_thin_logits": thin_logits,
            "fine_logits": fine_logits,
        }
