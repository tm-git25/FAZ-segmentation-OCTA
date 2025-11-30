from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
from skimage.morphology import reconstruction

from .cosnet import COSNet


def compute_thresholds(image: np.ndarray, factor_marker: float, factor_mask: float):
    """
    Compute marker and mask thresholds based on non-zero intensities of the image.

    Thresholds:
        T_marker = mean + factor_marker * std
        T_mask   = mean + factor_mask   * std

    If T_marker or T_mask produces no foreground pixels, the STD factor is halved.
    """
    # Extract non-zero pixel intensities
    nz = image[image > 0]

    if len(nz) == 0:
        # Degenerate case: image all zeros → return trivial thresholds
        return 0.5, 0.25

    mean = nz.mean()
    std = nz.std()

    T_marker = mean + factor_marker * std
    T_mask   = mean + factor_mask * std

    return float(T_marker), float(T_mask)


def adaptive_morphological_binarization(
    prob: torch.Tensor,
    marker_factor: float,
    mask_factor: float,
) -> torch.Tensor:
    """
    Apply the adaptive double-thresholding described in the paper:
        - Compute thresholds using mean/std of non-zero intensities.
        - Marker = prob > T_marker
        - Mask   = prob > T_mask
        - Apply morphological reconstruction (marker grows inside mask).
        - If the threshold produces no foreground pixels, reduce STD factor by half.

    Args:
        prob (Tensor): probability tensor (N,1,H,W)
        marker_factor (float): STD factor for marker threshold
        mask_factor (float): STD factor for mask threshold

    Returns:
        Tensor (N,1,H,W): binary mask after reconstruction.
    """
    prob_np = prob.detach().cpu().numpy()  # (N,1,H,W)
    out_list = []

    for i in range(prob_np.shape[0]):
        img = prob_np[i, 0]  # (H,W)
        nz = img[img > 0]

        if len(nz) == 0:
            out_list.append(np.zeros_like(img, dtype=np.float32))
            continue

        mean = nz.mean()
        std = nz.std()

        # Compute initial thresholds
        T_marker = mean + marker_factor * std
        T_mask   = mean + mask_factor   * std

        # Compute marker/mask
        marker = img > T_marker
        mask   = img > T_mask

        # If marker has zero pixels, reduce the factor
        if marker.sum() == 0:
            T_marker = mean + (marker_factor / 2.0) * std
            marker = img > T_marker

        # If mask has zero pixels, reduce mask factor
        if mask.sum() == 0:
            T_mask = mean + (mask_factor / 2.0) * std
            mask = img > T_mask

        # Morphological reconstruction (marker grows inside mask)
        rec = reconstruction(
            seed=marker.astype(float),
            mask=mask.astype(float),
            method="dilation",
        )

        out_list.append((rec > 0.5).astype("float32"))

    out_np = np.stack(out_list, axis=0)  # (N,H,W)
    out_t = torch.from_numpy(out_np)[:, None]  # (N,1,H,W)
    return out_t.to(prob.device)


class COSNetModified(nn.Module):
    """
    COSNetModified implementing the EXACT thresholding strategy described in the paper:

    Thick branch:
        T_marker = mean + 2 * std
        T_mask   = mean + 0.5 * std

    Thin branch:
        T_marker = mean + 4 * std
        T_mask   = mean + 0.5 * std

    If thresholds produce no foreground pixels, the STD factor is halved.
    Binary maps are fused with a logical OR.

    This module returns:
        - thick_logit
        - thin_logit
        - fused_binary_mask
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_c: int = 32,
        emb_dim: int = 256,
    ):
        super().__init__()
        self.cosnet = COSNet(in_channels=in_channels, base_c=base_c, emb_dim=emb_dim)

        # Pre-defined STD factors (as described)
        self.thick_marker_factor = 2.0
        self.thick_mask_factor   = 0.5

        self.thin_marker_factor  = 4.0
        self.thin_mask_factor    = 0.5

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        thick_logit, thin_logit, thick_emb, thin_emb = self.cosnet(x)

        thick_prob = torch.sigmoid(thick_logit)
        thin_prob  = torch.sigmoid(thin_logit)

        # Adaptive double-thresholding per branch
        thick_bin = adaptive_morphological_binarization(
            thick_prob,
            marker_factor=self.thick_marker_factor,
            mask_factor=self.thick_mask_factor,
        )

        thin_bin = adaptive_morphological_binarization(
            thin_prob,
            marker_factor=self.thin_marker_factor,
            mask_factor=self.thin_mask_factor,
        )

        # OR-based fusion
        fused_bin = torch.clamp(thick_bin + thin_bin, 0, 1)

        return thick_logit, thin_logit, fused_bin
