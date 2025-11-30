from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary or multi-class segmentation."""

    def __init__(self, smooth: float = 1.0, ignore_index: Optional[int] = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: (N, C, H, W)
            targets: (N, H, W) or (N, 1, H, W) for binary.
        """
        if logits.ndim != 4:
            raise ValueError("logits must have shape (N, C, H, W)")

        if targets.ndim == 4:
            targets = targets.squeeze(1)

        num_classes = logits.shape[1]

        if num_classes == 1:
            probs = torch.sigmoid(logits)
            targets_one_hot = targets.float()
        else:
            probs = torch.softmax(logits, dim=1)
            targets_one_hot = torch.nn.functional.one_hot(
                targets.long(), num_classes=num_classes
            ).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.ignore_index is not None and 0 <= self.ignore_index < num_classes:
            mask = torch.ones_like(dice, dtype=torch.bool)
            mask[self.ignore_index] = False
            dice = dice[mask]

        loss = 1.0 - dice.mean()
        return loss
