from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ConditionalFAZLossConfig:
    w1: float = 2.0  # weight for FAZ pixels misclassified as background
    k1: float = 1.0  # weight for background pixels outside vessels
    k2: float = 2.0  # weight for background pixels inside vessels


class ConditionalFAZLoss(nn.Module):
    """Vessel-aware loss function for FAZ segmentation.

    Implements:
        L = -E[ w1 * y_true * log10(y_pred) +
                w0 * (1 - y_true) * log10(1 - y_pred) ]

    where w0 depends on vessel membership.
    """

    def __init__(self, cfg: ConditionalFAZLossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        logits: torch.Tensor,
        target_faz: torch.Tensor,
        vessel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conditional FAZ loss.

        Args:
            logits: raw model predictions, shape (N, 1, H, W)
            target_faz: FAZ ground truth mask, shape (N, 1, H, W) or (N, H, W)
            vessel_mask: vessel mask (0/1), shape (N, 1, H, W) or (N, H, W)
        """
        if logits.shape[1] != 1:
            raise ValueError("This loss expects a single-channel (binary) prediction.")

        y_pred = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        if target_faz.ndim == 4:
            y_true = target_faz.squeeze(1)
        else:
            y_true = target_faz
        if vessel_mask.ndim == 4:
            y_vess = vessel_mask.squeeze(1)
        else:
            y_vess = vessel_mask

        y_true = y_true.float()
        y_vess = y_vess.float()

        # compute w0: k1 outside vessels, k2 inside vessels
        w0 = torch.where(
            y_vess > 0.5,
            torch.full_like(y_vess, self.cfg.k2),
            torch.full_like(y_vess, self.cfg.k1),
        )

        w1 = self.cfg.w1

        term1 = w1 * y_true * torch.log10(y_pred)
        term2 = w0 * (1.0 - y_true) * torch.log10(1.0 - y_pred)

        loss = -(term1 + term2).mean()
        return loss
