from typing import Optional, List

import matplotlib.pyplot as plt
import torch


def show_image_and_masks(image: torch.Tensor,
                         mask_list: Optional[List[torch.Tensor]] = None,
                         titles: Optional[List[str]] = None) -> None:
    """Quick visualization helper for OCTA image and one or more masks.

    Args:
        image: tensor (1, H, W) or (3, H, W).
        mask_list: list of tensors (H, W) or (1, H, W).
        titles: list of titles for subplots.
    """
    if mask_list is None:
        mask_list = []
    if titles is None:
        titles = []

    n_cols = 1 + len(mask_list)
    plt.figure(figsize=(4 * n_cols, 4))

    img = image.detach().cpu()
    if img.dim() == 3 and img.shape[0] == 1:
        img = img.squeeze(0)
        cmap = "gray"
    else:
        img = img.permute(1, 2, 0)
        cmap = None

    plt.subplot(1, n_cols, 1)
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.title(titles[0] if titles else "Image")

    for i, m in enumerate(mask_list):
        m = m.detach().cpu()
        if m.dim() == 3:
            m = m.squeeze(0)
        plt.subplot(1, n_cols, i + 2)
        plt.imshow(m, cmap="gray")
        plt.axis("off")
        if len(titles) > i + 1:
            plt.title(titles[i + 1])

    plt.tight_layout()
    plt.show()
