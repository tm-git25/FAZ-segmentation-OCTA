from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ROSEDataset(Dataset):
    """Template dataset class for ROSE vessel segmentation datasets.

    Assumes a directory structure like:

        root/
          images/
            train/
              <id>.png
            val/
            test/
          masks_vessels/
            train/
              <id>.png
            ...
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.img_dir = self.root / "images" / split
        self.mask_dir = self.root / "masks_vessels" / split
        self.ids = sorted([p.stem for p in self.img_dir.glob("*.png")])

        if not self.ids:
            raise RuntimeError(f"No images found in {self.img_dir}. Please adapt paths.")

    def __len__(self) -> int:
        return len(self.ids)

    def _load_png(self, path: Path) -> np.ndarray:
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise RuntimeError(f"Could not read image: {path}")
        return arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id = self.ids[idx]
        img_path = self.img_dir / f"{img_id}.png"
        mask_path = self.mask_dir / f"{img_id}.png"

        img = self._load_png(img_path)
        mask = self._load_png(mask_path)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        sample = {"image": img, "mask": mask}

        if self.transform is not None:
            augmented = self.transform(image=sample["image"], mask=sample["mask"])
            img_t = augmented["image"]
            mask_t = augmented["mask"].unsqueeze(0)
            return img_t, mask_t

        img_t = torch.from_numpy(sample["image"]).unsqueeze(0)
        mask_t = torch.from_numpy(sample["mask"]).unsqueeze(0)
        return img_t, mask_t
