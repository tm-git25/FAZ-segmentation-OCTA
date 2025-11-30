from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class OCTA500Dataset(Dataset):
    """Simple OCTA-500 dataset wrapper (template).

    Assumes a directory structure like:

        root/
          images/
            OCTA_ILM_OPL/
              train/
                <id>.png
              val/
              test/
          masks_faz/
            train/
              <id>.png
            ...
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        use_ilml_opl: bool = True,
        load_vessel_mask: bool = False,
        vessel_mask_root: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.use_ilml_opl = use_ilml_opl
        self.load_vessel_mask = load_vessel_mask
        self.vessel_mask_root = Path(vessel_mask_root) if vessel_mask_root else None

        img_subdir = "OCTA_ILM_OPL" if use_ilml_opl else "OCTA_FULL"
        self.img_dir = self.root / "images" / img_subdir / split
        self.faz_dir = self.root / "masks_faz" / split

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

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        img_id = self.ids[idx]
        img_path = self.img_dir / f"{img_id}.png"
        faz_path = self.faz_dir / f"{img_id}.png"

        img = self._load_png(img_path)
        faz = self._load_png(faz_path)

        img = img.astype(np.float32) / 255.0
        faz = (faz > 127).astype(np.float32)

        sample = {"image": img, "mask": faz}

        if self.load_vessel_mask:
            if self.vessel_mask_root is None:
                raise ValueError("vessel_mask_root must be set when load_vessel_mask=True")
            vess_path = Path(self.vessel_mask_root) / self.split / f"{img_id}.png"
            vess = self._load_png(vess_path)
            vess = (vess > 127).astype(np.float32)
            sample["vessel_mask"] = vess

        if self.transform is not None:
            augmented = self.transform(image=sample["image"], mask=sample["mask"])
            img_t = augmented["image"]
            mask_t = augmented["mask"].unsqueeze(0)
            vess_t = None

            if "vessel_mask" in sample:
                aug_vess = self.transform(image=sample["image"], mask=sample["vessel_mask"])
                vess_t = aug_vess["mask"].unsqueeze(0)

            return img_t, mask_t, vess_t

        img_t = torch.from_numpy(sample["image"]).unsqueeze(0)
        mask_t = torch.from_numpy(sample["mask"]).unsqueeze(0)

        vess_t = None
        if "vessel_mask" in sample:
            vess_t = torch.from_numpy(sample["vessel_mask"]).unsqueeze(0)

        return img_t, mask_t, vess_t
