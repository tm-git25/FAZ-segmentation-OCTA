from typing import Tuple, Dict, Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(img_size: int = 512) -> A.BasicTransform:
    """Return training augmentation for OCTA images and masks."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )


def get_val_transform(img_size: int = 512) -> A.BasicTransform:
    """Return validation / test preprocessing for OCTA images and masks."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )
