import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from scipy.spatial.distance import cdist

from src.datasets.octa500 import OCTA500Dataset
from src.models.faz_multitask import MultitaskFAZModel
from src.models.resnest_unet import ResNeStUNetConditional
from src.utils.transforms import get_val_transform


def dice_coefficient(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    inter = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0
    return (2.0 * inter + eps) / (union + eps)


def jaccard_index(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return (inter + eps) / (union + eps)


def surface_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute all distances between boundary pixels of A and B."""
    a_pts = np.column_stack(np.nonzero(A))
    b_pts = np.column_stack(np.nonzero(B))
    if len(a_pts) == 0 or len(b_pts) == 0:
        return np.array([0.0])
    dists = cdist(a_pts, b_pts)
    return dists.min(axis=1)


def hausdorff_95(pred: np.ndarray, target: np.ndarray) -> float:
    d1 = surface_distances(pred, target)
    d2 = surface_distances(target, pred)
    d = np.concatenate([d1, d2])
    return np.percentile(d, 95)


def assd(pred: np.ndarray, target: np.ndarray) -> float:
    d1 = surface_distances(pred, target)
    d2 = surface_distances(target, pred)
    return (d1.mean() + d2.mean()) / 2.0


def evaluate(cfg, model_type: str = "multitask"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = cfg["data"]["root"]
    img_size = cfg["data"]["img_size"]
    test_split = cfg.get("test_split", "test")

    ds = OCTA500Dataset(root=data_root, split=test_split, transform=get_val_transform(img_size))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"])

    if model_type == "multitask":
        model = MultitaskFAZModel(in_channels=1).to(device)
    elif model_type == "conditional":
        model = ResNeStUNetConditional(in_channels=1).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    ckpt_path = Path(cfg["logging"]["output_dir"]) / "best_model.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    dices, jaccs, hds, assds = [], [], [], []

    with torch.no_grad():
        for imgs, faz_masks, _ in loader:
            imgs = imgs.to(device)
            faz_masks = faz_masks.to(device)

            if model_type == "multitask":
                outputs = model(imgs)
                logits = outputs["faz_logits"]
            else:
                logits = model(imgs)

            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).float()

            pred_np = pred.squeeze().cpu().numpy().astype(bool)
            gt_np = faz_masks.squeeze().cpu().numpy().astype(bool)

            dices.append(dice_coefficient(pred_np, gt_np))
            jaccs.append(jaccard_index(pred_np, gt_np))
            hds.append(hausdorff_95(pred_np, gt_np))
            assds.append(assd(pred_np, gt_np))

    print(f"Dice:   {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"Jacc.:  {np.mean(jaccs):.4f} ± {np.std(jaccs):.4f}")
    print(f"HD95:   {np.mean(hds):.4f} ± {np.std(hds):.4f}")
    print(f"ASSD:   {np.mean(assds):.44f} ± {np.std(assds):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        default="multitask",
        choices=["multitask", "conditional"],
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, model_type=args.model_type)


if __name__ == "__main__":
    main()
