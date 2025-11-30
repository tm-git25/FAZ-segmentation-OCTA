import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.datasets.rose import ROSEDataset
from src.models.octanet import OCTANet
from src.models.cosnet import COSNet
from src.models.cosnet_modified import COSNetModified
from src.utils.transforms import get_val_transform


def dice_coefficient(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    inter = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0
    return (2.0 * inter + eps) / (union + eps)


def sensitivity(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(~pred, target).sum()
    denom = tp + fn
    if denom == 0:
        return 1.0
    return (tp + eps) / (denom + eps)


def specificity(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    tn = np.logical_and(~pred, ~target).sum()
    fp = np.logical_and(pred, ~target).sum()
    denom = tn + fp
    if denom == 0:
        return 1.0
    return (tn + eps) / (denom + eps)


def accuracy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    correct = (pred == target).sum()
    total = pred.size
    return (correct + eps) / (total + eps)


def false_discovery_rate(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    denom = tp + fp
    if denom == 0:
        return 0.0
    return (fp + eps) / (denom + eps)


def get_model(name: str):
    if name == "octanet":
        return OCTANet(in_channels=1)
    elif name == "cosnet":
        return COSNet(in_channels=1)
    elif name == "cosnet_modified":
        return COSNetModified(in_channels=1)
    else:
        raise ValueError(f"Unknown model: {name}")


def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = cfg["data"]["root"]
    img_size = cfg["data"]["img_size"]
    dataset_name = cfg.get("dataset", "rose")

    if dataset_name.lower() != "rose":
        raise NotImplementedError("metrics_vessels.py is configured for ROSE by default.")

    ds = ROSEDataset(root=data_root, split="test", transform=get_val_transform(img_size))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"])

    model = get_model(cfg["model"]).to(device)
    ckpt_path = Path(cfg["logging"]["output_dir"]) / "best_model.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    dices, sens_list, spec_list, acc_list, fdr_list = [], [], [], [], []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            out = model(imgs)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out

            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).float()

            pred_np = pred.squeeze().cpu().numpy().astype(bool)
            gt_np = masks.squeeze().cpu().numpy().astype(bool)

            dices.append(dice_coefficient(pred_np, gt_np))
            sens_list.append(sensitivity(pred_np, gt_np))
            spec_list.append(specificity(pred_np, gt_np))
            acc_list.append(accuracy(pred_np, gt_np))
            fdr_list.append(false_discovery_rate(pred_np, gt_np))

    print(f"Dice:   {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"Sens.:  {np.mean(sens_list):.4f} ± {np.std(sens_list):.4f}")
    print(f"Spec.:  {np.mean(spec_list):.4f} ± {np.std(spec_list):.4f}")
    print(f"Acc.:   {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"FDR:    {np.mean(fdr_list):.44f} ± {np.std(fdr_list):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg)


if __name__ == "__main__":
    main()
