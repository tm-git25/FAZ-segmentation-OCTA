import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.datasets.rose import ROSEDataset
from src.datasets.octa500 import OCTA500Dataset
from src.losses.dice import DiceLoss
from src.models.octanet import OCTANet
from src.models.cosnet import COSNet
from src.models.cosnet_modified import COSNetModified
from src.utils.seed import set_seed
from src.utils.transforms import get_train_transform, get_val_transform


def get_dataset(name: str, root: str, split: str, img_size: int):
    if name.lower() == "rose":
        ds = ROSEDataset(root=root, split=split, transform=get_train_transform(img_size))
    elif name.lower() == "octa500":
        ds = OCTA500Dataset(root=root, split=split, transform=get_train_transform(img_size))
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return ds


def get_model(name: str):
    if name == "octanet":
        return OCTANet(in_channels=1)
    elif name == "cosnet":
        return COSNet(in_channels=1)
    elif name == "cosnet_modified":
        return COSNetModified(in_channels=1)
    else:
        raise ValueError(f"Unknown model: {name}")


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["training"]["seed"])

    output_dir = Path(cfg["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg.get("dataset", "octa500")
    data_root = cfg["data"]["root"]
    img_size = cfg["data"]["img_size"]

    train_ds = get_dataset(dataset_name, data_root, "train", img_size)
    val_ds = get_dataset(dataset_name, data_root, "val", img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    model = get_model(cfg["model"]).to(device)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best_val_loss = float("inf")

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}"):
            if isinstance(batch, tuple):
                imgs, masks = batch[:2]
            else:
                imgs, masks = batch["image"], batch["mask"]

            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            out = model(imgs)
            if isinstance(out, tuple):
                # octanet / cosnet style
                thick_logit = out[0]
                loss = criterion(thick_logit, masks)
            else:
                loss = criterion(out, masks)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # simple val loop on Dice loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, tuple):
                    imgs, masks = batch[:2]
                else:
                    imgs, masks = batch["image"], batch["mask"]

                imgs = imgs.to(device)
                masks = masks.to(device)

                out = model(imgs)
                if isinstance(out, tuple):
                    thick_logit = out[0]
                    loss = criterion(thick_logit, masks)
                else:
                    loss = criterion(out, masks)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best model saved to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
