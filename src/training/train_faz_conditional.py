import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.datasets.octa500 import OCTA500Dataset
from src.losses.conditional_faz_loss import ConditionalFAZLoss, ConditionalFAZLossConfig
from src.models.resnest_unet import ResNeStUNetConditional
from src.utils.seed import set_seed
from src.utils.transforms import get_train_transform, get_val_transform


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["training"]["seed"])

    output_dir = Path(cfg["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = cfg["data"]["root"]
    img_size = cfg["data"]["img_size"]
    vessel_mask_root = cfg["data"]["vessel_mask_root"]

    train_ds = OCTA500Dataset(
        root=data_root,
        split="train",
        transform=get_train_transform(img_size),
        load_vessel_mask=True,
        vessel_mask_root=vessel_mask_root,
    )
    val_ds = OCTA500Dataset(
        root=data_root,
        split="val",
        transform=get_val_transform(img_size),
        load_vessel_mask=True,
        vessel_mask_root=vessel_mask_root,
    )

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

    model = ResNeStUNetConditional(in_channels=1).to(device)

    loss_cfg = ConditionalFAZLossConfig(
        w1=cfg["loss"]["w1"],
        k1=cfg["loss"]["k1"],
        k2=cfg["loss"]["k2"],
    )
    criterion = ConditionalFAZLoss(loss_cfg)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best_val_loss = float("inf")

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for imgs, faz_masks, vess_masks in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}"
        ):
            imgs = imgs.to(device)
            faz_masks = faz_masks.to(device)
            vess_masks = vess_masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, faz_masks, vess_masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, faz_masks, vess_masks in val_loader:
                imgs = imgs.to(device)
                faz_masks = faz_masks.to(device)
                vess_masks = vess_masks.to(device)

                logits = model(imgs)
                loss = criterion(logits, faz_masks, vess_masks)
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
