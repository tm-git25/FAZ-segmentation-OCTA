import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.datasets.octa500 import OCTA500Dataset
from src.losses.dice import DiceLoss
from src.models.faz_multitask import MultitaskFAZModel
from src.utils.seed import set_seed
from src.utils.transforms import get_train_transform, get_val_transform


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["training"]["seed"])

    output_dir = Path(cfg["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = cfg["data"]["root"]
    img_size = cfg["data"]["img_size"]

    train_ds = OCTA500Dataset(root=data_root, split="train", transform=get_train_transform(img_size))
    val_ds = OCTA500Dataset(root=data_root, split="val", transform=get_val_transform(img_size))

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

    model = MultitaskFAZModel(in_channels=1).to(device)

    dice_loss = DiceLoss()

    lamb_faz = cfg["loss"]["lambda_faz"]
    lamb_vess = cfg["loss"]["lambda_vessels"]
    lamb_bg = cfg["loss"]["lambda_background"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best_val_loss = float("inf")

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for imgs, faz_masks, _ in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}"
        ):
            imgs = imgs.to(device)
            faz_masks = faz_masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            faz_logits = outputs["faz_logits"]
            vessel_logits = outputs["vessel_thick_logits"]  # placeholder: use thick as vessel logits

            # compute multi-class style dice: background + vessels + FAZ
            # Here, we approximate by combining FAZ mask and vessel_fused_binary
            vessel_bin = outputs["vessel_fused_binary"].detach()
            background = (1.0 - torch.clamp(faz_masks + vessel_bin, 0, 1))

            # Stack channels: [bg, vessels, faz]
            target_multi = torch.cat([background, vessel_bin, faz_masks], dim=1)
            # For predicted, we approximate with logistic outputs for FAZ and vessels
            faz_prob = torch.sigmoid(faz_logits)
            vess_prob = torch.sigmoid(vessel_logits)
            bg_prob = torch.clamp(1.0 - (faz_prob + vess_prob), 0, 1)
            pred_multi = torch.cat([bg_prob, vess_prob, faz_prob], dim=1)

            loss_faz = dice_loss(faz_logits, faz_masks)
            loss_vess = dice_loss(vessel_logits, vessel_bin)
            # background component computed from multi-channel dice
            loss_bg = dice_loss(pred_multi, target_multi.argmax(dim=1))

            loss = lamb_faz * loss_faz + lamb_vess * loss_vess + lamb_bg * loss_bg
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, faz_masks, _ in val_loader:
                imgs = imgs.to(device)
                faz_masks = faz_masks.to(device)

                outputs = model(imgs)
                faz_logits = outputs["faz_logits"]
                vessel_logits = outputs["vessel_thick_logits"]
                vessel_bin = outputs["vessel_fused_binary"].detach()
                background = (1.0 - torch.clamp(faz_masks + vessel_bin, 0, 1))

                target_multi = torch.cat([background, vessel_bin, faz_masks], dim=1)
                faz_prob = torch.sigmoid(faz_logits)
                vess_prob = torch.sigmoid(vessel_logits)
                bg_prob = torch.clamp(1.0 - (faz_prob + vess_prob), 0, 1)
                pred_multi = torch.cat([bg_prob, vess_prob, faz_prob], dim=1)

                loss_faz = dice_loss(faz_logits, faz_masks)
                loss_vess = dice_loss(vessel_logits, vessel_bin)
                loss_bg = dice_loss(pred_multi, target_multi.argmax(dim=1))

                loss = lamb_faz * loss_faz + lamb_vess * loss_vess + lamb_bg * loss_bg
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
