import os
import time
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as ssim_metric

from mdas_dataset import MDASSRDataset
from mdas import MainNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 60
BATCH_SIZE = 1
LR = 2e-4
PATCH = 96
SCALE = 3

trainLR  = "./Augsburg_data_4_publication/sr_deep_model_data/EeteS_EnMAP_30m_deep_train.tif"
trainMSI = "./Augsburg_data_4_publication/sr_deep_model_data/EeteS_Sentinel_2_10m_deep_train.tif"
trainHR  = "./Augsburg_data_4_publication/sr_deep_model_data/EeteS_EnMAP_10m_deep_train.tif"

valLR  = "./Augsburg_data_4_publication/sr_deep_model_data/EeteS_EnMAP_30m_deep_valid.tif"
valMSI = "./Augsburg_data_4_publication/sr_deep_model_data/EeteS_Sentinel_2_10m_deep_valid.tif"
valHR  = "./Augsburg_data_4_publication/sr_deep_model_data/EeteS_EnMAP_10m_deep_valid.tif"

def compute_sam(img1, img2):
    # img1, img2: (H, W, C)
    eps = 1e-8
    dot = np.sum(img1 * img2, axis=-1)
    norm1 = np.linalg.norm(img1, axis=-1)
    norm2 = np.linalg.norm(img2, axis=-1)
    cos = dot / (norm1 * norm2 + eps)
    cos = np.clip(cos, -1, 1)
    return np.degrees(np.mean(np.arccos(cos)))


def compute_ergas(gt, pred, scale=3):
    eps = 1e-8
    mean_gt = np.mean(gt, axis=(0, 1))
    rmse = np.sqrt(np.mean((gt - pred) ** 2, axis=(0, 1)))
    return 100 / scale * np.sqrt(np.mean((rmse / (mean_gt + eps)) ** 2))


def spectral_angle_loss_torch(pred, gt, eps=1e-8):
    # pred, gt: (B, C, H, W)
    B, C, H, W = pred.shape
    pred_flat = pred.view(B, C, -1)
    gt_flat = gt.view(B, C, -1)

    dot = (pred_flat * gt_flat).sum(dim=1)
    pred_norm = torch.norm(pred_flat, dim=1)
    gt_norm = torch.norm(gt_flat, dim=1)

    cos = dot / (pred_norm * gt_norm + eps)
    cos = torch.clamp(cos, -1 + eps, 1 - eps)
    ang = torch.acos(cos)
    return ang.mean()


train_set = MDASSRDataset(
    lr_hsi_path=trainLR,
    hr_msi_path=trainMSI,
    hr_hsi_path=trainHR,
    crop_size=PATCH,
    num_samples=8000,
    scale_factor=SCALE,
)
val_set = MDASSRDataset(
    lr_hsi_path=valLR,
    hr_msi_path=valMSI,
    hr_hsi_path=valHR,
    crop_size=PATCH,
    num_samples=2000,
    scale_factor=SCALE,
)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

model = MainNet(C_hsi=242, C_msi=4, embed_dim=48).to(DEVICE)
l1_loss = nn.L1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scaler = GradScaler("cuda" if DEVICE == "cuda" else "cpu")

best_val = float("inf")


@torch.no_grad()
def validate():
    model.eval()
    losses = []
    for hr, lr_up, msi in tqdm(val_loader, desc="Valid", leave=False):
        hr = hr.to(DEVICE)
        lr_up = lr_up.to(DEVICE)
        msi = msi.to(DEVICE)

        with autocast(device_type="cuda" if DEVICE == "cuda" else "cpu"):
            sr = model(lr_up, msi)
            l1 = l1_loss(sr, hr)
            sam = spectral_angle_loss_torch(sr, hr)
            loss = l1 + 0.05 * sam

        if torch.isfinite(loss):
            losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("inf")


def main():
    global best_val

    # sanity check
    batch = next(iter(train_loader))
    print("Sanity shapes:")
    print("GT      :", batch[0].shape)
    print("LRHSI_up:", batch[1].shape)
    print("HRMSI   :", batch[2].shape)

    print("\n🔥 Training Started (MDAS)\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for hr, lr_up, msi in pbar:
            hr = hr.to(DEVICE)
            lr_up = lr_up.to(DEVICE)
            msi = msi.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda" if DEVICE == "cuda" else "cpu"):
                sr = model(lr_up, msi)
                l1 = l1_loss(sr, hr)
                sam = spectral_angle_loss_torch(sr, hr)
                loss = l1 + 0.05 * sam

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        val_loss = validate()

        print(
            f"Epoch {epoch}: Train {train_loss:.6f} | "
            f"Val {val_loss:.6f}"
        )

        if val_loss < best_val and np.isfinite(val_loss):
            best_val = val_loss
            torch.save(model.state_dict(), "best_mdas.pth")
            print("💾 Saved best_mdas.pth\n")

    print("✅ Training finished. Best val loss:", best_val)


if __name__ == "__main__":
    main()
