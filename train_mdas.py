import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_metric
import scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from data import MDASSRDataset
from sstb_mamba_2 import MainNet


# ================== Reproducibility =================== #

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

cudnn.benchmark = True
cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


# ================== Hyperparameters =================== #

lr = 1e-4
epochs = 100
batch_size = 3
ckpt_step = 10
grad_clip = 0.5

def load_checkpoint(model, optimizer=None, path=None):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["net"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    return model
# ================== Model =================== #

model = MainNet().to(device)

criterion = nn.L1Loss().to(device)

optimizer = optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=1e-4
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.5
)

scaler = GradScaler()

model_folder = "Trained_model/"
writer = SummaryWriter("train_logs/" + model_folder)


# ================= Utility Functions ================= #

def has_nan(x):
    return torch.isnan(x).any() or torch.isinf(x).any()


def save_checkpoint(model, optimizer, epoch, reason="normal"):

    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)

    path = os.path.join(model_folder, f"epoch_{epoch}_{reason}.pth")

    checkpoint = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "lr": lr
    }

    torch.save(checkpoint, path)

    print(f"\nCheckpoint saved ({reason}): {path}\n")


# ================= Metrics ================= #

def compute_sam(img1, img2):

    eps = 1e-8

    dot = np.sum(img1 * img2, axis=-1)

    norm1 = np.linalg.norm(img1, axis=-1)
    norm2 = np.linalg.norm(img2, axis=-1)

    cos = dot / (norm1 * norm2 + eps)

    cos = np.clip(cos, -1, 1)

    sam = np.arccos(cos)

    return np.degrees(np.mean(sam))


def compute_ergas(gt, pred, scale=4):

    eps = 1e-8

    mean_gt = np.mean(gt, axis=(0, 1))

    rmse = np.sqrt(np.mean((gt - pred) ** 2, axis=(0, 1)))

    ergas = 100 / scale * np.sqrt(np.mean((rmse / (mean_gt + eps)) ** 2))

    return ergas


# ================= Training ================= #

def train(train_loader, val_loader):

    print("Training started")

    time_start = time.time()

    for epoch in range(1, epochs + 1):

        model.train()

        epoch_losses = []

        for iteration, batch in enumerate(train_loader, 1):

            GT, LRHSI, HRMSI = batch

            GT = GT.to(device)
            LRHSI = LRHSI.to(device)
            HRMSI = HRMSI.to(device)

            # ===== Input Safety Check ===== #

            if has_nan(GT) or has_nan(LRHSI) or has_nan(HRMSI):

                print("NaN detected in input data")
                save_checkpoint(model, optimizer, epoch, "input_nan")
                return

            optimizer.zero_grad()

            with autocast():

                output_HRHSI, UP_LRHSI, Highpass = model(LRHSI, HRMSI)

                # ===== Output Safety ===== #

                if has_nan(output_HRHSI):

                    print("NaN detected in model output")
                    save_checkpoint(model, optimizer, epoch, "output_nan")
                    return

                loss = criterion(output_HRHSI, GT)

                if torch.isnan(loss) or torch.isinf(loss):

                    print("NaN detected in loss")
                    save_checkpoint(model, optimizer, epoch, "loss_nan")
                    return

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip
            )

            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

            if iteration % 10 == 0:

                max_val = output_HRHSI.abs().max().item()

                print(
                    f"Epoch[{epoch}] "
                    f"Iter[{iteration}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f} "
                    f"MaxAct: {max_val:.4f}"
                )

        lr_scheduler.step()

        train_loss = np.mean(epoch_losses)

        writer.add_scalar("train/loss", train_loss, epoch)

        print(f"\nEpoch {epoch}/{epochs}  Train Loss: {train_loss:.6f}")

        print("Elapsed:", time.time() - time_start)

        # ===== Regular Checkpoint ===== #

        if epoch % ckpt_step == 0:
            save_checkpoint(model, optimizer, epoch)

        # ================= Validation ================= #

        if epoch % 20 == 0:

            model.eval()

            val_losses = []

            with torch.no_grad():

                for batch in val_loader:

                    GT, LRHSI, HRMSI = batch

                    GT = GT.to(device)
                    LRHSI = LRHSI.to(device)
                    HRMSI = HRMSI.to(device)

                    output_HRHSI, _, _ = model(LRHSI, HRMSI)

                    loss = criterion(output_HRHSI, GT)

                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)

            writer.add_scalar("val/loss", val_loss, epoch)

            print("Validation Loss:", val_loss)

    writer.close()


# ================= Testing ================= #

def test():

    test_set = MDASSRDataset(
        lr_hsi_path="./Augsburg_data_4_publication/sr_deep_model_data/EeteS_EnMAP_30m_deep_valid.tif",
        hr_msi_path="./Augsburg_data_4_publication/sr_deep_model_data/EeteS_Sentinel_2_10m_deep_valid.tif",
        hr_hsi_path="./Augsburg_data_4_publication/sr_deep_model_data/EeteS_EnMAP_10m_deep_valid.tif",
        crop_size=64,
        num_samples=64,
        scale_factor=3,
    )

    loader = DataLoader(test_set, batch_size=1)

    model.eval()

    total_psnr = 0
    total_ssim = 0
    total_sam = 0
    total_ergas = 0

    count = 0

    with torch.no_grad():

        for GT, LRHSI, HRMSI in loader:

            GT = GT.to(device)
            LRHSI = LRHSI.to(device)
            HRMSI = HRMSI.to(device)

            output_HRHSI, _, _ = model(LRHSI, HRMSI)

            mse = F.mse_loss(output_HRHSI, GT)

            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))

            out_np = output_HRHSI.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt_np = GT.squeeze(0).permute(1, 2, 0).cpu().numpy()

            ssim_val = ssim_metric(
                gt_np.mean(axis=2),
                out_np.mean(axis=2),
                data_range=1.0
            )

            sam_val = compute_sam(gt_np, out_np)

            ergas_val = compute_ergas(gt_np, out_np)

            total_psnr += psnr.item()
            total_ssim += ssim_val
            total_sam += sam_val
            total_ergas += ergas_val

            count += 1

    print("\n=== Test Results ===")

    print("PSNR :", total_psnr / count)
    print("SSIM :", total_ssim / count)
    print("SAM  :", total_sam / count)
    print("ERGAS:", total_ergas / count)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters:", num_params)


# ================= Main ================= #

if __name__ == "__main__":

    train_set = MDASSRDataset(
        lr_hsi_path="./Augsburg_data_4_publication/sr_deep_model_data/EeteS_EnMAP_30m_deep_train.tif",
        hr_msi_path="./Augsburg_data_4_publication/sr_deep_model_data/EeteS_Sentinel_2_10m_deep_train.tif",
        hr_hsi_path="./Augsburg_data_4_publication/sr_deep_model_data/EeteS_EnMAP_10m_deep_train.tif",
        crop_size=96,
        num_samples=8000,
        scale_factor=3,
    )

    val_set = MDASSRDataset(
        lr_hsi_path="./Augsburg_data_4_publication/sr_deep_model_data/EeteS_EnMAP_30m_deep_valid.tif",
        hr_msi_path="./Augsburg_data_4_publication/sr_deep_model_data/EeteS_Sentinel_2_10m_deep_valid.tif",
        hr_hsi_path="./Augsburg_data_4_publication/sr_deep_model_data/EeteS_EnMAP_10m_deep_valid.tif",
        crop_size=96,
        num_samples=2000,
        scale_factor=3,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # ===== Load trained model ===== #
    ckpt_path = "Trained_model/epoch_100_normal.pth"

    load_checkpoint(model, path=ckpt_path)

    model.eval()

    # ===== Run test ===== #
    test()