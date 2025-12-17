import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_metric

from mdas_dataset import MDASSRDataset
from mdas import MainNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH = 96
SCALE = 3

valLR  = "./Augsburg_data_4_publication/sr_deep_model_data/EeteS_EnMAP_30m_deep_valid.tif"
valMSI = "./Augsburg_data_4_publication/sr_deep_model_data/EeteS_Sentinel_2_10m_deep_valid.tif"
valHR  = "./Augsburg_data_4_publication/sr_deep_model_data/EeteS_EnMAP_10m_deep_valid.tif"


def compute_psnr(gt, pred, max_val=1.0):
    """
    gt, pred: torch tensors in [0,1], shape (C,H,W)
    """
    mse = torch.mean((gt - pred) ** 2).item()
    if mse <= 1e-12:
        return 100.0
    return 10.0 * np.log10((max_val ** 2) / mse)


def compute_sam_np(gt, pred):
    """
    gt, pred: (H, W, C) arrays in [0,1]
    """
    eps = 1e-8
    dot = np.sum(gt * pred, axis=-1)
    n1 = np.linalg.norm(gt, axis=-1)
    n2 = np.linalg.norm(pred, axis=-1)
    cos = dot / (n1 * n2 + eps)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.arccos(cos)
    return np.degrees(np.mean(ang))


def compute_ergas(gt, pred, scale=SCALE):
    """
    gt, pred: (H, W, C) arrays
    """
    eps = 1e-8
    mean_gt = np.mean(gt, axis=(0, 1))
    rmse = np.sqrt(np.mean((gt - pred) ** 2, axis=(0, 1)))
    return 100.0 / scale * np.sqrt(np.mean((rmse / (mean_gt + eps)) ** 2))


def main():
    # dataset
    val_set = MDASSRDataset(
        lr_hsi_path=valLR,
        hr_msi_path=valMSI,
        hr_hsi_path=valHR,
        crop_size=PATCH,
        num_samples=1000,     # number of random patches to evaluate
        scale_factor=SCALE,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # model
    model = MainNet(C_hsi=242, C_msi=4, embed_dim=48).to(DEVICE)
    ckpt = "best_mdas.pth"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"{ckpt} not found. Train first.")
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_sam  = 0.0
    total_ergas = 0.0
    n = 0

    with torch.no_grad():
        for hr, lr_up, msi in val_loader:
            hr = hr.to(DEVICE).clamp(0, 1)          # [1,242,H,W]
            lr_up = lr_up.to(DEVICE).clamp(0, 1)    # [1,242,H,W]
            msi = msi.to(DEVICE).clamp(0, 1)        # [1,4,H,W]

            sr = model(lr_up, msi).clamp(0, 1)      # [1,242,H,W]

            # PSNR
            psnr = compute_psnr(hr.squeeze(0), sr.squeeze(0))
            total_psnr += psnr

            # convert to (H,W,C) numpy
            gt_np = hr.squeeze(0).permute(1, 2, 0).cpu().numpy()
            sr_np = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # SSIM on band-mean grayscale
            total_ssim += ssim_metric(
                gt_np.mean(axis=-1),
                sr_np.mean(axis=-1),
                data_range=1.0,
            )

            # SAM & ERGAS
            total_sam += compute_sam_np(gt_np, sr_np)
            total_ergas += compute_ergas(gt_np, sr_np)

            n += 1

    if n == 0:
        print("No samples evaluated.")
        return

    print("\n=== MDAS eval on patches ===")
    print(f"Samples: {n}")
    print(f"PSNR : {total_psnr / n:.4f} dB")
    print(f"SSIM : {total_ssim / n:.4f}")
    print(f"SAM  : {total_sam  / n:.4f} deg")
    print(f"ERGAS: {total_ergas / n:.4f}")


if __name__ == "__main__":
    main()
