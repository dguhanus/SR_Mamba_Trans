import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn.functional as F

# ============================================================
# Dataset (from data.py)
# ============================================================
def psnr(pred, target, max_val=1.0):
    """Compute PSNR between two tensors."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(max_val / math.sqrt(mse.item()))

def eval_model(model_path, test_path, batch_size=1, device=None):
    """
    Evaluate trained model on test H5 dataset and compute PSNR.
    Args:
        model_path (str): Path to .pth model file.
        test_path (str): Path to test .h5 file.
        batch_size (int): Batch size for evaluation.
        device (torch.device): CUDA or CPU.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    test_set = DatasetFromHdf5(test_path)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load model
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for gt, lrhsi, rgb in test_loader:
            gt, lrhsi, rgb = gt.to(device), lrhsi.to(device), rgb.to(device)
            pred = model(lrhsi, rgb)
            pred = torch.clamp(pred, 0, 1)

            # PSNR per sample
            for i in range(pred.size(0)):
                psnr_val = psnr(pred[i], gt[i])
                total_psnr += psnr_val
                count += 1

    avg_psnr = total_psnr / count
    print(f"📈 Average PSNR on test set: {avg_psnr:.2f} dB")
    return avg_psnr
class DatasetFromHdf5(Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path, 'r')
        self.GT = dataset['GT']
        self.LRHSI = dataset['LRHSI']
        self.RGB = dataset['RGB']

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.GT[index]).float(),
            torch.from_numpy(self.LRHSI[index]).float(),
            torch.from_numpy(self.RGB[index]).float(),
        )

    def __len__(self):
        return self.GT.shape[0]


# ============================================================
# Model (simplified MSST-like HSI–RGB fusion)
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Net(nn.Module):
    """
    Simplified MSST-style model:
    - Spectral branch processes LRHSI (31×16×16)
    - Spatial branch processes RGB (3×64×64)
    - Features are upsampled + fused → output 31×64×64
    """
    def __init__(self, hsi_channels=31):
        super(Net, self).__init__()

        # Spectral (HSI) branch
        self.spectral_layers = nn.Sequential(
            ConvBlock(hsi_channels, 64),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            ConvBlock(64, 64)
        )

        # Spatial (RGB) branch
        self.spatial_layers = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64)
        )

        # Fusion
        self.fusion = nn.Sequential(
            ConvBlock(128, 64),
            nn.Conv2d(64, hsi_channels, 3, padding=1)
        )

    def forward(self, lrhsi, rgb):
        # Expect shapes:
        # lrhsi: [B,31,16,16], rgb: [B,3,64,64]
        hsi_feat = self.spectral_layers(lrhsi)
        rgb_feat = self.spatial_layers(rgb)
        fused = torch.cat([hsi_feat, rgb_feat], dim=1)
        out = self.fusion(fused)
        return out


# ============================================================
# Training setup
# ============================================================
def main():
    # -----------------------------
    # Paths & parameters
    # -----------------------------
    train_path = "Train_CAVE.h5"
    valid_path = "Valid_CAVE.h5"
    test_path  = "demo_cave_patches.h5"
    save_dir   = "./checkpoints_msst"
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = 100
    batch_size = 8
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load data
    # -----------------------------
    train_set = DatasetFromHdf5(train_path)
    valid_set = DatasetFromHdf5(valid_path)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=2)

    # -----------------------------
    # Model, loss, optimizer
    # -----------------------------
    model = Net().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # -----------------------------
    # Validation function
    # -----------------------------
    def validate():
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for gt, lrhsi, rgb in valid_loader:
                gt, lrhsi, rgb = gt.to(device), lrhsi.to(device), rgb.to(device)
                output = model(lrhsi, rgb)
                loss = criterion(output, gt)
                total_loss += loss.item()
        return total_loss / len(valid_loader)

    # -----------------------------
    # Training loop
    # -----------------------------
    best_val = float("inf")
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for gt, lrhsi, rgb in train_loader:
            gt, lrhsi, rgb = gt.to(device), lrhsi.to(device), rgb.to(device)

            optimizer.zero_grad()
            pred = model(lrhsi, rgb)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        val_loss = validate()

        print(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("✅ Saved best model")

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    print("🎯 Training complete.")

    
if __name__ == "__main__":
    # main()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_model(
        model_path="./checkpoints_msst/best_model.pth",
        test_path="demo_cave_patches.h5",
        device=device
    )
