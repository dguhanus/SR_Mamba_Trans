import torch
import rasterio
import numpy as np
import random
from torch.utils.data import Dataset


def load_tif(path, scale=True):
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)  # [C, H, W]

    if scale:
        img /= 10000.0

    return img


class MDASSRDataset(Dataset):
    """
    LR-HSI (30m) + HR-MSI (10m) → HR-HSI (10m)
    """

    def __init__(
        self,
        lr_hsi_path,
        hr_msi_path,
        hr_hsi_path,
        crop_size=96,
        num_samples=2000,
        scale_factor=3,
    ):

        super().__init__()

        self.scale = scale_factor
        self.crop = crop_size
        self.num_samples = num_samples

        self.lr_hsi = load_tif(lr_hsi_path)
        self.hr_msi = load_tif(hr_msi_path)
        self.hr_hsi = load_tif(hr_hsi_path)

        self.H = self.hr_hsi.shape[1]
        self.W = self.hr_hsi.shape[2]


    def __len__(self):
        return self.num_samples


    def __getitem__(self, idx):

        y = random.randint(0, self.H - self.crop)
        x = random.randint(0, self.W - self.crop)

        hr_hsi = self.hr_hsi[:, y:y+self.crop, x:x+self.crop]
        hr_msi = self.hr_msi[:, y:y+self.crop, x:x+self.crop]

        lr_y = y // self.scale
        lr_x = x // self.scale

        lr_h = self.crop // self.scale
        lr_w = self.crop // self.scale

        lr_hsi = self.lr_hsi[:, lr_y:lr_y+lr_h, lr_x:lr_x+lr_w]

        return (
            torch.from_numpy(hr_hsi).float(),   # GT
            torch.from_numpy(lr_hsi).float(),   # LR-HSI (LOW RES)
            torch.from_numpy(hr_msi).float(),   # HR-MSI
        )