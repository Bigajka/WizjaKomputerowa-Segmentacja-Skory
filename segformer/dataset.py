import os
from pathlib import Path

import ions
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

REPO_NAME = "WizjaKomputerowa-Segmentacja-Skory"
DATASET_PATH = os.path.join(REPO_NAME, "dataset_final")

train_transform = A.Compose([
    A.Resize(512, 512), 
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.2),
    A.GaussianBlur(p=0.1),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(512, 512), 
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])


class ECUDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        mask = (mask > 127).astype(np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].clone().detach().to(torch.long)
        else:
            image = torch.tensor(image).permute(2, 0, 1).float()
            mask = torch.tensor(mask, dtype=torch.long)

        return {
            "pixel_values": image,
            "labels": mask
        }


def zbierz_pary(root_dir):
    root = Path(root_dir)
    pary = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        img_dir = folder / "images"
        mask_dir = folder / "masks"
        for img_path in sorted(img_dir.glob("*.png")):
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                pary.append((img_path, mask_path))
    return pary


def przygotuj_loadery(batch_size=8):
    pary = zbierz_pary(DATASET_PATH)

    # Split 80/20
    train_size = int(0.8 * len(pary))
    test_size = len(pary) - train_size

    train_pary = pary[:train_size]
    test_pary = pary[train_size:]

    train_ds = ECUDataset(train_pary, transform=train_transform)
    test_ds = ECUDataset(test_pary, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"✅ Train: {len(train_ds)} obrazów ({len(train_loader)} batchy)")
    print(f"✅ Test:  {len(test_ds)} obrazów ({len(test_loader)} batchy)")

    return train_loader, test_loader