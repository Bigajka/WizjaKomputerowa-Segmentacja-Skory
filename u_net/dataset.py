import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class SkinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.masks = []

        # Przeszukiwanie folderów 001 i 002
        for subfolder in ['001', '002']:
            img_path = os.path.join(root_dir, subfolder, 'images')
            mask_path = os.path.join(root_dir, subfolder, 'masks')
            
            # Pobieramy posortowane nazwy plików
            filenames = sorted(os.listdir(img_path))
            for f in filenames:
                if f.endswith('.png'):
                    self.images.append(os.path.join(img_path, f))
                    self.masks.append(os.path.join(mask_path, f))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Wczytywanie obrazu (RGB) i maski (Grayscale)
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        # Normalizacja maski do formatu 0 i 1 (skóra to 1)
        mask = (mask > 128).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask