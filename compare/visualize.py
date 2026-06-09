import os
import cv2
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
import matplotlib.pyplot as plt

# ==========================================
# USTAWIENIA
# ==========================================
IMAGE_ROOT = "test_images/Pratheepan_Dataset"
MASK_ROOT = "test_images/Ground_Truth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = "models"
SEGFORMER_DIR = "models"
UNET_PATH = os.path.join(MODELS_DIR, "unet_skin_best.pth")
DEEPLABV3_R50_PATH = os.path.join(MODELS_DIR, "deeplabv3_resnet50_model_best.pth")
DEEPLABV3_R101_PATH = os.path.join(MODELS_DIR, "deeplabv3_resnet101_model_best.pth")
DEEPLABV3P_R50_PATH = os.path.join(MODELS_DIR, "deeplabv3plus_resnet50_model_best.pth")
DEEPLABV3P_R101_PATH = os.path.join(MODELS_DIR, "deeplabv3plus_resnet101_model_best.pth")

FOLDER_MAPPING = {
    "FacePhoto": "GroundT_FacePhoto",
    "FamilyPhoto": "GroundT_FamilyPhoto"
}

# Transformacja dla modelu (z normalizacja i tensorami)
test_transform = A.Compose([
    A.LongestMaxSize(max_size=512), 
    A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_REFLECT_101),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Transformacja tylko do wyswietlania (aby oryginal mial identyczna geometrie co maski)
vis_transform = A.Compose([
    A.LongestMaxSize(max_size=512), 
    A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_REFLECT_101)
])

# ==========================================
# FUNKCJE ZBIERAJACE
# ==========================================
def zbierz_pary_pratheepan(image_root, mask_root, mapping):
    pary = []
    for img_subfolder, mask_subfolder in mapping.items():
        img_dir = os.path.join(image_root, img_subfolder)
        mask_dir = os.path.join(mask_root, mask_subfolder)
        
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            continue
            
        for img_name in sorted(os.listdir(img_dir)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(img_dir, img_name)
                base_name = os.path.splitext(img_name)[0]
                mask_name = base_name + ".png"
                mask_path = os.path.join(mask_dir, mask_name)
                    
                if os.path.exists(mask_path):
                    pary.append((img_path, mask_path, img_name))
                else:
                    alt_mask_path = os.path.join(mask_dir, img_name)
                    if os.path.exists(alt_mask_path):
                        pary.append((img_path, alt_mask_path, img_name))
    return pary

class VisDataset(Dataset):
    def __init__(self, lista_par, transform=None, vis_transform=None):
        self.lista_par = lista_par
        self.transform = transform
        self.vis_transform = vis_transform

    def __len__(self):
        return len(self.lista_par)

    def __getitem__(self, idx):
        img_path, mask_path, img_name = self.lista_par[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.int64)

        if self.vis_transform:
            orig_img_np = self.vis_transform(image=image)["image"]
        else:
            orig_img_np = image

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].clone().detach().to(torch.long)
        
        return {
            "pixel_values": image_tensor, 
            "labels": mask_tensor, 
            "name": img_name,
            "orig_img": orig_img_np
        }

# ==========================================
# GLOWNA PETLA WIZUALIZACJI
# ==========================================
if __name__ == "__main__":
    print("Skanowanie struktury Pratheepan Dataset...")
    wszystkie_pary = zbierz_pary_pratheepan(IMAGE_ROOT, MASK_ROOT, FOLDER_MAPPING)
    
    if len(wszystkie_pary) == 0:
        print("Blad krytyczny: Skrypt nie znalazl par obraz-maska. Przerywam.")
        exit(1)
        
    # 1. Losujemy dokladnie 5 par
    random.seed(42) 
    wylosowane_pary = random.sample(wszystkie_pary, min(5, len(wszystkie_pary)))
    
    test_dataset = VisDataset(wylosowane_pary, transform=test_transform, vis_transform=vis_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    wyniki_wizualne = {
        "orig": [],
        "gt": [],
        "SegFormer": [],
        "UNet": [],
        "DLV3 (R50)": [],
        "DLV3 (R101)": [],
        "DLV3+ (R50)": [],
        "DLV3+ (R101)": []
    }

    # Wyciagamy oryginaly i Ground Truth z Loadera
    for batch in test_loader:
        wyniki_wizualne["orig"].append(batch["orig_img"][0].numpy())
        wyniki_wizualne["gt"].append(batch["labels"][0].numpy())

    # --- 2. KONFIGURACJA MODELI ---
    models_config = [
        {"name": "SegFormer", "type": "segformer", "path": SEGFORMER_DIR},
        {"name": "UNet", "type": "unet", "path": UNET_PATH},
        {"name": "DLV3 (R50)", "type": "deeplabv3_r50", "path": DEEPLABV3_R50_PATH},
        {"name": "DLV3 (R101)", "type": "deeplabv3_r101", "path": DEEPLABV3_R101_PATH},
        {"name": "DLV3+ (R50)", "type": "deeplabv3plus_r50", "path": DEEPLABV3P_R50_PATH},
        {"name": "DLV3+ (R101)", "type": "deeplabv3plus_r101", "path": DEEPLABV3P_R101_PATH},
    ]

    # --- 3. INFERENCJA ---
    for config in models_config:
        print(f"\nGenerowanie masek dla: {config['name']}...")
        try:
            if config["type"] == "segformer":
                model = SegformerForSemanticSegmentation.from_pretrained(config["path"], num_labels=2, ignore_mismatched_sizes=True).to(DEVICE)
            elif config["type"] == "unet":
                model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
            elif config["type"] == "deeplabv3_r50":
                model = deeplabv3_resnet50(weights=None, aux_loss=True)
                model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
                model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
            elif config["type"] == "deeplabv3_r101":
                model = deeplabv3_resnet101(weights=None, aux_loss=True)
                model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
                model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
            elif config["type"] == "deeplabv3plus_r50":
                model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=2)
            elif config["type"] == "deeplabv3plus_r101":
                model = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights=None, in_channels=3, classes=2)

            if config["type"] != "segformer":
                state_dict = torch.load(config['path'], map_location=DEVICE, weights_only=True)
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                cleaned_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(cleaned_state_dict)
                model = model.to(DEVICE)
            
            model.eval()
            
            with torch.no_grad():
                for batch in test_loader:
                    images = batch["pixel_values"].to(DEVICE)
                    
                    if config["type"] == "segformer":
                        outputs = model(pixel_values=images)
                        logits = torch.nn.functional.interpolate(outputs.logits, size=(512, 512), mode="bilinear", align_corners=False)
                        pred_bin = (logits.argmax(dim=1) == 1)
                    elif config["type"] == "unet":
                        logits = model(images).squeeze(1)
                        pred_bin = (torch.sigmoid(logits) > 0.5)
                    elif config["type"] in ["deeplabv3_r50", "deeplabv3_r101"]:
                        probs = torch.softmax(model(images)["out"].float(), dim=1)[:, 1]
                        pred_bin = (probs > 0.5)
                    else: 
                        probs = torch.softmax(model(images).float(), dim=1)[:, 1]
                        pred_bin = (probs > 0.5)
                        
                    wyniki_wizualne[config['name']].append(pred_bin[0].cpu().numpy())

            del model
            torch.cuda.empty_cache()
            print(f"Zakonczono: {config['name']}")
            
        except Exception as e:
            print(f"Blad modelu {config['name']}: {e}")
            for _ in range(len(wylosowane_pary)):
                wyniki_wizualne[config['name']].append(np.zeros((512, 512)))

    kolumny = ["orig", "gt", "SegFormer", "UNet", "DLV3 (R50)", "DLV3 (R101)", "DLV3+ (R50)", "DLV3+ (R101)"]
    tytuly = ["Oryginal", "Prawda (GT)", "SegFormer", "UNet", "DLV3 R50", "DLV3 R101", "DLV3+ R50", "DLV3+ R101"]
    
    n_wierszy = len(wylosowane_pary)
    n_kolumn = len(kolumny)

    # --- 4. RYSOWANIE DUZEJ SIATKI ---
    print("\nGenerowanie glownego wykresu (siatki 5x8)...")
    fig, axes = plt.subplots(n_wierszy, n_kolumn, figsize=(22, 12))
    
    for w in range(n_wierszy):
        for k, kolumna in enumerate(kolumny):
            ax = axes[w, k]
            obraz_do_wyswietlenia = wyniki_wizualne[kolumna][w]
            
            if kolumna == "orig":
                ax.imshow(obraz_do_wyswietlenia)
            else:
                ax.imshow(obraz_do_wyswietlenia, cmap="gray", vmin=0, vmax=1)
                
            ax.set_xticks([])
            ax.set_yticks([])
            
            if w == 0:
                ax.set_title(tytuly[k], fontsize=14, fontweight="bold")

    plt.tight_layout()
    nazwa_pliku = "porownanie_masek_zbiorcze.png"
    plt.savefig(nazwa_pliku, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Zapisano zbiorczy obraz jako '{nazwa_pliku}'.")

    # --- 5. RYSOWANIE POJEDYNCZYCH SIATEK (4 KOLUMNY x 2 WIERSZE) ---
    print("\nGenerowanie osobnych wykresow dla poszczegolnych przykladow (siatki 4x2)...")
    for w in range(n_wierszy):
        fig_small, axes_small = plt.subplots(2, 4, figsize=(16, 8))
        
        for k, kolumna in enumerate(kolumny):
            row = k // 4
            col = k % 4
            ax = axes_small[row, col]
            
            obraz_do_wyswietlenia = wyniki_wizualne[kolumna][w]
            
            if kolumna == "orig":
                ax.imshow(obraz_do_wyswietlenia)
            else:
                ax.imshow(obraz_do_wyswietlenia, cmap="gray", vmin=0, vmax=1)
                
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(tytuly[k], fontsize=12, fontweight="bold")
            
        plt.tight_layout()
        nazwa_pliku_male = f"porownanie_masek_przyklad_{w+1}.png"
        plt.savefig(nazwa_pliku_male, bbox_inches='tight', dpi=150)
        plt.close(fig_small)
        print(f"Zapisano osobny obraz dla przykladu {w+1}: '{nazwa_pliku_male}'")