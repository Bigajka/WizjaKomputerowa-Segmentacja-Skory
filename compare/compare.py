import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from tabulate import tabulate
import segmentation_models_pytorch as smp
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

# ==========================================
# USTAWIENIA
# ==========================================
IMAGE_ROOT = "test_images/Pratheepan_Dataset"
MASK_ROOT = "test_images/Ground_Truth"
BATCH_SIZE = 4
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

test_transform = A.Compose([
    A.LongestMaxSize(max_size=512), 
    A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_REFLECT_101),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ==========================================
# FUNKCJE I KLASY
# ==========================================
def zbierz_pary_pratheepan(image_root, mask_root, mapping):
    pary = []
    for img_subfolder, mask_subfolder in mapping.items():
        img_dir = os.path.join(image_root, img_subfolder)
        mask_dir = os.path.join(mask_root, mask_subfolder)
        
        if not os.path.exists(img_dir):
            print(f"BLAD: Skrypt nie widzi folderu zdjec: {img_dir}")
            continue
        if not os.path.exists(mask_dir):
            print(f"BLAD: Skrypt nie widzi folderu masek: {mask_dir}")
            continue
            
        for img_name in sorted(os.listdir(img_dir)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
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
                    else:
                        print(f"UWAGA: Znaleziono obraz {img_name}, ale brakuje maski.")
                    
    return pary

class PratheepanDataset(Dataset):
    def __init__(self, lista_par, transform=None):
        self.lista_par = lista_par
        self.transform = transform

    def __len__(self):
        return len(self.lista_par)

    def __getitem__(self, idx):
        img_path, mask_path, img_name = self.lista_par[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].clone().detach().to(torch.long)

        return {"pixel_values": image, "labels": mask, "name": img_name}

def evaluate_model(model, dataloader, model_type):
    model.eval()
    iou_scores, dice_scores = [], []
    precision_scores, recall_scores, accuracy_scores = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Testowanie ({model_type})"):
            images = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            if model_type == "segformer":
                outputs = model(pixel_values=images)
                logits = torch.nn.functional.interpolate(
                    outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                preds = logits.argmax(dim=1)
                pred_bin = (preds == 1)
            
            elif model_type == "unet":
                logits = model(images)
                logits = logits.squeeze(1)
                pred_bin = (torch.sigmoid(logits) > 0.5)

            elif model_type == "deeplabv3":
                output = model(images)["out"]
                probs = torch.softmax(output.float(), dim=1)[:, 1]
                pred_bin = (probs > 0.5)

            elif model_type == "deeplabv3plus":
                output = model(images)
                probs = torch.softmax(output.float(), dim=1)[:, 1]
                pred_bin = (probs > 0.5)

            label_bin = (labels == 1)

            # Obliczanie podstawowych wartości (True Positives, False Positives, False Negatives, True Negatives)
            TP = (pred_bin & label_bin).sum(dim=(1, 2)).float()
            FP = (pred_bin & ~label_bin).sum(dim=(1, 2)).float()
            FN = (~pred_bin & label_bin).sum(dim=(1, 2)).float()
            TN = (~pred_bin & ~label_bin).sum(dim=(1, 2)).float()

            union = (pred_bin | label_bin).sum(dim=(1, 2)).float()

            # Obliczanie metryk z zabezpieczeniem przed dzieleniem przez zero (1e-8)
            iou = (TP / (union + 1e-8)).cpu().numpy()
            dice = (2 * TP / (2 * TP + FP + FN + 1e-8)).cpu().numpy() # To jest równe F1-Score
            precision = (TP / (TP + FP + 1e-8)).cpu().numpy()
            recall = (TP / (TP + FN + 1e-8)).cpu().numpy()
            accuracy = ((TP + TN) / (TP + TN + FP + FN + 1e-8)).cpu().numpy()

            iou_scores.extend(iou)
            dice_scores.extend(dice)
            precision_scores.extend(precision)
            recall_scores.extend(recall)
            accuracy_scores.extend(accuracy)

    return np.mean(iou_scores), np.mean(dice_scores), np.mean(precision_scores), np.mean(recall_scores), np.mean(accuracy_scores)

# ==========================================
# GLOWNA PETLA
# ==========================================
if __name__ == "__main__":
    print("Skanowanie struktury Pratheepan Dataset...")
    lista_par = zbierz_pary_pratheepan(IMAGE_ROOT, MASK_ROOT, FOLDER_MAPPING)
    
    if len(lista_par) == 0:
        print("Blad krytyczny: Skrypt nie znalazl par obraz-maska. Przerywam.")
        exit(1)
        
    test_dataset = PratheepanDataset(lista_par, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Znaleziono {len(test_dataset)} prawidlowych par do testow.\n")

    wyniki = []

    # --- TEST 1: SEGFORMER ---
    print("Ladowanie SegFormera...")
    try:
        model_seg = SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_DIR, 
            num_labels=2, 
            ignore_mismatched_sizes=True
        ).to(DEVICE)
        
        iou_val, dice_val, prec_val, rec_val, acc_val = evaluate_model(model_seg, test_loader, model_type="segformer")
        wyniki.append(["SegFormer", f"{iou_val:.4f}", f"{dice_val:.4f}", f"{prec_val:.4f}", f"{rec_val:.4f}", f"{acc_val:.4f}"])
        del model_seg
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Blad ladowania SegFormera: {e}")

    # --- TEST 2: POZOSTALE MODELE ---
    models_config = [
        {"name": "UNet (ResNet34)", "path": UNET_PATH, "type": "unet", "eval_type": "unet"},
        {"name": "DeepLabV3 (ResNet50)", "path": DEEPLABV3_R50_PATH, "type": "deeplabv3_r50", "eval_type": "deeplabv3"},
        {"name": "DeepLabV3 (ResNet101)", "path": DEEPLABV3_R101_PATH, "type": "deeplabv3_r101", "eval_type": "deeplabv3"},
        {"name": "DeepLabV3+ (ResNet50)", "path": DEEPLABV3P_R50_PATH, "type": "deeplabv3plus_r50", "eval_type": "deeplabv3plus"},
        {"name": "DeepLabV3+ (ResNet101)", "path": DEEPLABV3P_R101_PATH, "type": "deeplabv3plus_r101", "eval_type": "deeplabv3plus"},
    ]

    for config in models_config:
        print(f"\nLadowanie modelu {config['name']}...")
        try:
            # Tworzenie architektury na podstawie typu
            if config["type"] == "unet":
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

            # Ladowanie wag z pliku
            state_dict = torch.load(config['path'], map_location=DEVICE, weights_only=True)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            # Czyszczenie wag z przedrostka _orig_mod.
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('_orig_mod.', '')
                cleaned_state_dict[new_k] = v

            model.load_state_dict(cleaned_state_dict)
            model = model.to(DEVICE)
            
            iou_val, dice_val, prec_val, rec_val, acc_val = evaluate_model(model, test_loader, model_type=config['eval_type'])
            wyniki.append([config['name'], f"{iou_val:.4f}", f"{dice_val:.4f}", f"{prec_val:.4f}", f"{rec_val:.4f}", f"{acc_val:.4f}"])
            
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Blad ladowania {config['name']}: {e}")

    # --- PODSUMOWANIE ---
    print("\n" + "="*85)
    print("WYNIKI POROWNANIA MODELI (PRATHEEPAN DATASET)")
    print("="*85)
    if wyniki:
        print(tabulate(wyniki, headers=["Model", "IoU", "Dice (F1)", "Precision", "Recall", "Accuracy"], tablefmt="grid"))
    else:
        print("Nie udalo sie przetestowac zadnego modelu.")