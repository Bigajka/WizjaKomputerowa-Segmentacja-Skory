import os
# --- FIX DLA RETINAFACE (Wymuszenie starszego Keras) ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import torch
import numpy as np
import urllib.request
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

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from retinaface import RetinaFace

# ==========================================
# USTAWIENIA
# ==========================================
IMAGE_ROOT = "test_images/Pratheepan_Dataset"
MASK_ROOT = "test_images/Ground_Truth"
VIS_OUTPUT_DIR = "korelacja_wizualizacje"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIS_LIMIT_PER_MODEL = 5 # <--- Twój limit obrazków dla raportu

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

vis_transform = A.Compose([
    A.LongestMaxSize(max_size=512), 
    A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_REFLECT_101)
])

# ==========================================
# KLASY ANALIZY KORELACJI
# ==========================================
class CorrelationAnalyzer:
    def __init__(self, hand_margin=20, face_margin=10):
        self.hand_margin = hand_margin
        self.face_margin = face_margin
        
        self.model_path = "hand_landmarker.task"
        if not os.path.exists(self.model_path):
            print("Pobieranie oficjalnego modelu MediaPipe Hand...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, 
            num_hands=10,
            min_hand_detection_confidence=0.4
        )
        self.hands = vision.HandLandmarker.create_from_options(options)

    def get_face_bboxes(self, img_rgb):
        faces = RetinaFace.detect_faces(img_rgb)
        bboxes = []
        if isinstance(faces, dict):
            for key in faces.keys():
                identity = faces[key]
                x1, y1, x2, y2 = identity["facial_area"]
                x1 = max(0, x1 - self.face_margin)
                y1 = max(0, y1 - self.face_margin)
                x2 = min(img_rgb.shape[1], x2 + self.face_margin)
                y2 = min(img_rgb.shape[0], y2 + self.face_margin)
                bboxes.append([x1, y1, x2, y2])
        return bboxes

    def get_hand_bboxes(self, img_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = self.hands.detect(mp_image)
        bboxes = []
        h, w, _ = img_rgb.shape
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)
                bboxes.append([
                    max(0, x_min - self.hand_margin),
                    max(0, y_min - self.hand_margin),
                    min(w, x_max + self.hand_margin),
                    min(h, y_max + self.hand_margin)
                ])
        return bboxes

    def generate_detection_mask(self, image_shape, face_bboxes, hand_bboxes):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        for box in face_bboxes + hand_bboxes:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 1
        return mask

    def calculate_correlation(self, seg_mask, det_mask):
        intersection = np.logical_and(seg_mask, det_mask).sum()
        seg_area = seg_mask.sum()
        det_area = det_mask.sum()
        skin_in_bbox_ratio = intersection / seg_area if seg_area > 0 else 0.0
        bbox_coverage_ratio = intersection / det_area if det_area > 0 else 0.0
        return skin_in_bbox_ratio, bbox_coverage_ratio

def save_visualization(img_rgb, seg_mask, face_bboxes, hand_bboxes, output_path):
    vis_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    blue_mask = np.zeros_like(vis_img)
    blue_mask[seg_mask == 1] = [255, 0, 0]
    cv2.addWeighted(blue_mask, 0.5, vis_img, 1.0, 0, vis_img)
    for box in face_bboxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, 'Twarz', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for box in hand_bboxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis_img, 'Dlon', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(output_path, vis_img)

# ==========================================
# LADOWANIE DANYCH
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
                mask_path = os.path.join(mask_dir, base_name + ".png")
                if os.path.exists(mask_path):
                    pary.append((img_path, mask_path, img_name))
                else:
                    alt_mask_path = os.path.join(mask_dir, img_name)
                    if os.path.exists(alt_mask_path):
                        pary.append((img_path, alt_mask_path, img_name))
    return pary

class CorrelationDataset(Dataset):
    def __init__(self, lista_par, transform=None, vis_transform=None):
        self.lista_par = lista_par
        self.transform = transform
        self.vis_transform = vis_transform

    def __len__(self):
        return len(self.lista_par)

    def __getitem__(self, idx):
        img_path, _, img_name = self.lista_par[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        orig_img_np = self.vis_transform(image=image)["image"] if self.vis_transform else image
        if self.transform:
            image_tensor = self.transform(image=image)["image"]
        else:
            image_tensor = torch.tensor(image).permute(2, 0, 1).float()
        return {"pixel_values": image_tensor, "orig_img": orig_img_np, "name": img_name}

# ==========================================
# GLOWNA PETLA (Zoptymalizowana)
# ==========================================
if __name__ == "__main__":
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    
    print("Skanowanie struktury Pratheepan Dataset...")
    lista_par = zbierz_pary_pratheepan(IMAGE_ROOT, MASK_ROOT, FOLDER_MAPPING)
    dataset = CorrelationDataset(lista_par, transform=test_transform, vis_transform=vis_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # --- 1. ETAP: PRE-KALKULACJA DETEKCJI (Oszczednosc czasu) ---
    print("\nETAP 1: Analiza detekcyjna wszystkich obrazow (MediaPipe & RetinaFace)...")
    analyzer = CorrelationAnalyzer()
    detekcje_cache = {}
    
    for batch in tqdm(dataloader, desc="Wykrywanie Twarzy i Dloni"):
        img_rgb = batch["orig_img"][0].numpy()
        img_name = batch["name"][0]
        face_boxes = analyzer.get_face_bboxes(img_rgb)
        hand_boxes = analyzer.get_hand_bboxes(img_rgb)
        det_mask = analyzer.generate_detection_mask(img_rgb.shape, face_boxes, hand_boxes)
        
        detekcje_cache[img_name] = {
            "face": face_boxes,
            "hand": hand_boxes,
            "mask": det_mask,
            "rgb": img_rgb
        }

    # --- 2. ETAP: TESTOWANIE WSZYSTKICH MODELI ---
    models_config = [
        {"name": "SegFormer", "type": "segformer", "path": SEGFORMER_DIR},
        {"name": "UNet (ResNet34)", "type": "unet", "path": UNET_PATH},
        {"name": "DeepLabV3 (R50)", "type": "deeplabv3_r50", "path": DEEPLABV3_R50_PATH},
        {"name": "DeepLabV3 (R101)", "type": "deeplabv3_r101", "path": DEEPLABV3_R101_PATH},
        {"name": "DeepLabV3+ (R50)", "type": "deeplabv3plus_r50", "path": DEEPLABV3P_R50_PATH},
        {"name": "DeepLabV3+ (R101)", "type": "deeplabv3plus_r101", "path": DEEPLABV3P_R101_PATH},
    ]

    tabela_zbiorcza = []

    print("\nETAP 2: Analiza korelacji dla poszczegolnych modeli segmentacji...\n")
    for config in models_config:
        print(f"Ladowanie modelu: {config['name']} ...")
        try:
            # Tworzenie modelu
            if config["type"] == "segformer":
                model = SegformerForSemanticSegmentation.from_pretrained(config["path"], num_labels=2, ignore_mismatched_sizes=True)
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

            # Ladowanie wag dla wszystkich oprocz SegFormera
            if config["type"] != "segformer":
                state_dict = torch.load(config['path'], map_location=DEVICE, weights_only=True)
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                cleaned_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(cleaned_state_dict)

            model = model.to(DEVICE)
            model.eval()

            avg_skin_in_bbox, avg_bbox_cov = [], []
            zapisane_wizualizacje = 0

            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Inferencja {config['name']}"):
                    images = batch["pixel_values"].to(DEVICE)
                    img_name = batch["name"][0]
                    
                    # Generowanie maski w zaleznosci od architektury
                    if config["type"] == "segformer":
                        outputs = model(pixel_values=images)
                        logits = torch.nn.functional.interpolate(outputs.logits, size=(512, 512), mode="bilinear", align_corners=False)
                        pred_bin = (logits.argmax(dim=1) == 1)[0].cpu().numpy().astype(np.uint8)
                    elif config["type"] == "unet":
                        logits = model(images).squeeze(1)
                        pred_bin = (torch.sigmoid(logits) > 0.5)[0].cpu().numpy().astype(np.uint8)
                    elif config["type"] in ["deeplabv3_r50", "deeplabv3_r101"]:
                        probs = torch.softmax(model(images)["out"].float(), dim=1)[:, 1]
                        pred_bin = (probs > 0.5)[0].cpu().numpy().astype(np.uint8)
                    else: 
                        probs = torch.softmax(model(images).float(), dim=1)[:, 1]
                        pred_bin = (probs > 0.5)[0].cpu().numpy().astype(np.uint8)

                    # Pobieranie pre-kalkulowanych danych
                    cache = detekcje_cache[img_name]
                    skin_in_bbox, bbox_cov = analyzer.calculate_correlation(pred_bin, cache["mask"])
                    
                    avg_skin_in_bbox.append(skin_in_bbox)
                    avg_bbox_cov.append(bbox_cov)

                    # Zapisywanie wizualizacji tylko do ustalonego limitu
                    if zapisane_wizualizacje < VIS_LIMIT_PER_MODEL:
                        bezpieczna_nazwa = config['name'].replace("+", "plus").replace(" ", "_").replace("(", "").replace(")", "")
                        vis_path = os.path.join(VIS_OUTPUT_DIR, f"{bezpieczna_nazwa}_{img_name}")
                        save_visualization(cache["rgb"], pred_bin, cache["face"], cache["hand"], vis_path)
                        zapisane_wizualizacje += 1

            # Dodawanie srednich wynikow do glownej tabeli
            tabela_zbiorcza.append([
                config['name'], 
                f"{np.mean(avg_skin_in_bbox):.4f}", 
                f"{np.mean(avg_bbox_cov):.4f}"
            ])

            # Sprzatanie pamieci po modelu
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Blad testowania modelu {config['name']}: {e}")
            tabela_zbiorcza.append([config['name'], "BLAD", "BLAD"])

    # --- PODSUMOWANIE KONCOWE ---
    print("\n" + "="*70)
    print("FINALNY RAPORT KORELACJI MODELI SEGMENTACJI Z DETEKCJA")
    print("="*70)
    print(tabulate(tabela_zbiorcza, headers=["Model Segmentacji", "Srednia: Skin in BBox", "Srednia: BBox Coverage"], tablefmt="grid"))
    print(f"\nUwaga: Ograniczono zapisywanie obrazow podgladowych do {VIS_LIMIT_PER_MODEL} sztuk na model.")
    print(f"Zostaly one zapisane w folderze: '{VIS_OUTPUT_DIR}'")