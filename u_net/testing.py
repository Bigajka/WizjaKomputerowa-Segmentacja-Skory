import torch
import cv2
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Konfiguracja
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#MODEL_PATH = "models/unet_skin_1.pth"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "unet_skin_1.pth")
INPUT_IMAGE = os.path.join(os.path.dirname(__file__), "wanted.jpg")
#INPUT_IMAGE = "wojtas3.jpg"
OUTPUT_DIR = "results"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Przygotowanie modelu (identycznie jak przy trenowaniu)
model = smp.Unet(
    encoder_name="resnet34",
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

# Wczytanie wag
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval() # Tryb ewaluacji (wyłącza Dropout/BatchNorm)

# Preprocessing (taki sam jak val_transform w treningu)
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def predict_skin(img_path):
    # Wczytanie obrazu
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]

    # Transformacja
    input_tensor = transform(image=image)["image"].unsqueeze(0).to(DEVICE)

    # Predykcja
    with torch.no_grad():
        output = model(input_tensor)
        # Nakładamy sigmoid, bo model wypluwa logity
        probability_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
    # Zamiana na maskę binarną (próg 0.5)
    binary_mask = (probability_mask > 0.5).astype(np.uint8) * 255

    # Powrót do oryginalnego rozmiaru (jeśli obraz nie był 512x512)
    binary_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]))

    # Zapis wyniku
    filename = os.path.basename(img_path)
    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, binary_mask)
    print(f"Maska zapisana w: {save_path}")

# Uruchomienie
if __name__ == "__main__":
    predict_skin(INPUT_IMAGE)