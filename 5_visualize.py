import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerForSemanticSegmentation
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ================================================================
MODEL_PATH = "segformer_ecu"
TEST_IMAGES_DIR = "WizjaKomputerowa-Segmentacja-Skory/dataset_final/001/images"
TEST_MASKS_DIR = "WizjaKomputerowa-Segmentacja-Skory/dataset_final/001/masks"
# ================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# POPRAWKA 1: Dodane A.Resize(512, 512) aby rozmiar zgadzał się z modelem
val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


def generuj_wizualizacje(num_samples=3):
    # Ważne: ładujemy model z flagą output_attentions=True
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH, output_attentions=True)
    model.to(device)
    model.eval()

    image_paths = list(Path(TEST_IMAGES_DIR).glob("*.png"))[:num_samples]

    # Tworzymy siatkę: Obraz | Ground Truth | Predykcja | Mapa Atencji
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))

    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            mask_path = Path(TEST_MASKS_DIR) / img_path.name
            img_pil = Image.open(img_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L")

            img_np = np.array(img_pil)
            mask_np = (np.array(mask_pil) > 127).astype(np.uint8)

            transformed = val_transform(image=img_np)
            pixel_values = transformed["image"].unsqueeze(0).to(device)

            outputs = model(pixel_values=pixel_values)

            # 1. Wyciągnięcie predykcji
            logits = torch.nn.functional.interpolate(
                outputs.logits, size=img_pil.size[::-1], mode="bilinear", align_corners=False
            )
            pred_mask = logits.argmax(dim=1).squeeze().cpu().numpy()

            # POPRAWKA 2: Wyciągnięcie Mapy Atencji (Attention Map)
            # Bierzemy atencję z ostatniej warstwy
            attention = outputs.attentions[-1]  # shape: (batch, heads, seq_len, seq_len)

            # Uśredniamy po głowach (heads) - to nasz wymiar nr 1
            attn_weights = attention.mean(dim=1)  # shape: (batch, seq_len, seq_len)

            # Uśredniamy po zapytaniach (queries) żeby sprawdzić, na co patrzy cała sieć
            saliency = attn_weights.mean(dim=1).squeeze()  # shape: (seq_len,)

            # SegFormer tnie obraz na łatki, musimy to zrekonstruować do 2D
            grid_size = int(np.sqrt(saliency.shape[0]))
            attention_map = saliency.reshape(grid_size, grid_size).cpu().numpy()

            # Rysowanie
            axes[i, 0].imshow(img_pil)
            axes[i, 0].set_title("Oryginalny obraz")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(mask_np, cmap="gray")
            axes[i, 1].set_title("Maska (Ground Truth)")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred_mask, cmap="gray")
            axes[i, 2].set_title("Predykcja SegFormer")
            axes[i, 2].axis("off")

            # Pokazujemy oryginalny obraz z nałożoną mapą atencji typu heatmap
            axes[i, 3].imshow(img_pil)
            axes[i, 3].imshow(attention_map, cmap="jet", alpha=0.5, extent=(0, img_pil.width, img_pil.height, 0))
            axes[i, 3].set_title("Mapa Atencji (Attention)")
            axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig("5_visualize_results.png")
    print("✅ Zapisano wizualizację do: 5_visualize_results.png")
    plt.show()


if __name__ == "__main__":
    generuj_wizualizacje(num_samples=4)