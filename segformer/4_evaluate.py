import torch
import numpy as np
from transformers import SegformerForSemanticSegmentation
from dataset import przygotuj_loadery

MODEL_PATH = "segformer_ecu"
BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ewaluuj_model():
    print(f"Ładowanie modelu z: {MODEL_PATH}...")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    _, test_loader = przygotuj_loadery(batch_size=BATCH_SIZE)

    iou_scores = []
    dice_scores = []
    accuracy_scores = []

    print("\nRozpoczynam ewaluację na zbiorze testowym...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)

            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            preds = logits.argmax(dim=1)

            for pred, label in zip(preds, labels):
                pred_bin = (pred == 1)
                label_bin = (label == 1)

                intersection = (pred_bin & label_bin).sum().float()
                union = (pred_bin | label_bin).sum().float()

                iou = (intersection / (union + 1e-8)).item()
                dice = (2 * intersection / (pred_bin.sum() + label_bin.sum() + 1e-8)).item()
                accuracy = (pred_bin == label_bin).float().mean().item()

                iou_scores.append(iou)
                dice_scores.append(dice)
                accuracy_scores.append(accuracy)

    print("\n" + "=" * 40)
    print("🎯 WYNIKI EWALUACJI NA ZBIORZE TESTOWYM")
    print("=" * 40)
    print(f"Średnie IoU:      {np.mean(iou_scores):.4f}")
    print(f"Średni Dice:      {np.mean(dice_scores):.4f}")
    print(f"Pixel Accuracy:   {np.mean(accuracy_scores):.4f}")
    print("=" * 40)


if __name__ == "__main__":
    ewaluuj_model()