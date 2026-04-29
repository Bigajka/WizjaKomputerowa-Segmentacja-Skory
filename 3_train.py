import torch
import numpy as np
from transformers import SegformerForSemanticSegmentation
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler  # <--- NOWOŚĆ: Import do przyspieszenia sprzętowego
from dataset import przygotuj_loadery

# ================================================================
# USTAWIENIA
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 6e-5
MODEL_SAVE_PATH = "segformer_ecu"
# ================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Używam: {device}")


def zaladuj_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    print("✅ Model załadowany (SegFormer-b2)")
    return model


# NOWOŚĆ: Dodajemy scaler jako argument
def trenuj_epoke(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0

    for i, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # NOWOŚĆ: Trening w mieszanej precyzji (AMP)
        with autocast():
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        # NOWOŚĆ: Skalowanie gradientów dla AMP
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if i % 50 == 0:
            print(f"  Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader)


def ewaluuj(model, loader):
    model.eval()
    iou_scores, dice_scores = [], []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # AMP również przyspiesza ewaluację
            with autocast():
                outputs = model(pixel_values=pixel_values)
                logits = torch.nn.functional.interpolate(
                    outputs.logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            preds = logits.argmax(dim=1)

            # NOWOŚĆ: Przyspieszone (wektoryzowane) liczenie metryk na GPU
            pred_bin = (preds == 1)
            label_bin = (labels == 1)

            intersection = (pred_bin & label_bin).sum(dim=(1, 2)).float()
            union = (pred_bin | label_bin).sum(dim=(1, 2)).float()

            iou = (intersection / (union + 1e-8)).cpu().numpy()
            dice = (2 * intersection / (pred_bin.sum(dim=(1, 2)) + label_bin.sum(dim=(1, 2)) + 1e-8)).cpu().numpy()

            iou_scores.extend(iou)
            dice_scores.extend(dice)

    return {
        "IoU": np.mean(iou_scores),
        "Dice": np.mean(dice_scores)
    }


if __name__ == "__main__":
    print("=" * 50)
    print("TRENING SEGFORMERA (ZOPTYMALIZOWANY)")
    print("=" * 50)

    train_loader, test_loader = przygotuj_loadery(batch_size=BATCH_SIZE)
    model = zaladuj_model()

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # NOWOŚĆ: Inicjalizacja Scalera dla AMP
    scaler = GradScaler()

    historia = {"loss": [], "iou": [], "dice": []}
    best_iou = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'=' * 40}")
        print(f"Epoka {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'=' * 40}")

        # Przekazujemy scaler do funkcji trenującej
        loss = trenuj_epoke(model, train_loader, optimizer, scaler)
        metryki = ewaluuj(model, test_loader)
        scheduler.step()

        historia["loss"].append(loss)
        historia["iou"].append(metryki["IoU"])
        historia["dice"].append(metryki["Dice"])

        print(f"\n📊 Wyniki epoki {epoch + 1}:")
        print(f"   Loss:  {loss:.4f}")
        print(f"   IoU:   {metryki['IoU']:.4f}")
        print(f"   Dice:  {metryki['Dice']:.4f}")

        if metryki["IoU"] > best_iou:
            best_iou = metryki["IoU"]
            model.save_pretrained(MODEL_SAVE_PATH)
            print(f"   💾 Zapisano nowy najlepszy model! (IoU: {best_iou:.4f})")

    print(f"\n✅ Trening zakończony! Najlepszy IoU: {best_iou:.4f}")
    print(f"✅ Model zapisany w: {MODEL_SAVE_PATH}/")

    np.save("historia_treningu.npy", historia)
    print("✅ Historia treningu zapisana jako: historia_treningu.npy")