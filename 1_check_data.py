import subprocess
import os
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ================================================================
REPO_URL = "https://github.com/Bigajka/WizjaKomputerowa-Segmentacja-Skory.git"
REPO_NAME = "WizjaKomputerowa-Segmentacja-Skory"
DATASET_PATH = os.path.join(REPO_NAME, "dataset_final")
# ================================================================


def pobierz_repo():
    if not os.path.exists(REPO_NAME):
        print("⬇️  Pobieram repozytorium z GitHuba (może chwilę potrwać)...")
        subprocess.run(["git", "clone", REPO_URL], check=True)
        print("✅ Pobrano repozytorium!")
    else:
        print("✅ Repozytorium już istnieje lokalnie, pomijam pobieranie.")


def sprawdz_dataset(root_dir):
    root = Path(root_dir)
    wszystkie_pary = []

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        img_dir = folder / "images"
        mask_dir = folder / "masks"

        if not img_dir.exists() or not mask_dir.exists():
            print(f"⚠️  Brak folderu images/ lub masks/ w {folder.name}")
            continue

        for img_path in sorted(img_dir.glob("*.png")):
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                wszystkie_pary.append((img_path, mask_path))
            else:
                print(f"⚠️  Brak maski dla: {img_path.name}")

    print(f"\n✅ Znaleziono {len(wszystkie_pary)} par obraz-maska")
    return wszystkie_pary


def pokaz_przyklad(pary, ile=3):
    fig, axes = plt.subplots(ile, 2, figsize=(10, 5 * ile))

    for i in range(min(ile, len(pary))):
        img_path, mask_path = pary[i]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)

        print(f"\nObraz: {img_path.name}")
        print(f"  Rozmiar obrazu: {img.size}")
        print(f"  Rozmiar maski:  {mask.size}")
        print(f"  Unikalne wartości w masce: {np.unique(mask_np)}")
        print(f"  % skóry w masce: {(mask_np > 127).mean() * 100:.1f}%")

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Obraz: {img_path.name}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_np, cmap="gray")
        axes[i, 1].set_title(f"Maska: {mask_path.name}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("sprawdzenie_danych.png")
    print("\n✅ Zapisano podgląd jako: sprawdzenie_danych.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 50)
    print("SPRAWDZANIE DATASETU")
    print("=" * 50)

    pobierz_repo()
    pary = sprawdz_dataset(DATASET_PATH)

    if len(pary) == 0:
        print("❌ Nie znaleziono żadnych danych!")
    else:
        pokaz_przyklad(pary, ile=3)
        print("\n✅ Dataset wygląda poprawnie. Możesz przejść do kroku 2.")