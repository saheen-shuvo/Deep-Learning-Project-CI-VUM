import os
import numpy as np
from medmnist.dataset import PneumoniaMNIST

OUT_DIR = os.path.join("outputs", "medical")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Dataset: PneumoniaMNIST (MedMNIST)")
    print("Task: binary classification (normal vs pneumonia)")

    # Auto-download
    train = PneumoniaMNIST(split="train", download=True, size=28)
    val   = PneumoniaMNIST(split="val", download=True, size=28)
    test  = PneumoniaMNIST(split="test", download=True, size=28)

    X_train, y_train = train.imgs, train.labels
    X_val, y_val     = val.imgs, val.labels
    X_test, y_test   = test.imgs, test.labels

    print("Shapes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    # Save dataset info for report
    with open(os.path.join(OUT_DIR, "dataset_info.txt"), "w", encoding="utf-8") as f:
        f.write("Dataset: PneumoniaMNIST (MedMNIST)\n")
        f.write("Task: Binary classification (normal vs pneumonia)\n")
        f.write(f"Train shape: {X_train.shape}, labels: {y_train.shape}\n")
        f.write(f"Val shape:   {X_val.shape}, labels: {y_val.shape}\n")
        f.write(f"Test shape:  {X_test.shape}, labels: {y_test.shape}\n")
        f.write("Label mapping (binary): 0=normal, 1=pneumonia\n")

    # Save a small sample (optional)
    np.savez_compressed(
        os.path.join(OUT_DIR, "pneumoniamnist_small_sample.npz"),
        X_train=X_train[:200], y_train=y_train[:200],
        X_val=X_val[:100], y_val=y_val[:100],
        X_test=X_test[:100], y_test=y_test[:100],
    )

    print("Done: Download OK. Saved outputs/medical/dataset_info.txt")

if __name__ == "__main__":
    main()
