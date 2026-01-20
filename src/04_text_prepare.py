import os
from tensorflow.keras.datasets import imdb

OUT_DIR = os.path.join("outputs", "sentiment")
VOCAB_SIZE = 10000

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

    val_size = int(0.2 * len(X_train))
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train2, y_train2 = X_train[val_size:], y_train[val_size:]

    with open(os.path.join(OUT_DIR, "dataset_info.txt"), "w", encoding="utf-8") as f:
        f.write("Dataset: IMDB Reviews (Keras built-in)\n")
        f.write(f"Vocab size: {VOCAB_SIZE}\n")
        f.write(f"Train size: {len(X_train2)}\n")
        f.write(f"Val size:   {len(X_val)}\n")
        f.write(f"Test size:  {len(X_test)}\n")

    print("DONE: IMDB loaded.")
    print("Train/Val/Test sizes:", len(X_train2), len(X_val), len(X_test))

if __name__ == "__main__":
    main()
