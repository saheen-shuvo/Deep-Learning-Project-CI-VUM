import os, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from medmnist.dataset import PneumoniaMNIST

OUT_DIR = os.path.join("outputs", "medical")
SEED = 42
BATCH = 64
EPOCHS = 8

def make_tf_dataset(X, y, training: bool):
    X = X.astype("float32") / 255.0
    X = np.expand_dims(X, -1)  # (N, 28, 28, 1)
    y = y.astype("int32").reshape(-1)

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(2000, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

def build_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

def save_plots(hist, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title(f"{prefix} Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(["train", "val"])
    plt.savefig(os.path.join(out_dir, f"{prefix.lower()}_accuracy.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title(f"{prefix} Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.savefig(os.path.join(out_dir, f"{prefix.lower()}_loss.png"), dpi=200, bbox_inches="tight")
    plt.close()

def eval_and_save(model, test_ds, out_dir, prefix):
    y_true, y_pred = [], []

    for xb, yb in test_ds:
        p = model.predict(xb, verbose=0).ravel()
        y_true.extend(yb.numpy().tolist())
        y_pred.extend((p >= 0.5).astype(int).tolist())

    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred)

    with open(os.path.join(out_dir, f"{prefix.lower()}_confusion_matrix.txt"), "w", encoding="utf-8") as f:
        f.write(str(cm))

    with open(os.path.join(out_dir, f"{prefix.lower()}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train = PneumoniaMNIST(split="train", download=True, size=28)
    val   = PneumoniaMNIST(split="val", download=True, size=28)
    test  = PneumoniaMNIST(split="test", download=True, size=28)

    train_ds = make_tf_dataset(train.imgs, train.labels, training=True)
    val_ds   = make_tf_dataset(val.imgs, val.labels, training=False)
    test_ds  = make_tf_dataset(test.imgs, test.labels, training=False)

    model = build_cnn()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.summary()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    model.save(os.path.join(OUT_DIR, "model_med_cnn.keras"))
    save_plots(hist, OUT_DIR, prefix="MED_CNN")
    eval_and_save(model, test_ds, OUT_DIR, prefix="MED_CNN")

    with open(os.path.join(OUT_DIR, "med_cnn_history.json"), "w", encoding="utf-8") as f:
        json.dump(hist.history, f, indent=2)

    print("DONE: Medical CNN done. Check outputs/medical/")

if __name__ == "__main__":
    main()
