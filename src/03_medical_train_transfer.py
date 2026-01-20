import os, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from medmnist.dataset import PneumoniaMNIST

OUT_DIR = os.path.join("outputs", "medical")
SEED = 42
BATCH = 32
EPOCHS = 5

def preprocess_for_mobilenet(X, y):
    X = X.astype("float32")
    X = np.expand_dims(X, -1)      # (N,28,28,1)
    X = np.repeat(X, 3, axis=-1)   # (N,28,28,3)
    y = y.astype("int32").reshape(-1)
    return X, y

def make_tf_dataset(X, y, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    def _map(x, y):
        x = tf.image.resize(x, (224, 224))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # expects [0..255] RGB
        return x, y

    if training:
        ds = ds.shuffle(2000, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

def build_transfer():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

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

    X_train, y_train = preprocess_for_mobilenet(train.imgs, train.labels)
    X_val, y_val     = preprocess_for_mobilenet(val.imgs, val.labels)
    X_test, y_test   = preprocess_for_mobilenet(test.imgs, test.labels)

    train_ds = make_tf_dataset(X_train, y_train, training=True)
    val_ds   = make_tf_dataset(X_val, y_val, training=False)
    test_ds  = make_tf_dataset(X_test, y_test, training=False)

    model = build_transfer()
    model.summary()

    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    model.save(os.path.join(OUT_DIR, "model_med_transfer.keras"))
    save_plots(hist, OUT_DIR, prefix="MED_TRANSFER")
    eval_and_save(model, test_ds, OUT_DIR, prefix="MED_TRANSFER")

    with open(os.path.join(OUT_DIR, "med_transfer_history.json"), "w", encoding="utf-8") as f:
        json.dump(hist.history, f, indent=2)

    print("DONE: Medical Transfer model finished. Check outputs/medical/")

if __name__ == "__main__":
    main()
