import os, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix

OUT_DIR = os.path.join("outputs", "sentiment")
VOCAB_SIZE = 10000
MAXLEN = 200
BATCH = 64
EPOCHS = 5

def load_split_pad():
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

    val_size = int(0.2 * len(X_train))
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train2, y_train2 = X_train[val_size:], y_train[val_size:]

    X_train2 = pad_sequences(X_train2, maxlen=MAXLEN)
    X_val    = pad_sequences(X_val, maxlen=MAXLEN)
    X_test   = pad_sequences(X_test, maxlen=MAXLEN)

    return (X_train2, y_train2), (X_val, y_val), (X_test, y_test)

def build_lstm():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAXLEN,)),
        tf.keras.layers.Embedding(VOCAB_SIZE, 64),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def save_plots(hist, prefix):
    os.makedirs(OUT_DIR, exist_ok=True)

    plt.figure()
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title(f"{prefix} Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(["train", "val"])
    plt.savefig(os.path.join(OUT_DIR, f"{prefix.lower()}_accuracy.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title(f"{prefix} Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.savefig(os.path.join(OUT_DIR, f"{prefix.lower()}_loss.png"), dpi=200, bbox_inches="tight")
    plt.close()

def eval_and_save(model, X_test, y_test, prefix):
    probs = model.predict(X_test, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(y_test, preds)
    rep = classification_report(y_test, preds)

    with open(os.path.join(OUT_DIR, f"{prefix.lower()}_confusion_matrix.txt"), "w", encoding="utf-8") as f:
        f.write(str(cm))

    with open(os.path.join(OUT_DIR, f"{prefix.lower()}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_split_pad()

    model = build_lstm()
    model.summary()

    hist = model.fit(
        X_train, np.array(y_train),
        validation_data=(X_val, np.array(y_val)),
        epochs=EPOCHS,
        batch_size=BATCH
    )

    model.save(os.path.join(OUT_DIR, "model_sent_lstm.keras"))
    save_plots(hist, prefix="SENT_LSTM")
    eval_and_save(model, X_test, np.array(y_test), prefix="SENT_LSTM")

    with open(os.path.join(OUT_DIR, "sent_lstm_history.json"), "w", encoding="utf-8") as f:
        json.dump(hist.history, f, indent=2)

    print("DONE: Sentiment LSTM finished. Check outputs/sentiment/")

if __name__ == "__main__":
    main()
