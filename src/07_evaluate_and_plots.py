import os
import json
import matplotlib.pyplot as plt

MED_DIR = os.path.join("outputs", "medical")
SENT_DIR = os.path.join("outputs", "sentiment")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_plot(history, key, label):
    if key in history:
        plt.plot(history[key], label=label)

def compare_histories(out_path, title, histories):
    """
    histories: list of tuples -> (history_dict, label_prefix)
    Plots train/val accuracy and train/val loss comparisons.
    """
    # Accuracy comparison
    plt.figure()
    for h, name in histories:
        safe_plot(h, "accuracy", f"{name} train acc")
        safe_plot(h, "val_accuracy", f"{name} val acc")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(out_path.replace(".png", "_accuracy_compare.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Loss comparison
    plt.figure()
    for h, name in histories:
        safe_plot(h, "loss", f"{name} train loss")
        safe_plot(h, "val_loss", f"{name} val loss")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_path.replace(".png", "_loss_compare.png"), dpi=200, bbox_inches="tight")
    plt.close()

def main():
    # Medical histories
    med_histories = []
    cnn_path = os.path.join(MED_DIR, "med_cnn_history.json")
    tr_path  = os.path.join(MED_DIR, "med_transfer_history.json")

    if os.path.exists(cnn_path):
        med_histories.append((load_json(cnn_path), "CNN"))
    if os.path.exists(tr_path):
        med_histories.append((load_json(tr_path), "Transfer"))

    if med_histories:
        compare_histories(
            out_path=os.path.join(MED_DIR, "medical_compare.png"),
            title="Medical (PneumoniaMNIST)",
            histories=med_histories
        )
        print("DONE: Medical comparison plots saved in outputs/medical/")
    else:
        print("WARN: No medical histories found.")

    # Sentiment histories
    sent_histories = []
    lstm_path = os.path.join(SENT_DIR, "sent_lstm_history.json")
    mlp_path  = os.path.join(SENT_DIR, "sent_mlp_history.json")

    if os.path.exists(lstm_path):
        sent_histories.append((load_json(lstm_path), "LSTM"))
    if os.path.exists(mlp_path):
        sent_histories.append((load_json(mlp_path), "MLP"))

    if sent_histories:
        compare_histories(
            out_path=os.path.join(SENT_DIR, "sentiment_compare.png"),
            title="Sentiment (IMDB)",
            histories=sent_histories
        )
        print("DONE: Sentiment comparison plots saved in outputs/sentiment/")
    else:
        print("WARN: No sentiment histories found.")

if __name__ == "__main__":
    main()
