# CI-SP Report (CIS6005)

## 1. Introduction

This project includes:

- Medical image classification (PneumoniaMNIST)
- Sentiment analysis (IMDB movie reviews)

## 2. Dataset Preparation

### 2.1 Medical Dataset (PneumoniaMNIST - MedMNIST)

- Downloaded using `medmnist` library.
- Data split used: train / validation / test (provided by dataset).
- Preprocessing:
  - CNN: normalized to [0,1], shape (28,28,1)
  - Transfer learning: grayscale→RGB, resized to 224x224, MobileNetV2 preprocess.

### 2.2 Sentiment Dataset (IMDB - Keras built-in)

- Loaded from `tensorflow.keras.datasets.imdb`
- Train split was further split into train(80%) and validation(20%).
- Sequences padded to fixed length (MAXLEN=200).

## 3. Models and Hyperparameters

### Medical Models

- Model A: CNN (Conv2D layers + pooling + dense)
- Model B: Transfer learning (MobileNetV2 frozen + GAP + dense)

Hyperparameters:

- CNN: Adam(1e-3), batch=64, epochs=8, loss=binary_crossentropy
- Transfer: Adam(1e-3), batch=32, epochs=5, loss=binary_crossentropy

### Sentiment Models

- Model A: LSTM (Embedding + LSTM + Dense)
- Model B: MLP (Embedding + GlobalAveragePooling + Dense)

Hyperparameters:

- LSTM: Adam(1e-3), batch=64, epochs=5, MAXLEN=200, vocab=10000
- MLP: Adam(1e-3), batch=64, epochs=5, MAXLEN=200, vocab=10000

## 4. Evaluation and Results

Medical results are saved in `outputs/medical/`:

- accuracy/loss plots for CNN and Transfer learning
- confusion matrix + classification report
- comparison plots from `07_evaluate_and_plots.py`

Sentiment results are saved in `outputs/sentiment/`:

- accuracy/loss plots for LSTM and MLP
- confusion matrix + classification report
- comparison plots from `07_evaluate_and_plots.py`

## 5. Conclusion

Transfer learning performed strongly on the medical dataset.  
For sentiment, LSTM and MLP both worked, with different training behavior.  
Using validation data helped choose models and prevent overfitting.
