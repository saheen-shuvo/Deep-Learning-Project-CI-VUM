# CIS6005 – CI-SP Coursework

This repository contains the **CI-SP coursework submission** for the module **CIS6005 – Computational Intelligence**.  
The project applies **neural network–based computational intelligence techniques** to solve two real-world problems:

1. **Medical Image Classification**
2. **Sentiment Analysis**

Multiple models are implemented, trained, evaluated, and compared for each task.

---

## 📌 Project Overview

### 1️⃣ Medical Image Classification

- **Dataset:** PneumoniaMNIST (MedMNIST)
- **Task:** Binary classification (Normal vs Pneumonia)
- **Models Implemented:**
  - Custom Convolutional Neural Network (CNN)
  - Transfer Learning using MobileNetV2 (pretrained on ImageNet)
- **Evaluation:**
  - Accuracy & loss curves
  - Confusion matrix
  - Classification report
  - Model comparison plots

📂 Outputs saved in: `outputs/medical/`

---

### 2️⃣ Sentiment Analysis

- **Dataset:** IMDB Movie Reviews (Keras built-in dataset)
- **Task:** Binary sentiment classification (Positive vs Negative)
- **Models Implemented:**
  - Long Short-Term Memory (LSTM)
  - MLP-style model (Embedding + Global Average Pooling + Dense)
- **Evaluation:**
  - Accuracy & loss curves
  - Confusion matrix
  - Classification report
  - Model comparison plots

📂 Outputs saved in: `outputs/sentiment/`

---

## 🗂️ Project Structure

CI_SP/
│
├── src/                          # Source code
│   ├── 01_medical_prepare.py         # Download & prepare PneumoniaMNIST
│   ├── 02_medical_train_cnn.py       # Medical CNN model training
│   ├── 03_medical_train_transfer.py  # Transfer learning (MobileNetV2)
│   ├── 04_text_prepare.py            # IMDB dataset preparation
│   ├── 05_text_train_lstm.py         # Sentiment analysis using LSTM
│   ├── 06_text_train_mlp.py          # Sentiment analysis using MLP
│   └── 07_evaluate_and_plots.py      # Model comparison & evaluation plots
│
├── outputs/                      # Generated results
│   ├── medical/                     # Medical model outputs
│   │   ├── *.png                    # Accuracy & loss plots
│   │   ├── *_history.json           # Training history
│   │   ├── *_confusion_matrix.txt   # Confusion matrices
│   │   └── *_classification_report.txt
│   │
│   └── sentiment/                   # Sentiment model outputs
│       ├── *.png
│       ├── *_history.json
│       ├── *_confusion_matrix.txt
│       └── *_classification_report.txt
│
├── docs/                         # Documentation
│   ├── report.docx                  # Coursework report
│   ├── test_cases.md                # Test cases
│   ├── user_manual.md               # User manual
│   ├── user_manual.pdf              # User manual (PDF)
│   ├── architecture.png             # System architecture diagram
│   └── workflow.png                 # Workflow diagram
│
├── README.md                     # Project overview (GitHub)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── ci_env/                       # Virtual environment (ignored in Git)


---

## ⚙️ Setup Instructions (Windows)

````bash
cd /d "E:\CI Project VUM\CI_SP"
ci_env\Scripts\activate
pip install -r requirements.txt

<!-- # CIS6005 CI-SP Submission

## Overview

This submission contains two tasks:

### 1: Medical Image Classification (PneumoniaMNIST - MedMNIST)

- Model A: Custom CNN (from scratch)
- Model B: Transfer Learning (MobileNetV2 pretrained)

Outputs saved in: `outputs/medical/`

### 2: Sentiment Analysis (IMDB - Keras built-in dataset)

- Model A: LSTM
- Model B: MLP-style (Embedding + GlobalAveragePooling + Dense)

Outputs saved in: `outputs/sentiment/`

## Setup (Windows)

```bash
cd /d "E:\CI Project VUM\CI_SP"
ci_env\Scripts\activate
pip install -r requirements.txt
``` -->
````
