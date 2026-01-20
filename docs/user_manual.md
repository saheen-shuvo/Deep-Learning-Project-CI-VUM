# User Manual

## 1. Setup

1. Open CMD
2. Go to project:
   cd /d "E:\CI Project VUM\CI_SP"
3. Activate environment:
   ci_env\Scripts\activate
4. Install packages:
   pip install -r requirements.txt

## 2. Run Medical

python -u src\01_medical_prepare.py
python -u src\02_medical_train_cnn.py
python -u src\03_medical_train_transfer.py
Results: outputs/medical/

## 3. Run Sentiment

python -u src\04_text_prepare.py
python -u src\05_text_train_lstm.py
python -u src\06_text_train_mlp.py
Results: outputs/sentiment/

## 4. Compare Plots

python -u src\07_evaluate_and_plots.py
