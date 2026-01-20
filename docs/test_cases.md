# Test Cases (CI-SP)

TC1 Medical dataset download:

- Command: python src\01_medical_prepare.py
- Expected: outputs/medical/dataset_info.txt created
- Result: PASS

TC2 Medical CNN training:

- Command: python src\02_medical_train_cnn.py
- Expected: model + plots + reports in outputs/medical/
- Result: PASS

TC3 Medical Transfer Learning:

- Command: python src\03_medical_train_transfer.py
- Expected: model + plots + reports in outputs/medical/
- Result: PASS

TC4 Sentiment dataset load:

- Command: python src\04_text_prepare.py
- Expected: outputs/sentiment/dataset_info.txt created
- Result: PASS

TC5 Sentiment LSTM:

- Command: python src\05_text_train_lstm.py
- Expected: model + plots + reports in outputs/sentiment/
- Result: PASS

TC6 Sentiment MLP:

- Command: python src\06_text_train_mlp.py
- Expected: model + plots + reports in outputs/sentiment/
- Result: PASS

TC7 Comparison plots:

- Command: python src\07_evaluate_and_plots.py
- Expected: comparison plots saved in outputs/medical and outputs/sentiment
- Result: PASS
