# credit_card_fraud_detection
# 💳 Credit Card Fraud Detection with Decision Tree

This project implements a machine learning pipeline to detect fraudulent credit card transactions using a **Decision Tree Classifier**. The dataset is highly imbalanced and contains real transactions made by European cardholders in 2013.

---

## 📂 Dataset

- **File**: `creditcard.csv`
- **Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Rows**: 284,807 transactions
- **Target column**: `Class` (0 = genuine, 1 = fraud)
- **Features**: 30 (28 anonymized via PCA + `Time` and `Amount`)

---

## ⚙️ Technologies Used

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

---

## 🚀 How to Run the Project

1. Make sure you have Python installed.
2. Install required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn

