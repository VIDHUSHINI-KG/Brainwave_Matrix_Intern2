# 💳 Credit Card Fraud Detection

This project builds a machine learning model to detect fraudulent credit card transactions. The dataset used is highly imbalanced, so techniques like SMOTE are employed to handle the skewed class distribution. The model aims to correctly identify fraudulent transactions with high precision and recall.

---

## 📁 Dataset

* **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Size**: 284,807 transactions
* **Features**: 30 numerical features (PCA-transformed), `Time`, `Amount`, and `Class` (target)
* **Target**:

  * `0` → Legitimate transaction
  * `1` → Fraudulent transaction

---

## 🛠️ Tech Stack

| Category             | Tools / Libraries                                                           |
| -------------------- | --------------------------------------------------------------------------- |
| Programming Language | Python                                                                      |
| Data Manipulation    | pandas, numpy                                                               |
| Visualization        | matplotlib, seaborn                                                         |
| Machine Learning     | scikit-learn (Logistic Regression, Random Forest), imbalanced-learn (SMOTE) |
| Model Evaluation     | confusion\_matrix, classification\_report, ROC-AUC                          |

---

## 🔍 Problem Statement

Fraudulent transactions are rare (\~0.17%) in this dataset, making it a classic example of an imbalanced classification problem. Traditional models tend to be biased towards the majority class (legitimate transactions), which makes detection of frauds difficult. This project solves this using data balancing techniques and robust models.

---

## 🧪 Approach

1. **Data Preprocessing**

   * Dropped irrelevant `Time` feature
   * Scaled `Amount` using `StandardScaler`

2. **Class Imbalance Handling**

   * Applied **SMOTE** to oversample the minority class (fraud)

3. **Modeling**

   * Used **Random Forest Classifier** as the baseline
   * Can also try Logistic Regression or XGBoost

4. **Evaluation Metrics**

   * **Confusion Matrix**
   * **Precision, Recall, F1-score**
   * **ROC-AUC Score**

---

## 📊 Results

* Achieved high **Recall** and **ROC-AUC** for fraud detection
* Class imbalance handled effectively with **SMOTE**

---

## 📌 Conclusion

This project demonstrates a practical machine learning pipeline for handling imbalanced data and detecting credit card fraud. It highlights the importance of data preprocessing, class balancing, and choosing the right evaluation metrics.

---

## 🤝 Acknowledgements

* Dataset: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

